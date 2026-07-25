"""
Automated PDF report generation.
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER


def _get_styles():
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(name="MainTitle", parent=styles["Heading1"], fontSize=22,
                              textColor=colors.HexColor("#1E293B"), spaceAfter=6, alignment=TA_CENTER,
                              fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="SubTitle", parent=styles["Normal"], fontSize=11,
                              textColor=colors.HexColor("#64748B"), spaceAfter=20, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="SectionHeader", parent=styles["Heading2"], fontSize=14,
                              textColor=colors.HexColor("#4338CA"), spaceBefore=16, spaceAfter=8,
                              fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="ReportBody", parent=styles["Normal"], fontSize=10,
                              textColor=colors.HexColor("#334155"), spaceAfter=6, leading=14))
    styles.add(ParagraphStyle(name="Footer", parent=styles["Normal"], fontSize=8,
                              textColor=colors.HexColor("#94A3B8"), alignment=TA_CENTER))
    return styles


def generate_pdf_report(output_path, dataset_meta, model_name, metrics, feature_importance,
                        cv_mean=None, cv_std=None, classification_report=None, extra_notes=""):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.7*inch, bottomMargin=0.7*inch)
    
    styles = _get_styles()
    story = []
    
    story.append(Paragraph("Predictive Analytics Dashboard", styles["MainTitle"]))
    story.append(Paragraph("Automated Model Performance Report", styles["SubTitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#6366F1"), spaceAfter=12))
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_data = [
        ["Generated", now],
        ["Dataset", dataset_meta.get("name", "N/A")],
        ["Task Type", dataset_meta.get("task", "N/A").title()],
        ["Model", model_name],
        ["Samples", str(dataset_meta.get("n_samples", "N/A"))],
        ["Features", str(dataset_meta.get("n_features", "N/A"))],
    ]
    
    meta_table = Table(meta_data, colWidths=[1.8*inch, 4.5*inch])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#475569")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 16))
    
    story.append(Paragraph("Dataset Description", styles["SectionHeader"]))
    story.append(Paragraph(dataset_meta.get("description", "No description available."), styles["ReportBody"]))
    
    story.append(Paragraph("Model Performance Metrics", styles["SectionHeader"]))
    
    metric_rows = [["Metric", "Value"]]
    for k, v in metrics.items():
        metric_rows.append([k, f"{v:.4f}"])
    if cv_mean is not None:
        metric_rows.append(["Cross-Val Mean", f"{cv_mean:.4f}"])
        if cv_std is not None:
            metric_rows.append(["Cross-Val Std", f"{cv_std:.4f}"])
    
    m_table = Table(metric_rows, colWidths=[3.2*inch, 2.2*inch])
    m_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6366F1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F1F5F9"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(m_table)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Top Feature Importance", styles["SectionHeader"]))
    
    fi_rows = [["Rank", "Feature", "Importance"]]
    for i, row in feature_importance.head(12).iterrows():
        fi_rows.append([str(i+1), str(row["feature"])[:40], f"{row['importance']:.5f}"])
    
    fi_table = Table(fi_rows, colWidths=[0.7*inch, 3.5*inch, 1.5*inch])
    fi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8B5CF6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F3FF"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDD6FE")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(fi_table)
    
    if classification_report:
        story.append(Paragraph("Detailed Classification Report", styles["SectionHeader"]))
        cr_rows = [["Class", "Precision", "Recall", "F1-Score", "Support"]]
        for cls, vals in classification_report.items():
            if isinstance(vals, dict) and "precision" in vals:
                cr_rows.append([
                    str(cls),
                    f"{vals['precision']:.3f}",
                    f"{vals['recall']:.3f}",
                    f"{vals['f1-score']:.3f}",
                    str(int(vals.get("support", 0))),
                ])
        
        cr_table = Table(cr_rows, colWidths=[1.4*inch, 1.1*inch, 1.1*inch, 1.1*inch, 1.0*inch])
        cr_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0EA5E9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#E0F2FE"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BAE6FD")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(cr_table)
    
    if extra_notes:
        story.append(Paragraph("Additional Notes", styles["SectionHeader"]))
        story.append(Paragraph(extra_notes, styles["ReportBody"]))
    
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=8))
    story.append(Paragraph("Generated by Predictive Analytics Dashboard", styles["Footer"]))
    
    doc.build(story)
    return output_path