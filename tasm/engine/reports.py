"""
PDF Report Generator for TASM analysis results.
Generates a self-contained report with metrics and embedded plots.
"""

import os
import io
import base64
import time
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable,
)
from reportlab.lib import colors

# ─── Colors ──────────────────────────────────────────────────────
C_BG = HexColor("#0d1117")
C_DARK = HexColor("#161b22")
C_BORDER = HexColor("#30363d")
C_TEXT = HexColor("#c9d1d9")
C_HEAD = HexColor("#f0f6fc")
C_DIM = HexColor("#8b949e")
C_GREEN = HexColor("#2d936c")
C_RED = HexColor("#c44536")
C_BLUE = HexColor("#4a6fa5")
C_PURPLE = HexColor("#7b2d8b")
C_ORANGE = HexColor("#e0a458")
C_WHITE = HexColor("#ffffff")
C_BLACK = HexColor("#000000")

# ─── Styles ──────────────────────────────────────────────────────
STYLES = {
    "title": ParagraphStyle(
        "title", fontName="Helvetica-Bold", fontSize=18,
        textColor=C_BLACK, spaceAfter=4, alignment=TA_LEFT,
    ),
    "subtitle": ParagraphStyle(
        "subtitle", fontName="Helvetica", fontSize=9,
        textColor=C_DIM, spaceAfter=16, alignment=TA_LEFT,
    ),
    "heading": ParagraphStyle(
        "heading", fontName="Helvetica-Bold", fontSize=12,
        textColor=C_BLACK, spaceBefore=14, spaceAfter=6,
    ),
    "body": ParagraphStyle(
        "body", fontName="Helvetica", fontSize=9,
        textColor=C_BLACK, spaceAfter=6, leading=13,
    ),
    "mono": ParagraphStyle(
        "mono", fontName="Courier", fontSize=8,
        textColor=C_BLACK, spaceAfter=4, leading=11,
    ),
    "label": ParagraphStyle(
        "label", fontName="Helvetica", fontSize=7,
        textColor=C_DIM, spaceAfter=1,
    ),
    "value": ParagraphStyle(
        "value", fontName="Helvetica-Bold", fontSize=14,
        textColor=C_BLACK, spaceAfter=2,
    ),
}

REPORTS_DIR = Path("reports")


def ensure_reports_dir():
    REPORTS_DIR.mkdir(exist_ok=True)


def _b64_to_image(b64_str: str, width=None, height=None):
    """Convert a base64-encoded PNG to a reportlab Image."""
    if not b64_str:
        return None
    img_data = base64.b64decode(b64_str)
    buf = io.BytesIO(img_data)
    img = Image(buf)
    if width:
        ratio = img.imageHeight / img.imageWidth
        img.drawWidth = width
        img.drawHeight = width * ratio
    elif height:
        ratio = img.imageWidth / img.imageHeight
        img.drawHeight = height
        img.drawWidth = height * ratio
    return img


def _fmt(v, fmt=".4f"):
    if v is None:
        return "--"
    if isinstance(v, str):
        return v
    try:
        return format(v, fmt)
    except (TypeError, ValueError):
        return str(v)


def _pct(v):
    if v is None:
        return "--"
    return f"{v*100:.1f}%"


def _ln_label(v):
    if v is None:
        return ""
    sign = "+" if v > 0 else ""
    return f"  ({sign}{v:.2f} sd)"


def generate_single_report(result_dict: dict, plots: dict,
                           model_name: str = "") -> str:
    """
    Generate a PDF report for a single prompt analysis.
    Returns the file path.
    """
    ensure_reports_dir()
    r = result_dict

    # Filename: sanitized prompt + timestamp
    safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in r["prompt"][:40]).strip()
    safe_prompt = safe_prompt.replace(" ", "_") or "analysis"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_prompt}_{timestamp}.pdf"
    filepath = REPORTS_DIR / filename

    doc = SimpleDocTemplate(
        str(filepath), pagesize=letter,
        leftMargin=0.6*inch, rightMargin=0.6*inch,
        topMargin=0.6*inch, bottomMargin=0.6*inch,
    )
    story = []
    usable_width = letter[0] - 1.2*inch

    # ─── Header ──────────────────────────────────────────────
    story.append(Paragraph("TASM Analysis Report", STYLES["title"]))
    meta = f"Model: {model_name}" if model_name else ""
    meta += f"  |  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    story.append(Paragraph(meta, STYLES["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BORDER))
    story.append(Spacer(1, 8))

    # ─── Prompt ──────────────────────────────────────────────
    story.append(Paragraph("Prompt", STYLES["heading"]))
    story.append(Paragraph(r["prompt"], STYLES["mono"]))
    info_parts = [f"Tokens: {r['seq_len']}"]
    if r.get("category"):
        info_parts.append(f"Category: {r['category']}")
    story.append(Paragraph(" | ".join(info_parts), STYLES["label"]))
    story.append(Spacer(1, 8))

    # ─── Metrics Table ───────────────────────────────────────
    story.append(Paragraph("Distribution Metrics", STYLES["heading"]))

    metrics_data = [
        ["Metric", "Value", "Length-Normalized"],
        ["Stress Score", _fmt(r["stress_score"]),
         _ln_label(r.get("stress_score_ln"))],
        ["Net Correction", _fmt(r["net_correction"]), ""],
        ["Entropy", _fmt(r["entropy"]),
         _ln_label(r.get("entropy_ln"))],
        ["Gini", _fmt(r["gini"]), ""],
        ["Boundary Share", _pct(r["top2_share"]),
         _ln_label(r.get("top2_share_ln"))],
        ["Interior Share", _pct(r["middle_share"]),
         _ln_label(r.get("middle_share_ln"))],
        ["Interior CV", _fmt(r["interior_cv"]), ""],
    ]
    if r.get("kl_divergence") is not None:
        metrics_data.append(
            ["KL Divergence", _fmt(r["kl_divergence"]), ""])

    neg_row = [
        "Negative Tokens",
        f"{r['n_negative_tokens']}/{r['seq_len']}",
        "detected" if r["has_negative_tokens"] else "none",
    ]
    metrics_data.append(neg_row)

    t = Table(metrics_data, colWidths=[1.8*inch, 1.4*inch, 1.6*inch])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), C_BLACK),
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e8ecf0")),
        ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # ─── Per-Token Table ─────────────────────────────────────
    if r.get("tokens") and r.get("signed_attr"):
        story.append(Paragraph("Per-Token Attribution", STYLES["heading"]))

        tok_data = [["Token", "Signed Attr", "Stress"]]
        attrs = r["signed_attr"]
        stresses = r.get("per_token_stress", [])
        for i, tok in enumerate(r["tokens"]):
            attr_val = attrs[i] if i < len(attrs) else 0
            stress_val = stresses[i] if i < len(stresses) else 0
            tok_data.append([
                tok.strip(),
                f"{attr_val:+.4f}",
                _fmt(stress_val),
            ])

        col_w = [2.0*inch, 1.2*inch, 1.2*inch]
        tt = Table(tok_data, colWidths=col_w, repeatRows=1)
        tt.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Courier"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, -1), C_BLACK),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e8ecf0")),
            ("GRID", (0, 0), (-1, -1), 0.4, C_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(tt)
        story.append(Spacer(1, 8))

    # ─── Plots ───────────────────────────────────────────────
    plot_order = [
        ("signed_attribution", "Signed Attribution"),
        ("stress_per_token", "Focused Stress Score"),
        ("distribution_metrics", "Distribution Metrics"),
        ("amplitude_trajectory", "Amplitude Trajectory"),
        ("heatmap", "Sensitivity Heatmap"),
    ]

    for key, label in plot_order:
        b64 = plots.get(key)
        if b64:
            story.append(Paragraph(label, STYLES["heading"]))
            img = _b64_to_image(b64, width=usable_width)
            if img:
                story.append(img)
            story.append(Spacer(1, 6))

    # Build
    doc.build(story)
    return str(filepath)


def generate_batch_report(aggregate: dict, per_prompt: list,
                          plots: dict, model_name: str = "",
                          n_prompts: int = 0) -> str:
    """
    Generate a PDF report for batch analysis.
    Returns the file path.
    """
    ensure_reports_dir()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"batch_{n_prompts}prompts_{timestamp}.pdf"
    filepath = REPORTS_DIR / filename

    doc = SimpleDocTemplate(
        str(filepath), pagesize=letter,
        leftMargin=0.6*inch, rightMargin=0.6*inch,
        topMargin=0.6*inch, bottomMargin=0.6*inch,
    )
    story = []
    usable_width = letter[0] - 1.2*inch

    # ─── Header ──────────────────────────────────────────────
    story.append(Paragraph("TASM Batch Analysis Report", STYLES["title"]))
    meta = f"{n_prompts} prompts"
    if model_name:
        meta += f"  |  Model: {model_name}"
    meta += f"  |  {time.strftime('%Y-%m-%d %H:%M:%S')}"
    story.append(Paragraph(meta, STYLES["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BORDER))
    story.append(Spacer(1, 8))

    # ─── Category Summary Table ──────────────────────────────
    cats = aggregate.get("categories", {})
    if cats:
        story.append(Paragraph("Category Summary", STYLES["heading"]))

        cat_data = [["Category", "N", "Avg Length", "Stress", "Entropy",
                      "Boundary%", "Interior%", "Net", "Neg Rate"]]
        for cat_name in ["benign", "mild", "harmful", "jailbreak"]:
            if cat_name not in cats:
                continue
            s = cats[cat_name]
            m = s.get("metrics", {})
            cat_data.append([
                cat_name.title(),
                str(s["n"]),
                _fmt(s.get("mean_seq_len"), ".1f"),
                _fmt(m.get("stress_score", {}).get("estimate")),
                _fmt(m.get("entropy", {}).get("estimate")),
                _pct(m.get("top2_share", {}).get("estimate")),
                _pct(m.get("middle_share", {}).get("estimate")),
                _fmt(m.get("net_correction", {}).get("estimate")),
                _pct(s.get("negative_token_rate")),
            ])

        t = Table(cat_data, repeatRows=1)
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, -1), C_BLACK),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e8ecf0")),
            ("GRID", (0, 0), (-1, -1), 0.4, C_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

    # ─── Separability Table ──────────────────────────────────
    sep = aggregate.get("separability", {})
    if sep:
        story.append(Paragraph("Separability: Benign vs Harmful", STYLES["heading"]))

        sep_data = [["Metric", "Cohen's d", "95% CI", "Best Acc"]]
        for metric, s in sep.items():
            es = s.get("effect_size", {})
            thr = s.get("threshold", {})
            sep_data.append([
                metric.replace("_", " ").title(),
                _fmt(es.get("estimate"), ".2f"),
                f"[{_fmt(es.get('ci_low'), '.2f')}, {_fmt(es.get('ci_high'), '.2f')}]",
                _pct(thr.get("accuracy")),
            ])

        t = Table(sep_data, repeatRows=1)
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, -1), C_BLACK),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e8ecf0")),
            ("GRID", (0, 0), (-1, -1), 0.4, C_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

    # ─── Correlations ────────────────────────────────────────
    corr = aggregate.get("correlations", {})
    if corr.get("stress_vs_kl"):
        c = corr["stress_vs_kl"]
        story.append(Paragraph(
            f"Stress vs KL correlation: r = {c['r']:.3f}, "
            f"p = {c['p']:.4f}, n = {c['n']}",
            STYLES["body"]))
        story.append(Spacer(1, 6))

    # ─── Batch Plots ─────────────────────────────────────────
    for key, label in [("batch_summary", "Category Distributions"),
                        ("separability", "Effect Sizes")]:
        b64 = plots.get(key)
        if b64:
            story.append(Paragraph(label, STYLES["heading"]))
            img = _b64_to_image(b64, width=usable_width)
            if img:
                story.append(img)
            story.append(Spacer(1, 6))

    # ─── Per-Prompt Summary Table ────────────────────────────
    if per_prompt:
        story.append(PageBreak())
        story.append(Paragraph("Per-Prompt Results", STYLES["heading"]))

        pp_data = [["Prompt", "Cat", "Tok", "Stress", "Entropy",
                     "Bnd%", "Int%", "Net"]]
        for p in per_prompt:
            pp_data.append([
                p["prompt"][:45] + ("..." if len(p["prompt"]) > 45 else ""),
                p.get("category", "")[:4],
                str(p["seq_len"]),
                _fmt(p["stress_score"]),
                _fmt(p["entropy"]),
                _pct(p["top2_share"]),
                _pct(p["middle_share"]),
                _fmt(p["net_correction"]),
            ])

        t = Table(pp_data, repeatRows=1,
                  colWidths=[2.5*inch, 0.4*inch, 0.35*inch,
                             0.65*inch, 0.65*inch, 0.55*inch, 0.55*inch, 0.65*inch])
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Courier"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("TEXTCOLOR", (0, 0), (-1, -1), C_BLACK),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e8ecf0")),
            ("GRID", (0, 0), (-1, -1), 0.3, C_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t)

    doc.build(story)
    return str(filepath)
