"""
PDF Report Generator for TASM analysis results.
Professional template with user info, descriptions, and proper layout.
"""

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
    Image, PageBreak, HRFlowable, KeepTogether,
)
from reportlab.lib import colors
from functools import partial

# ─── Colors ──────────────────────────────────────────────────────
C_BORDER = HexColor("#cccccc")
C_HEADER_BG = HexColor("#1a365d")
C_HEADER_TEXT = HexColor("#ffffff")
C_ACCENT = HexColor("#2b6cb0")
C_LIGHT_BG = HexColor("#f7fafc")
C_DIM = HexColor("#718096")
C_BLACK = HexColor("#1a202c")

REPORTS_DIR = Path("reports")

# ─── Styles ──────────────────────────────────────────────────────
S_TITLE = ParagraphStyle("s_title", fontName="Helvetica-Bold", fontSize=22,
                          textColor=C_HEADER_BG, spaceAfter=4)
S_SUBTITLE = ParagraphStyle("s_subtitle", fontName="Helvetica", fontSize=11,
                             textColor=C_DIM, spaceAfter=2)
S_H1 = ParagraphStyle("s_h1", fontName="Helvetica-Bold", fontSize=14,
                        textColor=C_HEADER_BG, spaceBefore=16, spaceAfter=8)
S_H2 = ParagraphStyle("s_h2", fontName="Helvetica-Bold", fontSize=11,
                        textColor=C_ACCENT, spaceBefore=10, spaceAfter=4)
S_BODY = ParagraphStyle("s_body", fontName="Helvetica", fontSize=10,
                          textColor=C_BLACK, spaceAfter=6, leading=14)
S_SMALL = ParagraphStyle("s_small", fontName="Helvetica", fontSize=10,
                           textColor=C_DIM, spaceAfter=4, leading=13)
S_MONO = ParagraphStyle("s_mono", fontName="Courier", fontSize=10,
                          textColor=C_BLACK, spaceAfter=4, leading=13)
S_FOOTER = ParagraphStyle("s_footer", fontName="Helvetica", fontSize=10,
                            textColor=C_DIM)


def _fmt(v, fmt=".4f"):
    if v is None: return "--"
    try: return format(v, fmt)
    except: return str(v)

def _pct(v):
    if v is None: return "--"
    return f"{v*100:.1f}%"

def _b64_to_image(b64_str, width=None):
    if not b64_str: return None
    buf = io.BytesIO(base64.b64decode(b64_str))
    img = Image(buf)
    if width:
        ratio = img.imageHeight / img.imageWidth
        img.drawWidth = width
        img.drawHeight = width * ratio
    return img

def _table_style():
    return TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (-1, -1), C_BLACK),
        ("BACKGROUND", (0, 0), (-1, 0), C_LIGHT_BG),
        ("GRID", (0, 0), (-1, -1), 0.4, C_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("WORDWRAP", (0, 0), (-1, -1), True),
    ])


class _HeaderFooter:
    """Draws header and footer on every page."""
    def __init__(self, title, user_name="", org="", timestamp=""):
        self.title = title
        self.user = user_name
        self.org = org
        self.ts = timestamp

    def __call__(self, canvas, doc):
        canvas.saveState()
        w, h = letter
        # Header line
        canvas.setStrokeColor(C_ACCENT)
        canvas.setLineWidth(0.5)
        canvas.line(0.6*inch, h - 0.5*inch, w - 0.6*inch, h - 0.5*inch)
        canvas.setFont("Helvetica", 10)
        canvas.setFillColor(C_DIM)
        canvas.drawString(0.6*inch, h - 0.45*inch, f"TASM Report: {self.title}")
        canvas.drawRightString(w - 0.6*inch, h - 0.45*inch, self.ts)

        # Footer
        canvas.line(0.6*inch, 0.55*inch, w - 0.6*inch, 0.55*inch)
        canvas.setFont("Helvetica", 10)
        canvas.drawString(0.6*inch, 0.38*inch,
                         f"{self.org} | {self.user}" if self.org else self.user or "TASM Analyzer")
        canvas.drawRightString(w - 0.6*inch, 0.38*inch, f"Page {doc.page}")
        canvas.restoreState()


def _cover_page(story, title, user_info, model_name, timestamp, usable_width):
    """Generate a cover page."""
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph(
        '<font size="36"><b>TASM</b></font>'
        '&nbsp;&nbsp;'
        '<font size="14" color="#718096">The Alignment Stress Map</font>',
        ParagraphStyle("cover_line", fontName="Helvetica-Bold",
                        fontSize=36, textColor=C_ACCENT, alignment=TA_CENTER,
                        spaceAfter=8)))
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="60%", thickness=1, color=C_ACCENT,
                              spaceAfter=12, hAlign="CENTER"))
    story.append(Paragraph(title, ParagraphStyle("cover_title",
        fontName="Helvetica-Bold", fontSize=18, textColor=C_BLACK,
        alignment=TA_CENTER, spaceAfter=24)))

    # User info table
    info_data = []
    if user_info.get("name"):
        info_data.append(["Analyst", user_info["name"]])
    if user_info.get("organization"):
        info_data.append(["Organization", user_info["organization"]])
    info_data.append(["Model", model_name or "--"])
    info_data.append(["Generated", timestamp])

    if info_data:
        t = Table(info_data, colWidths=[1.5*inch, 3.5*inch], hAlign="CENTER")
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (-1, -1), C_BLACK),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(t)

    story.append(Spacer(1, 1*inch))
    story.append(Paragraph(
        "Runtime Per-Token Sensitivity Attribution via Weight Delta Projection "
        "in Transformer Language Models",
        ParagraphStyle("cover_desc", fontName="Helvetica", fontSize=11,
                        textColor=C_DIM, alignment=TA_CENTER, leading=14)))
    story.append(PageBreak())


def _apparatus_section(story):
    """Describe the TASM apparatus for context."""
    story.append(Paragraph("About This Analysis", S_H1))
    story.append(Paragraph(
        "This report was generated by TASM (The Alignment Stress Map), a runtime monitoring "
        "tool that measures alignment tension in transformer language models. TASM computes "
        "the weight delta between a base model and its alignment-tuned counterpart, then "
        "projects inference-time activations through this delta to quantify where and how "
        "strongly alignment training corrects the base model's behavior.",
        S_BODY))
    story.append(Paragraph(
        "Key metrics include the stress score (how hard alignment correction pushes at "
        "signal layers), signed attribution (per-token decomposition of correction direction), "
        "and distribution metrics (entropy, boundary/interior concentration) that characterize "
        "the shape of the correction signal across token positions. Length-normalized metrics "
        "report deviation from benign baselines at matched token lengths, addressing the "
        "token-length confound. The Lateral Tension Profile (LTP) extends the ASM with "
        "directional information, probing how the alignment field varies perpendicular to the "
        "generation path using counterfactual token directions. LTP summary statistics include "
        "offset magnitude (M), consistency (C), variance (V), and lateral coverage (L).",
        S_BODY))
    story.append(Spacer(1, 6))


def ensure_reports_dir():
    REPORTS_DIR.mkdir(exist_ok=True)


def generate_single_report(result_dict, plots, model_name="",
                            user_info=None) -> str:
    ensure_reports_dir()
    user_info = user_info or {}
    r = result_dict
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in r["prompt"][:40]).strip()
    safe_prompt = safe_prompt.replace(" ", "_") or "analysis"
    filename = f"{safe_prompt}_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = REPORTS_DIR / filename

    usable_width = letter[0] - 1.2*inch
    hf = _HeaderFooter("Single Prompt Analysis",
                         user_info.get("name", ""), user_info.get("organization", ""),
                         timestamp)

    doc = SimpleDocTemplate(str(filepath), pagesize=letter,
        leftMargin=0.6*inch, rightMargin=0.6*inch,
        topMargin=0.6*inch, bottomMargin=0.6*inch)
    story = []

    _cover_page(story, "Single Prompt Analysis", user_info, model_name, timestamp, usable_width)
    _apparatus_section(story)

    # Prompt section
    story.append(Paragraph("Prompt Under Analysis", S_H1))
    story.append(Paragraph(r["prompt"], S_MONO))
    parts = [f"Tokens: {r['seq_len']}"]
    if r.get("category"): parts.append(f"Category: {r['category']}")
    story.append(Paragraph(" | ".join(parts), S_SMALL))
    story.append(Spacer(1, 8))

    # Metrics
    story.append(Paragraph("Distribution Metrics", S_H1))
    story.append(Paragraph(
        "These metrics quantify the shape and strength of the alignment correction signal. "
        "The stress score measures overall correction pressure at discriminative layers. "
        "Entropy and boundary/interior share characterize how the correction distributes "
        "across token positions -- benign prompts typically show boundary-concentrated "
        "attribution while adversarial prompts show interior-distributed patterns.",
        S_BODY))

    data = [["Metric", "Value", "Length-Norm"],
            ["Stress Score", _fmt(r["stress_score"]), ""],
            ["Net Correction", _fmt(r["net_correction"]), ""],
            ["Entropy", _fmt(r["entropy"]), ""],
            ["Gini", _fmt(r["gini"]), ""],
            ["Boundary Share", _pct(r["top2_share"]), ""],
            ["Interior Share", _pct(r["middle_share"]), ""],
            ["Interior CV", _fmt(r["interior_cv"]), ""]]
    if r.get("kl_divergence") is not None:
        data.append(["KL Divergence", _fmt(r["kl_divergence"]), ""])
    data.append(["Negative Tokens", f"{r['n_negative_tokens']}/{r['seq_len']}",
                  "detected" if r["has_negative_tokens"] else "none"])

    # LTP metrics
    ltp = r.get("ltp")
    if ltp:
        data.append(["LTP Offset Mag (M)", _fmt(ltp.get("mean_M")), ""])
        data.append(["LTP Consistency (C)", _fmt(ltp.get("mean_C")), ""])
        data.append(["LTP Variance (V)", _fmt(ltp.get("mean_V")), ""])
        data.append(["LTP Coverage (L)", _fmt(ltp.get("mean_L")), ""])
        data.append(["LTP Strategy", ltp.get("layer_strategy", "--"), f"k={ltp.get('k', '--')}"])

    t = Table(data, colWidths=[1.8*inch, 1.4*inch, 1.4*inch])
    t.setStyle(_table_style())
    story.append(t)
    story.append(Spacer(1, 10))

    # Token table
    if r.get("tokens") and r.get("signed_attr"):
        story.append(Paragraph("Per-Token Attribution", S_H1))
        story.append(Paragraph(
            "Each token's signed attribution indicates whether it pushes with (+) or against (-) "
            "the alignment correction at the last position. The stress column shows the total "
            "correction pressure through that token's representation at signal layers.",
            S_BODY))
        tok_data = [["Token", "Signed Attr", "Stress"]]
        for i, tok in enumerate(r["tokens"]):
            a = r["signed_attr"][i] if i < len(r["signed_attr"]) else 0
            s = r["per_token_stress"][i] if i < len(r.get("per_token_stress", [])) else 0
            tok_data.append([tok.strip(), f"{a:+.4f}", _fmt(s)])
        tt = Table(tok_data, colWidths=[2.0*inch, 1.2*inch, 1.2*inch], repeatRows=1)
        tt.setStyle(_table_style())
        story.append(tt)
        story.append(Spacer(1, 8))

    # Plots
    plot_descs = {
        "signed_attribution": ("Signed Attribution Chart",
            "Per-token signed attribution to the last position, averaged across signal layers and heads. "
            "Green bars push with the alignment correction; red bars push against it."),
        "stress_per_token": ("Focused Stress Score",
            "Per-token stress at discriminative middle layers (normalized by delta Frobenius norm). "
            "Higher values indicate stronger correction pressure through that token's representation."),
        "distribution_metrics": ("Distribution Metrics Summary",
            "Overview of the four key distribution metrics for this prompt."),
        "amplitude_trajectory": ("Full Amplitude Trajectory",
            "Normalized sensitivity across all sublayers (alternating attention/MLP). "
            "The shape reveals where in the model's depth the alignment correction is most active."),
        "heatmap": ("Sensitivity Heatmap",
            "Per-token, per-layer normalized sensitivity. Bright regions indicate where specific "
            "tokens drive strong correction at specific depths."),
        # LTP plots
        "ltp_profiles": ("Lateral Tension Profiles",
            "Per-token lateral tension across counterfactual rank positions. Each bar segment represents "
            "how much alignment correction the weight delta applies toward each unchosen alternative token."),
        "ltp_tension_magnitudes": ("Lateral Tension Magnitudes",
            "Per-token magnitude of the net lateral tension point, colored by profile shape classification. "
            "Steep profiles indicate directionally concentrated correction; inverted profiles are anomalous."),
        "ltp_dual_trajectory": ("Dual Trajectory (PCA 2D)",
            "PCA projection of the semantic trajectory (where the model went) and the tension trajectory "
            "(where the alignment field displaces it). Offset lines show lateral tension at each token."),
        "ltp_summary_stats": ("LTP Summary Statistics",
            "The four LTP summary metrics: offset magnitude (M), consistency (C), variance (V), "
            "and lateral coverage (L). High M with high C suggests systematic boundary-threading."),
        "ltp_profile_heatmap": ("LTP Profile Heatmap",
            "Token-by-rank heatmap of lateral tension. Rows are tokens; columns are counterfactual "
            "alternatives ranked by logit score. The shape of each row classifies the local alignment landscape."),
    }
    for key, (title, desc) in plot_descs.items():
        b64 = plots.get(key)
        if b64:
            story.append(Paragraph(title, S_H2))
            story.append(Paragraph(desc, S_SMALL))
            img = _b64_to_image(b64, width=usable_width)
            if img: story.append(img)
            story.append(Spacer(1, 6))

    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    return str(filepath)


def generate_batch_report(aggregate, per_prompt, plots,
                           model_name="", n_prompts=0,
                           user_info=None) -> str:
    ensure_reports_dir()
    user_info = user_info or {}
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    filename = f"batch_{n_prompts}prompts_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = REPORTS_DIR / filename

    usable_width = letter[0] - 1.2*inch
    hf = _HeaderFooter(f"Batch Analysis ({n_prompts} prompts)",
                         user_info.get("name", ""), user_info.get("organization", ""),
                         timestamp)

    doc = SimpleDocTemplate(str(filepath), pagesize=letter,
        leftMargin=0.6*inch, rightMargin=0.6*inch,
        topMargin=0.6*inch, bottomMargin=0.6*inch)
    story = []

    _cover_page(story, f"Batch Analysis: {n_prompts} Prompts", user_info,
                model_name, timestamp, usable_width)
    _apparatus_section(story)

    # Category summary
    cats = aggregate.get("categories", {})
    if cats:
        story.append(Paragraph("Category Summary", S_H1))
        story.append(Paragraph(
            "Prompts grouped by category with bootstrapped mean estimates (5000 resamples, 95% CI). "
            "The negative token rate indicates how often correction-suppressing tokens are observed.",
            S_BODY))

        cat_data = [["Category", "N", "Avg Len", "Stress", "Entropy",
                      "Bnd%", "Int%", "Net", "Neg Rate"]]
        for cat_name in ["benign", "mild", "harmful", "jailbreak"]:
            if cat_name not in cats: continue
            s = cats[cat_name]; m = s.get("metrics", {})
            cat_data.append([
                cat_name.title(), str(s["n"]),
                _fmt(s.get("mean_seq_len"), ".1f"),
                _fmt(m.get("stress_score", {}).get("estimate")),
                _fmt(m.get("entropy", {}).get("estimate")),
                _pct(m.get("top2_share", {}).get("estimate")),
                _pct(m.get("middle_share", {}).get("estimate")),
                _fmt(m.get("net_correction", {}).get("estimate")),
                _pct(s.get("negative_token_rate")),
            ])
        t = Table(cat_data, repeatRows=1)
        t.setStyle(_table_style())
        story.append(t)
        story.append(Spacer(1, 10))

    # Separability
    sep = aggregate.get("separability", {})
    if sep:
        story.append(Paragraph("Separability Analysis", S_H1))
        story.append(Paragraph(
            "Effect sizes (Cohen's d) with bootstrap 95% confidence intervals comparing benign "
            "prompts against harmful/jailbreak prompts. Values above 0.8 indicate large effects. "
            "Best threshold accuracy shows the optimal single-threshold classification performance.",
            S_BODY))

        sep_data = [["Metric", "Cohen's d", "95% CI", "Best Acc"]]
        for metric, s in sep.items():
            es = s.get("effect_size", {})
            sep_data.append([
                metric.replace("_", " ").title(),
                _fmt(es.get("estimate"), ".2f"),
                f"[{_fmt(es.get('ci_low'), '.2f')}, {_fmt(es.get('ci_high'), '.2f')}]",
                _pct(s.get("threshold", {}).get("accuracy")),
            ])
        t = Table(sep_data, repeatRows=1)
        t.setStyle(_table_style())
        story.append(t)
        story.append(Spacer(1, 10))

    # Comparative plots
    comp_descs = {
        "trajectory_overlay": ("Amplitude Trajectories (Overlay)",
            "All prompts' normalized amplitude trajectories overlaid, colored by category. "
            "Divergence between categories indicates where alignment correction differentiates."),
        "difference_from_benign": ("Difference from Benign Baseline",
            "Per-category mean trajectory minus the benign mean. Positive regions show where "
            "adversarial prompts trigger stronger correction than benign inputs."),
        "discriminative_sublayers": ("Discriminative Sublayers",
            "Sublayers ranked by their ability to distinguish adversarial from benign prompts. "
            "Middle-layer attention sublayers are predicted to dominate."),
        "metric_scatters": ("Metric Scatter Plots",
            "Pairwise scatter plots of key metrics, colored by category. Clustering indicates "
            "separability; overlap indicates confounded metrics."),
        "behavioral_comparison": ("Behavioral Comparison",
            "Instruct vs base model top-1 next-token probabilities per prompt. Larger gaps "
            "indicate stronger behavioral divergence from alignment training."),
        "proof1_summary": ("Proof 1 Exactness Verification",
            "Verification that the sum of signed attributions equals the correction norm, "
            "confirming mathematical correctness of the decomposition."),
        "batch_summary": ("Category Distributions",
            "Box plots of key metrics across categories."),
        "separability": ("Effect Size Visualization",
            "Cohen's d with 95% bootstrap confidence intervals for each metric."),
        # LTP comparative plots
        "ltp_category_comparison": ("LTP Summary by Category",
            "Box plots of the four LTP summary statistics (M, C, V, L) across prompt categories. "
            "Hypothesis 1 predicts adversarial prompts show higher offset magnitude."),
        "ltp_m_vs_stress": ("LTP Offset Magnitude vs Stress Score",
            "Scatter plot of LTP offset magnitude (M) against ASM stress score. Prompts that "
            "are separable by M but not by stress validate Hypothesis 3 (lateral structure captures "
            "information that amplitude alone does not)."),
        "ltp_profile_shapes": ("Profile Shape Distribution",
            "Distribution of steep, flat, and inverted lateral tension profiles by category. "
            "Hypothesis 5 predicts that shape distributions differ between categories."),
    }
    for key, (title, desc) in comp_descs.items():
        b64 = plots.get(key)
        if b64:
            story.append(Paragraph(title, S_H2))
            story.append(Paragraph(desc, S_SMALL))
            img = _b64_to_image(b64, width=usable_width)
            if img: story.append(img)
            story.append(Spacer(1, 6))

    # Per-prompt table
    if per_prompt:
        story.append(PageBreak())
        story.append(Paragraph("Per-Prompt Results", S_H1))
        story.append(Paragraph(
            "Individual metrics for each prompt in the dataset. Prompts are listed in analysis order.",
            S_BODY))

        pp_data = [["Prompt", "Cat", "Tok", "Stress", "Ent",
                     "Bnd%", "Int%", "Net"]]
        wrap_style = ParagraphStyle("pp_wrap", fontName="Courier", fontSize=10,
                                     textColor=C_BLACK, leading=12, wordWrap="CJK")
        for p in per_prompt:
            prompt_text = p.get("prompt", "")[:60]
            if len(p.get("prompt", "")) > 60:
                prompt_text += "..."
            pp_data.append([
                Paragraph(prompt_text, wrap_style),
                (p.get("category", "") or "?")[:5],
                str(p.get("seq_len", "")),
                _fmt(p.get("stress_score")),
                _fmt(p.get("entropy")),
                _pct(p.get("top2_share")),
                _pct(p.get("middle_share")),
                _fmt(p.get("net_correction")),
            ])
        t = Table(pp_data, repeatRows=1,
                  colWidths=[2.6*inch, 0.45*inch, 0.35*inch,
                             0.55*inch, 0.55*inch, 0.45*inch, 0.45*inch, 0.55*inch])
        t.setStyle(_table_style())
        story.append(t)

    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    return str(filepath)
