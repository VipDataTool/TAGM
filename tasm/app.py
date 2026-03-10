"""
TASM Analyzer - The Alignment Stress Map
FastAPI web application for runtime alignment signal analysis.
"""

import os
import io
import csv
import json
import time
import zipfile
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from contextlib import asynccontextmanager

from engine.model_manager import ModelManager, KNOWN_PAIRS
from engine.analyzer import Analyzer, result_to_dict
from engine.baselines import BaselineManager
from engine.statistics import aggregate_batch
from engine.visualizations import (
    plot_signed_attribution, plot_stress_per_token,
    plot_amplitude_trajectory, plot_heatmap,
    plot_distribution_metrics, plot_batch_summary,
    plot_separability,
)
from engine.reports import generate_single_report, generate_batch_report

# ─── Global state ────────────────────────────────────────────────
mm = ModelManager()
analyzer = None
baselines = BaselineManager()
progress_log = []

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def log_progress(stage, message):
    progress_log.append({"stage": stage, "message": message,
                         "time": time.time()})


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="TASM Analyzer", lifespan=lifespan)


# ─── API Routes ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/status")
async def get_status():
    return {
        "model_loaded": mm.is_loaded(),
        "model_pair": mm.state.pair_id if mm.state else None,
        "model_name": mm.state.display_name if mm.state else None,
        "available_pairs": mm.get_available_pairs(),
        "baseline_summary": baselines.get_summary(),
    }


@app.post("/api/load_model")
async def load_model(pair_id: str = Form(None),
                     base_id: str = Form(None),
                     instruct_id: str = Form(None)):
    global analyzer
    progress_log.clear()
    try:
        mm.load_pair(pair_id=pair_id, base_id=base_id,
                     instruct_id=instruct_id, callback=log_progress)
        analyzer = Analyzer(mm)
        baselines.compute_builtin_baselines(analyzer, callback=log_progress)
        return {"ok": True, "message": progress_log[-1]["message"] if progress_log else "Loaded"}
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"ok": False, "error": str(e),
                                     "trace": traceback.format_exc()})


@app.post("/api/reset")
async def reset_all():
    """Full reset: unload model, clear baselines and results."""
    global analyzer, baselines
    progress_log.clear()
    mm.reset()
    analyzer = None
    baselines = BaselineManager()
    log_progress("reset", "All resources released")
    return {"ok": True, "message": "Reset complete. Select a model pair to begin."}


@app.get("/api/progress")
async def get_progress():
    return {"log": progress_log}


@app.post("/api/analyze")
async def analyze_single(prompt: str = Form(...),
                         category: str = Form(""),
                         compute_kl: bool = Form(False),
                         compute_trajectory: bool = Form(True)):
    if not analyzer:
        return JSONResponse(status_code=400,
                            content={"error": "No model loaded"})
    try:
        if compute_kl:
            mm.load_base_for_kl(callback=log_progress)

        result = analyzer.analyze_prompt(
            prompt, category=category,
            compute_kl=compute_kl,
            compute_full_trajectory=compute_trajectory)

        baselines.normalize_result(result)

        # Generate visualizations
        plots = {
            "signed_attribution": plot_signed_attribution(result),
            "stress_per_token": plot_stress_per_token(result),
            "distribution_metrics": plot_distribution_metrics(result),
        }
        if compute_trajectory:
            plots["amplitude_trajectory"] = plot_amplitude_trajectory(result)
            plots["heatmap"] = plot_heatmap(result)

        if compute_kl:
            mm.unload_base()

        result_dict = result_to_dict(result)

        # Auto-generate PDF report
        model_name = mm.state.display_name if mm.state else ""
        try:
            pdf_path = generate_single_report(result_dict, plots,
                                               model_name=model_name)
            pdf_filename = Path(pdf_path).name
            log_progress("report", f"Report saved: {pdf_filename}")
        except Exception as e:
            pdf_filename = None
            log_progress("report", f"PDF generation failed: {e}")

        return {
            "ok": True,
            "result": result_dict,
            "plots": plots,
            "pdf_filename": pdf_filename,
        }
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"error": str(e),
                                     "trace": traceback.format_exc()})


@app.post("/api/analyze_batch")
async def analyze_batch(file: UploadFile = File(...),
                        compute_kl: bool = Form(False),
                        compute_trajectory: bool = Form(False),
                        baseline_file: Optional[UploadFile] = File(None)):
    if not analyzer:
        return JSONResponse(status_code=400,
                            content={"error": "No model loaded"})
    try:
        progress_log.clear()

        content = (await file.read()).decode("utf-8")
        reader = csv.DictReader(io.StringIO(content))
        prompts = []
        for row in reader:
            prompts.append({
                "prompt": row.get("prompt", row.get("Prompt", "")),
                "category": row.get("category", row.get("Category", "unknown")),
            })

        if not prompts:
            return JSONResponse(status_code=400,
                                content={"error": "No prompts found in CSV. "
                                         "Expected columns: prompt, category"})

        log_progress("batch", f"Loaded {len(prompts)} prompts from CSV")

        if baseline_file:
            bl_content = (await baseline_file.read()).decode("utf-8")
            bl_reader = csv.DictReader(io.StringIO(bl_content))
            bl_prompts = [row.get("prompt", row.get("Prompt", ""))
                          for row in bl_reader if row.get("prompt") or row.get("Prompt")]
            if bl_prompts:
                baselines.add_user_baselines(bl_prompts, analyzer, callback=log_progress)

        if compute_kl:
            mm.load_base_for_kl(callback=log_progress)

        results = []
        for i, p in enumerate(prompts):
            log_progress("analyzing",
                         f"[{i+1}/{len(prompts)}] {p['prompt'][:60]}...")
            r = analyzer.analyze_prompt(
                p["prompt"], category=p["category"],
                compute_kl=compute_kl,
                compute_full_trajectory=compute_trajectory)
            baselines.normalize_result(r)
            results.append(r)

        if compute_kl:
            mm.unload_base()

        log_progress("aggregating", "Computing aggregate statistics...")
        agg = aggregate_batch(results)

        plots = {
            "batch_summary": plot_batch_summary(agg),
            "separability": plot_separability(agg),
        }

        per_prompt = []
        for r in results:
            d = result_to_dict(r)
            d.pop("heatmap", None)
            d.pop("amplitude_trajectory", None)
            d.pop("amplitude_normalized", None)
            per_prompt.append(d)

        # Auto-generate PDF report
        model_name = mm.state.display_name if mm.state else ""
        try:
            pdf_path = generate_batch_report(
                agg, per_prompt, plots,
                model_name=model_name, n_prompts=len(results))
            pdf_filename = Path(pdf_path).name
            log_progress("report", f"Report saved: {pdf_filename}")
        except Exception as e:
            pdf_filename = None
            log_progress("report", f"PDF generation failed: {e}")

        log_progress("done", f"Batch analysis complete: {len(results)} prompts")

        return {
            "ok": True,
            "aggregate": agg,
            "per_prompt": per_prompt,
            "plots": plots,
            "n_prompts": len(results),
            "pdf_filename": pdf_filename,
        }
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"error": str(e),
                                     "trace": traceback.format_exc()})


@app.get("/api/reports")
async def list_reports():
    """List all generated PDF reports."""
    reports = []
    for f in sorted(REPORTS_DIR.glob("*.pdf"), key=os.path.getmtime, reverse=True):
        reports.append({
            "filename": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "created": time.strftime("%Y-%m-%d %H:%M:%S",
                                      time.localtime(f.stat().st_mtime)),
        })
    return {"reports": reports}


@app.get("/api/reports/{filename}")
async def download_report(filename: str):
    """Download a specific PDF report."""
    filepath = REPORTS_DIR / filename
    if not filepath.exists() or not filepath.suffix == ".pdf":
        return JSONResponse(status_code=404,
                            content={"error": "Report not found"})
    return FileResponse(
        path=str(filepath),
        media_type="application/pdf",
        filename=filename,
    )


@app.post("/api/export_zip")
async def export_zip(data: Request):
    """Export batch results as a ZIP with CSV + plots + PDF."""
    body = await data.json()
    results = body.get("per_prompt", [])
    agg = body.get("aggregate", {})
    plots = body.get("plots", {})

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        csv_buf = io.StringIO()
        if results:
            fieldnames = ["prompt", "category", "seq_len", "stress_score",
                          "entropy", "gini", "top2_share", "middle_share",
                          "interior_cv", "net_correction", "n_negative_tokens",
                          "kl_divergence", "entropy_ln", "top2_share_ln",
                          "middle_share_ln", "stress_score_ln"]
            writer = csv.DictWriter(csv_buf, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        zf.writestr("results.csv", csv_buf.getvalue())
        zf.writestr("aggregate_statistics.json",
                     json.dumps(agg, indent=2, default=str))

        import base64
        for name, b64 in plots.items():
            if b64:
                zf.writestr(f"plots/{name}.png", base64.b64decode(b64))

        # Include latest matching PDF if it exists
        pdf_filename = body.get("pdf_filename")
        if pdf_filename:
            pdf_path = REPORTS_DIR / pdf_filename
            if pdf_path.exists():
                zf.write(str(pdf_path), f"report.pdf")

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=tasm_results.zip"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
