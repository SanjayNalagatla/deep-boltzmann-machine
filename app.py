"""
app.py — DBM Dashboard Backend (Flask)
========================================
Serves the dashboard UI and provides REST + SSE endpoints for:
  • Starting/stopping training
  • Streaming live training logs (Server-Sent Events)
  • Serving generated figure images
  • Returning experiment metrics
"""

import os
import sys
import json
import time
import uuid
import queue
import threading
import subprocess
from pathlib import Path
from flask import (
    Flask, render_template, request,
    Response, jsonify, send_file, abort, make_response
)

BASE_DIR = Path(__file__).parent.resolve()
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["SECRET_KEY"] = "dbm-dashboard-2026"
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ── Global job state ──────────────────────────────────────────────────────────
jobs: dict[str, dict] = {}          # job_id -> {status, log_queue, metrics}
active_job_id: str | None = None
active_process: subprocess.Popen | None = None


# ─────────────────────────────────────────────────────────────────────────────
#  Routes — Pages
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    figs = sorted(BASE_DIR.glob("fig_*.png"), key=lambda p: p.stat().st_mtime)
    figures_list = [
        {
            "name": p.stem,
            "file": p.name,
            "label": _label(p.stem),
            "mtime": p.stat().st_mtime,
        }
        for p in figs
    ]
    resp = make_response(render_template("index.html", figures=figures_list))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


# ─────────────────────────────────────────────────────────────────────────────
#  Routes — API
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/run", methods=["POST"])
def run_training():
    """Start a new training job with provided hyperparameters."""
    global active_job_id, active_process

    # Kill any existing job
    if active_process and active_process.poll() is None:
        active_process.terminate()

    data = request.get_json(force=True, silent=True) or {}
    job_id = str(uuid.uuid4())[:8]

    jobs[job_id] = {
        "status": "running",
        "log_queue": queue.Queue(),
        "metrics": {},
        "start_time": time.time(),
        "params": data,
    }
    active_job_id = job_id

    # Build override env for train.py
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["DBM_PRETRAIN_EPOCHS"]  = str(data.get("pretrain_epochs", 25))
    env["DBM_FINETUNE_EPOCHS"]  = str(data.get("finetune_epochs", 15))
    env["DBM_BATCH_SIZE"]       = str(data.get("batch_size", 64))
    env["DBM_LEARNING_RATE"]    = str(data.get("learning_rate", 0.01))
    env["DBM_N_SAMPLES"]        = str(data.get("n_samples", 2000))
    env["DBM_H1"]               = str(data.get("h1", 0))
    env["DBM_H2"]               = str(data.get("h2", 0))
    env["DBM_H3"]               = str(data.get("h3", 0))

    def stream_job(job_id: str, env: dict):
        global active_process
        q = jobs[job_id]["log_queue"]

        try:
            proc = subprocess.Popen(
                [sys.executable, str(BASE_DIR / "train.py")],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=str(BASE_DIR),
            )
            active_process = proc

            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    q.put({"type": "log", "text": line})
                    _parse_metrics(job_id, line)

            proc.wait()
            jobs[job_id]["status"] = "done" if proc.returncode == 0 else "done"
            jobs[job_id]["elapsed"] = round(time.time() - jobs[job_id]["start_time"], 1)
            q.put({"type": "done", "elapsed": jobs[job_id]["elapsed"]})

        except Exception as exc:
            q.put({"type": "error", "text": str(exc)})
            jobs[job_id]["status"] = "error"

    thread = threading.Thread(target=stream_job, args=(job_id, env), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/stop", methods=["POST"])
def stop_training():
    global active_process
    if active_process and active_process.poll() is None:
        active_process.terminate()
        if active_job_id and active_job_id in jobs:
            jobs[active_job_id]["status"] = "stopped"
            jobs[active_job_id]["log_queue"].put({"type": "done", "elapsed": 0})
        return jsonify({"ok": True})
    return jsonify({"ok": False, "reason": "No active process"})


@app.route("/api/stream/<job_id>")
def stream(job_id: str):
    """Server-Sent Events stream for live training log."""
    if job_id not in jobs:
        abort(404)

    def event_generator():
        q = jobs[job_id]["log_queue"]
        while True:
            try:
                item = q.get(timeout=30)
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield "data: {\"type\":\"ping\"}\n\n"

    return Response(
        event_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/figures")
def list_figures():
    """List all generated PNG figures."""
    figs = sorted(BASE_DIR.glob("fig_*.png"), key=lambda p: p.stat().st_mtime)
    return jsonify([
        {
            "name": p.stem,
            "file": p.name,
            "label": _label(p.stem),
            "mtime": p.stat().st_mtime,
        }
        for p in figs
    ])


@app.route("/api/figure/<filename>")
def serve_figure(filename: str):
    """Serve a PNG figure file."""
    path = BASE_DIR / filename
    if not path.exists() or path.suffix != ".png":
        abort(404)
    return send_file(str(path), mimetype="image/png")


@app.route("/api/metrics/<job_id>")
def get_metrics(job_id: str):
    if job_id not in jobs:
        abort(404)
    return jsonify(jobs[job_id].get("metrics", {}))


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_metrics(job_id: str, line: str):
    """Extract key metrics from training output lines."""
    m = jobs[job_id]["metrics"]
    q = jobs[job_id]["log_queue"]

    # Layer pre-training error
    if "Pre-training Layer" in line:
        try:
            layer = int(line.split("Layer")[1].strip().split()[0])
            m["current_layer"] = layer
        except Exception:
            pass
        q.put({"type": "layer", "text": line})

    # Epoch error line: "  Epoch  N/M  |  Recon Error: X.XXXXXX"
    if "Recon Error:" in line and "Epoch" in line:
        try:
            err = float(line.split("Recon Error:")[1].strip())
            parts = line.split("Epoch")[1].strip().split("/")
            epoch = int(parts[0].strip())
            total = int(parts[1].split()[0].strip())
            layer = m.get("current_layer", 1)
            key = f"layer{layer}_errors"
            m.setdefault(key, []).append(err)
            m["last_error"] = err
            m["last_epoch"] = epoch
            m["last_total"] = total
            q.put({"type": "epoch", "layer": layer, "epoch": epoch,
                   "total": total, "error": err})
        except Exception:
            pass

    # Pre-train / fine-tune MSE
    if "Validation MSE (post pre-train)" in line:
        try:
            m["pretrain_mse"] = float(line.split(":")[1].strip())
        except Exception:
            pass
    if "Validation MSE (post fine-tune)" in line:
        try:
            m["finetune_mse"] = float(line.split(":")[1].strip())
        except Exception:
            pass
    if "MSE improvement" in line:
        try:
            m["improvement"] = line.split(":")[1].strip()
        except Exception:
            pass
    if "Architecture" in line and "->" in line:
        try:
            m["architecture"] = line.split(":")[1].strip()
        except Exception:
            pass
    if "Dataset" in line and "Bars" in line or "MNIST" in line:
        try:
            m["dataset"] = line.split(":")[1].strip()
        except Exception:
            pass


def _label(stem: str) -> str:
    labels = {
        "fig_dataset_samples":        "Dataset Samples",
        "fig_training_curves":        "Training Loss Curves",
        "fig_weight_filters":         "Learned Weight Filters",
        "fig_reconstructions":        "Reconstruction Quality",
        "fig_tsne_representations":   "t-SNE Representations",
        "fig_activation_heatmap":     "Activation Heatmap",
        "fig_generated_samples":      "Generated Samples",
    }
    return labels.get(stem, stem.replace("fig_", "").replace("_", " ").title())


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  DBM Dashboard running at  http://127.0.0.1:5000\n")
    app.run(debug=False, threaded=True, port=5000)
