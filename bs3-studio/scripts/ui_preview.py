"""Serve the BS Profiler 3.0 page with an already finished job pre-rendered (for UI checks without the GPU analysis).
Usage: ui_preview.py [JOB_DIR] [PORT]     default: newest job in ~/bs3_data/web_jobs, port 7882
"""
import os
import sys
from pathlib import Path

# own Gradio temp folder (the shared /tmp/gradio is cleaned by another service); must be set before gradio is imported
os.environ.setdefault("GRADIO_TEMP_DIR", str(Path.home() / "bs3_data" / "gradio_tmp"))
Path(os.environ["GRADIO_TEMP_DIR"]).mkdir(parents=True, exist_ok=True)

from bs3.pipeline import Studio  # noqa: E402
from bs3.webapp import build_app  # noqa: E402

jobs = Path.home() / "bs3_data" / "web_jobs"
job = sys.argv[1] if len(sys.argv) > 1 else str(sorted(p for p in jobs.iterdir() if (p / "result.json").exists())[-1])
port = int(sys.argv[2]) if len(sys.argv) > 2 else 7882
print("preview job:", job, "port:", port, flush=True)
demo = build_app(Studio(), jobs, preview_job=job)
demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=port, show_api=False, show_error=True,
                                                allowed_paths=[str(jobs)])
