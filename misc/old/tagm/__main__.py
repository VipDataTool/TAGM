"""Run TAGM as ``python -m tagm``.

Starts the FastAPI server bound to 0.0.0.0:8000 by default. Host, port,
and log level are configurable via CLI flags or the TAGM_HOST / TAGM_PORT
environment variables.

Access logging is off by default because TAGM's frontend polls the
status/progress endpoints every ~2s, which otherwise floods the terminal
with access-log lines. Application-level logs (loads, errors, batch
progress) still print. Pass --access-log to enable per-request lines.
"""
from __future__ import annotations

import argparse
import os


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tagm",
        description="Start the TAGM FastAPI server.",
    )
    parser.add_argument("--host", default=os.environ.get("TAGM_HOST", "0.0.0.0"),
                        help="Host to bind (default 0.0.0.0)")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("TAGM_PORT", "8000")),
                        help="Port to bind (default 8000)")
    parser.add_argument("--log-level", default="info",
                        choices=("critical", "error", "warning", "info", "debug"))
    parser.add_argument("--access-log", action="store_true",
                        help="Enable per-request access logging (off by default "
                             "because the frontend polls every ~2s).")
    parser.add_argument("--reload", action="store_true",
                        help="Enable uvicorn auto-reload (development only).")
    args = parser.parse_args(argv)

    import uvicorn
    uvicorn.run(
        "tagm.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=args.access_log,
        reload=args.reload,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
