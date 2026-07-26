"""Run TAGM as ``python -m src``.

Starts the FastAPI server bound to 127.0.0.1:8000 by default. Host, port,
and log level are configurable via CLI flags or the TAGM_HOST / TAGM_PORT
environment variables. Binding to all interfaces (0.0.0.0) exposes an
unauthenticated API that can load models and read files, so it is opt-in:
set TAGM_HOST=0.0.0.0 or pass --host 0.0.0.0 deliberately.

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
        prog="python -m src",
        description="Start the TAGM FastAPI server.",
    )
    # Loopback by default: the API is unauthenticated, so a LAN-visible
    # bind is opt-in via TAGM_HOST / --host.
    parser.add_argument("--host", default=os.environ.get("TAGM_HOST", "127.0.0.1"),
                        help="Host to bind (default 127.0.0.1)")
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
        "src.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=args.access_log,
        reload=args.reload,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
