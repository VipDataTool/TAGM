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
from pathlib import Path


def _build_log_config(log_file: Path, level: str) -> dict:
    """Uvicorn LOGGING_CONFIG augmented with a FileHandler writing to tagm.log.

    Uvicorn owns logging configuration; the application module must not call
    logging.basicConfig() because uvicorn has already configured the root
    logger by the time the app is imported. The correct integration point is
    here: hand uvicorn a config dict that includes our FileHandler alongside
    its own console handlers, and let it apply the whole thing at startup.

    Note on formatter naming: uvicorn looks for formatters named "default"
    and "access" and tries to inject a `use_colors` key into them, which is
    only valid on uvicorn's own formatter classes. To avoid that mutation,
    we name our formatters "tagm" and use plain logging.Formatter.
    """
    level = level.upper()
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "tagm": {
                "format": "%(asctime)s [%(levelname)s] %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "tagm",
                "stream": "ext://sys.stderr",
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "tagm",
                "filename": str(log_file),
                "encoding": "utf-8",
            },
        },
        "loggers": {
            # Root: app + uvicorn server messages → both console and file.
            "": {"handlers": ["console", "file"], "level": level},
            # Uvicorn loggers: route through root rather than their own
            # handlers. propagate=True is the default but explicit here.
            "uvicorn":       {"handlers": [], "level": level, "propagate": True},
            "uvicorn.error": {"handlers": [], "level": level, "propagate": True},
            "uvicorn.access":{"handlers": [], "level": level, "propagate": True},
        },
    }


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
    parser.add_argument("--log-file",
                        default=os.environ.get(
                            "TAGM_LOG_FILE",
                            str(Path(__file__).parent.parent / "tagm.log")),
                        help="Path to the application log file "
                             "(default: <repo>/tagm.log; override with "
                             "TAGM_LOG_FILE env var or this flag).")
    parser.add_argument("--access-log", action="store_true",
                        help="Enable per-request access logging (off by default "
                             "because the frontend polls every ~2s).")
    parser.add_argument("--reload", action="store_true",
                        help="Enable uvicorn auto-reload (development only).")
    args = parser.parse_args(argv)

    # Expose log path to the app so /api/log/download can find it.
    log_file = Path(args.log_file).resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    os.environ["TAGM_LOG_FILE"] = str(log_file)

    import uvicorn
    uvicorn.run(
        "tagm.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=args.access_log,
        reload=args.reload,
        timeout_keep_alive=300,
        log_config=_build_log_config(log_file, args.log_level),
    )


if __name__ == "__main__":
    main()
