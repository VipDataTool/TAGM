"""Run TAGM as ``python -m src``.

Host, port and log level are configurable via CLI flags or the TAGM_HOST /
TAGM_PORT environment variables.

Default bind is environment-dependent, because the right answer differs:

* **In a container (Codespaces, devcontainer, Docker)** — 0.0.0.0. The port
  is reached through a forwarding proxy that connects from outside the
  container's loopback, so a 127.0.0.1 bind is unreachable and the forwarded
  URL serves nothing (a blank page). The container boundary is the security
  boundary here, and Codespaces ports are private to the account by default.
* **Anywhere else** — 127.0.0.1. This API is unauthenticated and can load
  models and read files, so a LAN-visible bind must be deliberate.

Either default can be overridden with TAGM_HOST or --host.

Access logging is off by default because TAGM's frontend polls the
status/progress endpoints every ~2s, which otherwise floods the terminal
with access-log lines. Application-level logs (loads, errors, batch
progress) still print. Pass --access-log to enable per-request lines.
"""
from __future__ import annotations

import argparse
import os


def _in_container() -> bool:
    """True when a forwarding proxy, not the browser, connects to us.

    Codespaces and devcontainers reach the app through a proxy outside the
    container's network namespace, so binding to loopback makes the forwarded
    URL resolve but return nothing — the blank-page symptom. Docker is the
    same story via published ports.
    """
    if os.environ.get("CODESPACES") or os.environ.get("CODESPACE_NAME"):
        return True
    if os.environ.get("REMOTE_CONTAINERS") or os.environ.get("DEVCONTAINER"):
        return True
    return os.path.exists("/.dockerenv")


def default_host() -> str:
    """0.0.0.0 inside a container, loopback otherwise. See module docstring."""
    env = os.environ.get("TAGM_HOST")
    if env:
        return env
    return "0.0.0.0" if _in_container() else "127.0.0.1"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Start the TAGM FastAPI server.",
    )
    parser.add_argument("--host", default=default_host(),
                        help="Host to bind (0.0.0.0 in a container so port "
                             "forwarding works, else 127.0.0.1)")
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

    # Say which interface we bound and why, so an unreachable forwarded URL
    # is diagnosable from the log instead of presenting as a blank page.
    print(f"-> binding {args.host}:{args.port}"
          + ("  (container detected — 0.0.0.0 so port forwarding works)"
             if args.host == "0.0.0.0" and _in_container()
             else "  (loopback only; set TAGM_HOST=0.0.0.0 to expose)"
                  if args.host.startswith("127.") else ""))

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
