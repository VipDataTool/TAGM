"""Server-Sent Events broker for TAGM.

Replaces polling-based state observation with a publish/subscribe event
model.  Every state transition in the application (model loaded, module
completed, export ready, progress tick) becomes a typed JSON event pushed
to every connected client exactly once.

Design constraints:
  - Single-user, low-tab: TAGM runs locally, so the subscriber set is
    tiny.  The broker is kept simple — no persistence, no partitioning.
  - Thread-safe publication: events originate from background threads
    (_load_worker, module threads, export threads).  publish() is
    guarded by a lock.
  - Snapshot on connect: a client that connects (or reconnects) after
    a state change must still learn the current state.  The broker
    maintains a small snapshot of the most recent event per type.
  - No asyncio dependency in publish(): background threads are plain
    threading.Thread, not asyncio tasks.  publish() writes to a
    thread-safe queue; the SSE generator drains it asynchronously.

Usage (server side):
    from tagm.service.events import broker

    # In any thread:
    broker.publish("model_loaded", {"model": "llama-3.2"})

    # In FastAPI:
    @app.get("/api/events")
    async def events(request: Request):
        return broker.sse_response(request)
"""
from __future__ import annotations

import json
import time
import asyncio
import logging
import threading
from collections import deque
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger("tagm")


class EventBroker:
    """Thread-safe pub/sub broker with SSE delivery.

    Publishers (any thread) call broker.publish(event_type, payload).
    Subscribers are SSE connections; each gets its own asyncio.Queue
    bridged from the synchronous publish() via call_soon_threadsafe.
    """

    # Types whose most recent event is kept in the snapshot.
    SNAPSHOT_TYPES = {
        "model_loaded", "model_error",
        "batch_done", "export_ready", "export_error",
    }

    # Maximum progress events kept in the trailing buffer.
    _PROGRESS_TAIL = 50

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers: list[tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = []
        self._snapshot: dict[str, dict] = {}
        self._progress_tail: deque = deque(maxlen=self._PROGRESS_TAIL)

    # ── Publishing (any thread) ──────────────────────────────────

    def publish(self, event_type: str, payload: dict | None = None):
        """Broadcast an event to all connected clients.

        Thread-safe.  Can be called from background threads, FastAPI
        request handlers, or anywhere else.
        """
        evt = {
            "type": event_type,
            "time": time.time(),
            **(payload or {}),
        }

        with self._lock:
            # Update snapshot
            if event_type in self.SNAPSHOT_TYPES:
                self._snapshot[event_type] = evt
                # model_loaded clears model_error and vice versa
                if event_type == "model_loaded":
                    self._snapshot.pop("model_error", None)
                elif event_type == "model_error":
                    self._snapshot.pop("model_loaded", None)

            if event_type == "progress":
                self._progress_tail.append(evt)

            # Push to all subscriber queues
            dead = []
            for i, (q, loop) in enumerate(self._subscribers):
                try:
                    loop.call_soon_threadsafe(q.put_nowait, evt)
                except Exception:
                    dead.append(i)

            # Clean up dead subscribers (loop closed, etc.)
            for i in reversed(dead):
                self._subscribers.pop(i)

    # ── Snapshot (for reconnecting clients) ──────────────────────

    def snapshot(self) -> list[dict]:
        """Return events a freshly-connected client needs to reach
        current state.  Includes the most recent terminal events and
        the trailing progress buffer."""
        with self._lock:
            events = []
            for evt in self._snapshot.values():
                events.append(evt)
            for evt in self._progress_tail:
                events.append(evt)
            return events

    # ── SSE Response (FastAPI endpoint helper) ───────────────────

    def sse_response(self, request: Request) -> StreamingResponse:
        """Return a StreamingResponse that pushes SSE events.

        Usage:
            @app.get("/api/events")
            async def events(request: Request):
                return broker.sse_response(request)
        """
        async def generate():
            loop = asyncio.get_event_loop()
            q: asyncio.Queue = asyncio.Queue()

            with self._lock:
                self._subscribers.append((q, loop))

            try:
                # Replay snapshot so the client reaches current state
                for evt in self.snapshot():
                    yield f"data: {json.dumps(evt)}\n\n"

                # Send a "connected" heartbeat
                yield f"data: {json.dumps({'type': 'connected', 'time': time.time()})}\n\n"

                # Stream events as they arrive
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        evt = await asyncio.wait_for(q.get(), timeout=30.0)
                        yield f"data: {json.dumps(evt)}\n\n"
                    except asyncio.TimeoutError:
                        # Keepalive comment to prevent proxy/browser timeout
                        yield ": keepalive\n\n"
            finally:
                with self._lock:
                    self._subscribers = [
                        (sq, sl) for sq, sl in self._subscribers if sq is not q
                    ]

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


# Module-level singleton
broker = EventBroker()
