"""A Shared Crew guest never names a location on the host's disk.

One checkpoint (``guest_scope.guest_host_path_denial``, run by the slot-ownership
middleware) refuses a guest write whose query or body holds a field of
``guest_scope.HOST_PATH_FIELDS``, and a sweep fails when a guest-open write handler
reads a path-like field that is neither in that list nor reviewed here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from test_shared_crew_guest_parity import GUEST

from kiro_crew.shared_crew import guest_scope
from kiro_crew.shared_crew.contracts import ACTOR_REQUEST_KEY

# ── host-path fields: one checkpoint for the class ──


@pytest.fixture(autouse=True)
def _preview_on(_floor_monkeypatch: pytest.MonkeyPatch) -> None:
    _floor_monkeypatch.setenv("KIROCREW_SHARED_CREW_PREVIEW", "1")


def _guest_checkpoint_app(*, guest: bool, seen: dict[str, Any]) -> web.Application:
    from kiro_crew.dashboard.handlers._shared_request_body import read_bounded_json
    from kiro_crew.dashboard.slot_ownership import slot_ownership_middleware

    @web.middleware
    async def mw(request: web.Request, handler: Any) -> Any:
        request["app"] = ""
        request["user"] = "guest@example.test"
        request["is_dashboard_user"] = True
        if guest:
            request[ACTOR_REQUEST_KEY] = GUEST
        else:
            request["internal_auth"] = True
        return await handler(request)

    async def bounded(request: web.Request) -> web.Response:
        body, err = await read_bounded_json(request, max_bytes=1 << 20)
        seen["bounded"] = body
        return err or web.json_response({"ok": True})

    async def whole(request: web.Request) -> web.Response:
        seen["json"] = await request.json()
        return web.json_response({"ok": True})

    async def upload(request: web.Request) -> web.Response:
        seen["upload"] = [part.name async for part in await request.multipart()]
        return web.json_response({"ok": True})

    async def post_fields(request: web.Request) -> web.Response:
        form = await request.post()
        seen["post"] = {k: v if isinstance(v, str) else "<file>" for k, v in form.items()}
        return web.json_response({"ok": True})

    app = web.Application(middlewares=[mw, slot_ownership_middleware])
    app["state"] = MagicMock(owner_id="owner", _slots={})
    app.router.add_post("/api/x/post", post_fields)
    app.router.add_post("/api/x/bounded", bounded)
    app.router.add_post("/api/x/whole", whole)
    app.router.add_put("/api/x/whole", whole)
    app.router.add_post("/api/x/upload", upload)
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"source_path": "/srv/owner/notes.md", "content": "x"},
        {"spec": "/srv/owner/TASK.md"},
        {"spec": "//attacker/share/TASK.md", "source": "text", "input": "hi"},
        {"workspace_dir": "\\\\attacker\\share\\work"},
        {"workspace_dir": "/srv/owner"},
        {"path": "/srv/owner/.ssh/config"},
        {"project_dir": "/srv/owner/code"},
        {"cwd": "/"},
        {"Source_Path": "/etc/hosts"},
        {"steering_dirs": ["/srv/owner/steer"]},
    ],
)
async def test_a_guest_write_naming_a_host_path_is_refused_at_the_checkpoint(
    body: dict[str, Any],
) -> None:
    """The class, once: whatever the route, a host-path field is owner-only."""
    seen: dict[str, Any] = {}
    async with TestClient(TestServer(_guest_checkpoint_app(guest=True, seen=seen))) as client:
        for route in ("/api/x/bounded", "/api/x/whole"):
            resp = await client.post(route, json=body)
            payload = await resp.json()
            assert (resp.status, payload["code"]) == (403, "owner_only"), (route, body)
    assert seen == {}, "the handler must not run"


@pytest.mark.asyncio
async def test_a_guest_host_path_in_the_query_or_a_form_or_a_plain_text_body_is_refused() -> None:
    seen: dict[str, Any] = {}
    async with TestClient(TestServer(_guest_checkpoint_app(guest=True, seen=seen))) as client:
        responses = [
            await client.post("/api/x/bounded?workspace_dir=/srv/owner", json={}),
            await client.put("/api/x/whole?path=/etc/hosts", data=b""),
            await client.post("/api/x/whole", data={"source_path": "/etc/hosts"}),
            await client.post(
                "/api/x/whole",
                data=b'{"spec": "/etc/hosts"}',
                headers={"Content-Type": "text/plain"},
            ),
        ]
    assert [r.status for r in responses] == [403, 403, 403, 403]
    assert seen == {}


@pytest.mark.asyncio
async def test_a_guest_keeps_typing_task_text_and_uploading_attachments() -> None:
    """Content is not a location: inline specs, task text, attachment objects and uploads pass."""
    seen: dict[str, Any] = {}
    bodies = [
        {"spec": "__inline__:# Task\nwrite /srv/owner/x", "input": "hi", "workspace_dir": ""},
        {"input": "read /etc/hosts", "source": "text", "workspace_dir": None, "path": ""},
        {"text": "see this", "attachments": [{"path": "/tmp/uploads/a.png", "name": "a.png"}]},
        ["not", "an", "object"],
    ]
    async with TestClient(TestServer(_guest_checkpoint_app(guest=True, seen=seen))) as client:
        for body in bodies:
            seen.clear()
            resp = await client.post("/api/x/bounded", json=body)
            assert resp.status in (200, 400), body
            seen.clear()
            resp = await client.post("/api/x/whole", json=body)
            assert resp.status == 200, body
            # The handler reads the body the checkpoint already read, unchanged.
            assert seen["json"] == body
        seen.clear()
        form = aiohttp.FormData()
        form.add_field("file", b"png-bytes", filename="a.png", content_type="image/png")
        resp = await client.post("/api/x/upload", data=form)
        assert resp.status == 200
        assert seen["upload"] == ["file"]
        seen.clear()
        resp = await client.post("/api/x/bounded", json={"input": "t"})
        assert (resp.status, seen["bounded"]) == (200, {"input": "t"})


@pytest.mark.asyncio
async def test_the_owner_is_not_held_to_the_host_path_fields() -> None:
    seen: dict[str, Any] = {}
    async with TestClient(TestServer(_guest_checkpoint_app(guest=False, seen=seen))) as client:
        resp = await client.post("/api/x/whole", json={"source_path": "/srv/owner/n.md"})
        assert resp.status == 200
    assert seen["json"] == {"source_path": "/srv/owner/n.md"}


@pytest.mark.asyncio
async def test_a_guest_body_over_the_apps_size_limit_is_answered_413() -> None:
    seen: dict[str, Any] = {}
    async with TestClient(TestServer(_guest_checkpoint_app(guest=True, seen=seen))) as client:
        resp = await client.post(
            "/api/x/whole",
            data=b'{"pad": "' + b"a" * (2 * 1024 * 1024) + b'"}',
            headers={"Content-Type": "application/json"},
        )
    assert resp.status == 413
    assert seen == {}


def test_the_host_path_list_names_the_fields_the_findings_were_about() -> None:
    assert {
        "source_path",
        "spec",
        "spec_path",
        "workspace_dir",
        "project_dir",
        "path",
        "cwd",
    } <= guest_scope.HOST_PATH_FIELDS


#: Path-like fields a guest-open write handler reads that are NOT host locations, each
#: with the reason. Anything else path-like must be in ``HOST_PATH_FIELDS``.
_REVIEWED_NOT_HOST_PATHS = {
    "filename": "a download's suggested basename; the handler writes nothing from it",
    "source_uri": "an identifier recorded on a knowledge document; never opened",
}

_PATH_TOKENS = frozenset(
    {
        "path",
        "paths",
        "dir",
        "dirs",
        "directory",
        "file",
        "files",
        "filename",
        "root",
        "cwd",
        "spec",
        "yaml",
        "workdir",
        "uri",
        "location",
    }
)
_FIELD_READ = re.compile(
    r"(?:\b(?:body|data|payload|params|j|req|raw)\b|query)(?:\.get\(|\[)\s*[\"']([A-Za-z_]+)[\"']"
)


def _is_path_like(name: str) -> bool:
    return bool(_PATH_TOKENS & set(name.lower().split("_")))


@pytest.mark.asyncio
async def test_every_guest_open_write_handler_field_is_classified_as_host_path_or_reviewed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Walk every guest-open write route's handler for the fields it reads.

    A path-like field must be refused at the checkpoint (``HOST_PATH_FIELDS``) or be
    listed above with the reason it names no location. A new route that reads
    ``body["log_dir"]`` fails here until someone decides which.
    """
    import inspect

    from test_dashboard_server_startup_coverage import _dashboard

    unclassified: dict[str, set[str]] = {}
    walked = 0
    async with _dashboard(tmp_path, monkeypatch) as (runner, _state, _spies):
        for route in runner.app.router.routes():
            if route.resource is None or route.method in ("HEAD", "OPTIONS", "GET"):
                continue
            path = route.resource.canonical
            if guest_scope.is_owner_only_path(path, route.method):
                continue
            handler = route.handler
            while hasattr(handler, "__wrapped__"):
                handler = handler.__wrapped__
            try:
                source = inspect.getsource(handler)
            except (OSError, TypeError):
                continue
            walked += 1
            names = {n.lower() for n in _FIELD_READ.findall(source) if _is_path_like(n)}
            names -= guest_scope.HOST_PATH_FIELDS | set(_REVIEWED_NOT_HOST_PATHS)
            if names:
                unclassified[f"{route.method} {path}"] = names
    assert walked > 100
    assert unclassified == {}


def test_the_reviewed_not_host_path_fields_are_not_also_refused() -> None:
    assert not set(_REVIEWED_NOT_HOST_PATHS) & guest_scope.HOST_PATH_FIELDS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type",
    ["image/png", "audio/mpeg", "video/mp4", "application/octet-stream", "text/plain"],
)
async def test_a_guest_host_path_in_a_body_labelled_with_any_type_is_refused(
    content_type: str,
) -> None:
    """A handler's ``request.json()`` parses whatever the client labelled the body."""
    seen: dict[str, Any] = {}
    async with TestClient(TestServer(_guest_checkpoint_app(guest=True, seen=seen))) as client:
        resp = await client.post(
            "/api/x/whole",
            data=b'{"project_dir": "/srv/owner/code"}',
            headers={"Content-Type": content_type},
        )
        payload = await resp.json()
        assert (resp.status, payload["code"]) == (403, "owner_only")
        resp = await client.post(
            "/api/x/whole",
            data=b'{"input": "hello"}',
            headers={"Content-Type": content_type},
        )
        assert resp.status == 200
    assert seen["json"] == {"input": "hello"}


@pytest.mark.parametrize(
    "field", ["root", "project_dir", "cwd", "paths", "file", "source_path", "workspace_dir"]
)
def test_the_host_path_list_covers_every_field_the_findings_named(field: str) -> None:
    assert field in guest_scope.HOST_PATH_FIELDS


def test_an_upload_directory_swapped_for_a_symlink_serves_a_guest_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store root is lstat-ed unresolved: a link there is refused, not followed."""
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "home"))
    (tmp_path / "home" / "sessions").mkdir(parents=True)
    real = tmp_path / "elsewhere"
    real.mkdir()
    (real / "pic.png").write_bytes(b"png")
    uploads = tmp_path / "home" / "uploads"
    uploads.mkdir()
    (uploads / "pic.png").write_bytes(b"png")
    assert guest_scope.attachment_read(str(uploads / "pic.png"), 1024) is not None

    (uploads / "pic.png").unlink()
    uploads.rmdir()
    uploads.symlink_to(real, target_is_directory=True)
    assert guest_scope.attachment_read(str(uploads / "pic.png"), 1024) is None
    assert guest_scope.attachment_read(str(real / "pic.png"), 1024) is None


def _multipart(boundary: str, *parts: tuple[str, str | None, bytes]) -> bytes:
    """A multipart body: ``(field name, filename or None, content)`` parts."""
    out = b""
    for name, filename, content in parts:
        disposition = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        out += f"--{boundary}\r\nContent-Disposition: {disposition}\r\n\r\n".encode()
        out += content + b"\r\n"
    return out + f"--{boundary}--\r\n".encode()


_JSON_PATH = b'{"project_dir": "/srv/owner/code"}'
_UPLOAD = _multipart("XbX", ("file", "a.png", b"\x89PNG" + b"0" * 5000))
_HIDDEN = _multipart("XbX", ("file", "a.png", b"png"), ("project_dir", None, b"/srv/owner"))
_NO_CONTENT_TYPE: dict[str, str] = {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (_JSON_PATH, "multipart/form-data; boundary=XbX"),
        (_JSON_PATH, "multipart/form-data"),
        (_JSON_PATH, None),
        (_HIDDEN, "multipart/form-data; boundary=XbX"),
        # Labelled with one boundary, split on another further in.
        (
            _multipart("Y", ("file", "a.png", b"junk\r\n" + _HIDDEN)),
            "multipart/form-data; boundary=XbX",
        ),
        # A path sent as a "file" part under a name that is not an upload's.
        (
            _multipart("XbX", ("workspace_dir", "x.txt", b"/srv")),
            "multipart/form-data; boundary=XbX",
        ),
    ],
)
async def test_a_host_path_the_handlers_parser_reads_is_refused_whatever_the_label(
    body: bytes, content_type: str | None
) -> None:
    seen: dict[str, Any] = {}
    headers = {"Content-Type": content_type} if content_type else {}
    async with TestClient(TestServer(_guest_checkpoint_app(guest=True, seen=seen))) as client:
        resp = await client.post(
            "/api/x/whole",
            data=body,
            headers=headers,
            skip_auto_headers=() if content_type else ("Content-Type",),
        )
        payload = await resp.json()
    assert (resp.status, payload["code"]) == (403, "owner_only")
    assert seen == {}


@pytest.mark.asyncio
async def test_a_real_file_upload_still_passes() -> None:
    seen: dict[str, Any] = {}
    big = _multipart("XbX", ("file", "big.bin", b"\0" * (900 * 1024)))
    async with TestClient(TestServer(_guest_checkpoint_app(guest=True, seen=seen))) as client:
        resp = await client.post(
            "/api/x/upload",
            data=_UPLOAD,
            headers={"Content-Type": "multipart/form-data; boundary=XbX"},
        )
        assert resp.status == 200
        assert seen["upload"] == ["file"]
        seen.clear()
        resp = await client.post(
            "/api/x/upload", data=big, headers={"Content-Type": "multipart/form-data; boundary=XbX"}
        )
        assert resp.status == 200
        # A text field that is empty is not a location.
        seen.clear()
        empty = _multipart("XbX", ("file", "a.png", b"png"), ("project_dir", None, b""))
        resp = await client.post(
            "/api/x/upload",
            data=empty,
            headers={"Content-Type": "multipart/form-data; boundary=XbX"},
        )
        assert resp.status == 200
        assert seen["upload"] == ["file", "project_dir"]


@pytest.mark.asyncio
async def test_a_body_that_parses_as_nothing_is_let_through() -> None:
    seen: dict[str, Any] = {}
    async with TestClient(TestServer(_guest_checkpoint_app(guest=True, seen=seen))) as client:
        resp = await client.post(
            "/api/x/bounded",
            data=b"\x00\x01\x02 not a document",
            headers={"Content-Type": "image/png"},
        )
    assert resp.status in (400, 415)  # the handler's own refusal, not ours


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("charset", "encode"),
    [
        ("utf-8-sig", lambda text: b"\xef\xbb\xbf" + text.encode("utf-8")),
        ("utf-16", lambda text: text.encode("utf-16")),
        ("utf-16-le", lambda text: text.encode("utf-16-le")),
        ("utf-16-be", lambda text: text.encode("utf-16-be")),
        ("utf-8", lambda text: text.encode("utf-8")),
    ],
)
async def test_a_json_body_in_any_encoding_the_handler_parses_is_judged_the_same(
    charset: str, encode: Any
) -> None:
    """The checkpoint calls ``request.json()``: whatever the handler can parse, it sees."""
    body = encode('{"project_dir": "/srv/owner/code"}')
    headers = {"Content-Type": f"application/json; charset={charset}"}
    seen: dict[str, Any] = {}
    # The owner's handler parses it (so a guest's handler would too) ...
    async with TestClient(TestServer(_guest_checkpoint_app(guest=False, seen=seen))) as client:
        assert (await client.post("/api/x/whole", data=body, headers=headers)).status == 200
    assert seen["json"] == {"project_dir": "/srv/owner/code"}
    # ... so the checkpoint refuses it for a guest.
    seen.clear()
    async with TestClient(TestServer(_guest_checkpoint_app(guest=True, seen=seen))) as client:
        resp = await client.post("/api/x/whole", data=body, headers=headers)
        assert (resp.status, (await resp.json())["code"]) == (403, "owner_only")
    assert seen == {}


@pytest.mark.asyncio
async def test_the_handler_still_receives_a_body_the_checkpoint_has_read() -> None:
    """Every way a handler reads a request gets the same bytes after the checkpoint."""
    seen: dict[str, Any] = {}
    note = {"input": "hello", "attachments": [{"path": "/tmp/uploads/a.png"}]}
    form = _multipart("XbX", ("input", None, b"hello"), ("file", "a.png", b"png-bytes"))
    async with TestClient(TestServer(_guest_checkpoint_app(guest=True, seen=seen))) as client:
        assert (await client.post("/api/x/whole", json=note)).status == 200
        assert (await client.post("/api/x/bounded", json=note)).status == 200
        assert (
            await client.post(
                "/api/x/post",
                data=form,
                headers={"Content-Type": "multipart/form-data; boundary=XbX"},
            )
        ).status == 200
        assert (
            await client.post(
                "/api/x/upload",
                data=form,
                headers={"Content-Type": "multipart/form-data; boundary=XbX"},
            )
        ).status == 200
        assert (await client.post("/api/x/post", data={"input": "hello", "n": "1"})).status == 200
    assert seen["json"] == note
    assert seen["bounded"] == note
    assert seen["upload"] == ["input", "file"]
    assert seen["post"] == {"input": "hello", "n": "1"}
