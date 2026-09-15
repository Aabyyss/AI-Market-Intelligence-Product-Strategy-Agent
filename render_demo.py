"""Render docs/demo.html into a real MP4 — no screen recorder, no editing.

The reel in `docs/demo.html` plays itself in real time (open it, press F11,
record the window). That is the zero-tooling path, but it depends on the
machine keeping up and it makes every take slightly different. This script
takes the other path: it drives headless Chrome over the DevTools Protocol,
asks the page to freeze at an exact point in virtual time for each frame, and
encodes the result. Same input, same video, every time.

    python render_demo.py                 # docs/demo.mp4, 30 fps, 1280x720
    python render_demo.py --fps 24        # smaller file
    python render_demo.py --keep-frames   # leave the PNGs behind to inspect

Needs a Chromium browser (Chrome or Edge) and an ffmpeg binary. ffmpeg is
located from PATH first, then from the `imageio-ffmpeg` wheel, which is the
easy install on a machine that doesn't have one:

    pip install imageio-ffmpeg

No other dependencies — the CDP client below is ~40 lines over `websockets`.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent
REEL = ROOT / "docs" / "demo.html"

WIDTH, HEIGHT = 1280, 720

# Where to look for a browser, most ordinary locations first. Edge ships with
# Windows and is a Chromium browser, so it is a perfectly good fallback.
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/microsoft-edge",
]


def find_browser(explicit: str | None = None) -> str:
    if explicit:
        path = shutil.which(explicit) or explicit
        if Path(path).exists():
            return path
        raise SystemExit(f"browser not found: {explicit}")
    for name in ("google-chrome", "chromium", "chromium-browser", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in BROWSERS:
        if Path(candidate).exists():
            return candidate
    raise SystemExit(
        "no Chrome/Edge found — pass --browser /path/to/chrome "
        "(any Chromium build supports the DevTools Protocol)"
    )


def find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # noqa: PLC0415 — optional, only needed here

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - depends on the machine
        raise SystemExit(
            "no ffmpeg found — install one, or `pip install imageio-ffmpeg` "
            "to get a self-contained binary"
        ) from None


def reel_stamp(path: Path = REEL) -> str:
    """Identity of the reel source — any edit to it changes this."""
    return "reel-" + hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class CDP:
    """Minimal DevTools Protocol client: one call, one reply."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self._id = 0

    async def call(self, method: str, **params):
        self._id += 1
        awaited = self._id
        await self.ws.send(json.dumps({"id": awaited, "method": method, "params": params}))
        while True:
            message = json.loads(await self.ws.recv())
            if message.get("id") != awaited:
                continue  # an event, not our reply
            if "error" in message:
                raise RuntimeError(f"{method} failed: {message['error']}")
            return message.get("result", {})

    async def evaluate(self, expression: str, await_promise: bool = False):
        result = await self.call(
            "Runtime.evaluate",
            expression=expression,
            returnByValue=True,
            awaitPromise=await_promise,
        )
        return result.get("result", {}).get("value")


async def connect(port: int, timeout: float = 60.0) -> CDP:
    """Wait for the browser's page target, then attach to it."""
    deadline = time.monotonic() + timeout
    target = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as r:
                targets = json.load(r)
            pages = [t for t in targets if t.get("type") == "page"]
            if pages:
                target = pages[0]
                break
        except Exception:
            pass
        await asyncio.sleep(0.4)
    if not target:
        raise SystemExit("browser started but never exposed a page target")

    ws = await websockets.connect(target["webSocketDebuggerUrl"], max_size=None)
    client = CDP(ws)
    await client.call("Page.enable")
    await client.call("Runtime.enable")
    return client


async def open_reel(client: CDP, url: str, timeout: float = 60.0) -> None:
    await client.call(
        "Emulation.setDeviceMetricsOverride",
        width=WIDTH,
        height=HEIGHT,
        deviceScaleFactor=1,
        mobile=False,
    )
    await client.call("Page.navigate", url=url)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready = await client.evaluate("!!(window.__reel && window.__reel.seek)")
        if ready:
            return
        await asyncio.sleep(0.2)
    raise SystemExit(f"{url} never exposed window.__reel — is ?render=1 missing?")


async def grab(
    client: CDP, path: Path, settle: int, fmt: str = "png", quality: int = 95
) -> None:
    if settle:
        # Let the seek's style flush and the paint it caused land before the
        # capture; rAF is the paint boundary, so one frame is usually enough.
        await client.evaluate(
            "new Promise(r => { let n = %d; (function step() { "
            "if (n-- <= 0) return r(1); requestAnimationFrame(step); })(); })" % settle,
            await_promise=True,
        )
    params = {"format": fmt, "captureBeyondViewport": False}
    if fmt in ("jpeg", "webp"):
        params["quality"] = quality
    shot = await client.call("Page.captureScreenshot", **params)
    path.write_bytes(base64.b64decode(shot["data"]))


async def render(
    frames_dir: Path, fps: int, keep: bool, limit: float | None,
    settle: int, fmt: str, quality: int, browser_arg: str | None,
) -> float:
    url = REEL.resolve().as_uri() + "?render=1"
    browser = find_browser(browser_arg)
    port = free_port()
    profile = Path(tempfile.mkdtemp(prefix="reel-profile-"))
    log = profile / "browser.log"
    ext = "jpg" if fmt == "jpeg" else fmt
    # jpeg is ~2x faster to capture than png and the difference is invisible
    # once h264 has encoded both, which matters at ~2000 frames.
    if fmt == "jpeg":
        print(f"frames  : jpeg q{quality} (use --format png for lossless source)")

    print(f"browser : {browser}")
    print(f"reel    : {url}")
    with log.open("wb") as handle:
        proc = subprocess.Popen(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                "--no-default-browser-check",
                "--force-device-scale-factor=1",
                "--run-all-compositor-stages-before-draw",
                f"--window-size={WIDTH},{HEIGHT}",
                f"--remote-debugging-port={port}",
                f"--user-data-dir={profile}",
                "about:blank",
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
        )

    try:
        client = await connect(port)
        await open_reel(client, url)
        total_ms = float(await client.evaluate("window.__reel.total"))
        if limit:
            total_ms = min(total_ms, limit * 1000)
        count = max(1, int(round(total_ms / 1000 * fps)))
        print(f"duration: {total_ms / 1000:.1f}s -> {count} frames at {fps} fps")

        started = time.monotonic()
        for i in range(count):
            await client.evaluate(f"window.__reel.seek({i * 1000 / fps:.3f})")
            await grab(client, frames_dir / f"f{i:05d}.{ext}", settle, fmt, quality)
            if i and i % 150 == 0:
                rate = i / (time.monotonic() - started)
                print(f"  {i:>5}/{count} frames | {rate:.0f} fps render | "
                      f"{time.monotonic() - started:5.0f}s")
        print(f"captured {count} frames in {time.monotonic() - started:.0f}s")

        # A poster frame (t=1.4s, after the title has revealed) makes the video
        # embeddable in a README without autoplaying it.
        await client.evaluate("window.__reel.seek(1400)")
        await grab(client, ROOT / "docs" / "demo-poster.png", settle)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover - rare
            proc.kill()
        if not keep:
            shutil.rmtree(profile, ignore_errors=True)
        else:
            print(f"browser profile kept at {profile}")

    return total_ms / 1000


def encode(frames_dir: Path, fps: int, out: Path, ext: str) -> None:
    ffmpeg = find_ffmpeg()
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-stats",
        "-framerate", str(fps),
        "-i", str(frames_dir / f"f%05d.{ext}"),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        # Stamp the video with the reel it came from. Editing the reel's text or
        # timings invalidates the committed video, and this is what lets the
        # test suite notice instead of shipping a stale demo.
        "-metadata", f"comment={reel_stamp()}",
        str(out),
    ]
    print(f"encoding {out.name} ({reel_stamp()}) ...")
    subprocess.run(cmd, check=True)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({size_mb:.1f} MB)")


def main() -> None:
    if not REEL.exists():
        raise SystemExit(f"missing {REEL}")
    frames_dir = ROOT / "data" / "demo_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    started = time.monotonic()
    seconds = asyncio.run(
        render(
            frames_dir, ARGS.fps, ARGS.keep_frames,
            ARGS.only_seconds, ARGS.settle, ARGS.format, ARGS.quality, ARGS.browser,
        )
    )
    encode(frames_dir, ARGS.fps, ARGS.out, "jpg" if ARGS.format == "jpeg" else ARGS.format)
    if not ARGS.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"done in {time.monotonic() - started:.0f}s | {seconds:.1f}s of video")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fps", type=int, default=30, help="frames per second (default 30)")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "demo.mp4",
        help="output file (default docs/demo.mp4)",
    )
    parser.add_argument("--browser", help="path to a Chromium binary")
    parser.add_argument(
        "--format",
        choices=("jpeg", "png"),
        default="jpeg",
        help="frame capture format (default jpeg q95: lossless png is ~2x "
        "slower to capture for no visible gain after encoding)",
    )
    parser.add_argument("--quality", type=int, default=95, help="jpeg quality (default 95)")
    parser.add_argument(
        "--settle",
        type=int,
        default=1,
        help="paint frames to wait for before each capture (default 1; 0 is "
        "faster but can sample a stale frame)",
    )
    parser.add_argument(
        "--only-seconds",
        type=float,
        help="render only the first N seconds — a smoke test for the pipeline",
    )
    parser.add_argument("--keep-frames", action="store_true", help="keep the PNG frames")
    ARGS = parser.parse_args()
    main()
