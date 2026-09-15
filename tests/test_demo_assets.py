"""Tests for the demo assets: the reel, the rendered video, the poster.

Nothing else in the suite touches `docs/`. Those files are easy to break
silently in ways no other test would notice: add one `@import` and the reel
stops working offline, change a scene duration and the committed video is
instantly stale, re-encode at the wrong size and it only shows up on the
project page. They are checked here without loading a browser or a codec —
the MP4 boxes and the PNG header are parsed directly, so `pytest` needs no
extra dependency to run these.
"""

from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REEL = ROOT / "docs" / "demo.html"
POSTER = ROOT / "docs" / "demo-poster.png"
VIDEO = ROOT / "docs" / "demo.mp4"
RENDERER = ROOT / "render_demo.py"

WIDTH, HEIGHT, FPS = 1280, 720, 30


def reel() -> str:
    return REEL.read_text(encoding="utf-8")


def scene_durations_ms(html: str) -> list[int]:
    """Pull the scene table out of the reel's own script."""
    return [int(ms) for ms in re.findall(r'\{\s*id:\s*"s\d+",\s*ms:\s*(\d+)', html)]


def png_size(path: Path) -> tuple[int, int]:
    """Width/height straight out of the IHDR chunk — no image library."""
    data = path.read_bytes()[:24]
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    assert data[12:16] == b"IHDR"
    return struct.unpack(">II", data[16:24])


def mp4_shape(path: Path) -> dict:
    """Read duration, frame rate and pixel size from the container's own boxes."""
    data = path.read_bytes()
    assert data[4:8] == b"ftyp"
    ftyp_end = struct.unpack(">I", data[:4])[0]

    head = data.find(b"mvhd")
    assert head > 0, "no mvhd: not a playable MP4"
    version = data[head + 4]
    if version == 0:
        timescale, duration = struct.unpack(">II", data[head + 16:head + 24])
    else:
        timescale, duration = struct.unpack(">IQ", data[head + 24:head + 36])

    mdhd = data.find(b"mdhd")
    media_timescale = struct.unpack(">I", data[mdhd + 16:mdhd + 20])[0]
    stts = data.find(b"stts")
    entries = struct.unpack(">I", data[stts + 8:stts + 12])[0]
    samples, delta = struct.unpack(">II", data[stts + 12:stts + 20])

    avc1 = data.find(b"avc1", ftyp_end)  # 'avc1' also appears in ftyp's brands
    assert avc1 > 0, "no avc1 sample entry: expected H.264"
    width, height = struct.unpack(">HH", data[avc1 + 28:avc1 + 32])

    return {
        "duration": duration / timescale,
        "fps": media_timescale / delta,
        "frames": samples,
        "stts_entries": entries,
        "size": (width, height),
        "faststart": data.find(b"moov") < data.find(b"mdat"),
    }


@pytest.mark.parametrize("asset", [REEL, POSTER, VIDEO, RENDERER])
def test_demo_assets_exist(asset: Path):
    assert asset.exists(), f"missing demo asset: {asset.relative_to(ROOT)}"


def test_reel_loads_nothing_from_outside_itself():
    """The whole point is that it plays offline from a single file."""
    html = reel()
    external = re.findall(
        r"""(?:src|href)\s*=\s*["'](?!#|data:)[^"']+["']""", html
    )
    assert external == [], f"reel references external resources: {external}"
    assert "@import" not in html
    assert "fonts.googleapis" not in html


def test_reel_exposes_the_hook_the_renderer_drives():
    """render_demo.py depends on these three things; renaming one breaks the video."""
    html = reel()
    for token in ('location.search).has("render")', "function seek(", "window.__reel", "getAnimations"):
        assert token in html, f"reel no longer exposes {token!r}"


def test_reel_scene_table_is_well_formed():
    durations = scene_durations_ms(reel())
    assert len(durations) == 7, f"expected 7 scenes, parsed {durations}"
    assert all(ms >= 1000 for ms in durations), durations
    assert mp4_shape(VIDEO)["duration"] == pytest.approx(sum(durations) / 1000, abs=0.1)


def test_committed_video_is_not_stale():
    """The video is stamped with the hash of the reel it was rendered from.

    Edit any word or timing in docs/demo.html and this fails, because the
    committed video now shows something the reel no longer claims.
    """
    stamp = "reel-" + hashlib.sha256(REEL.read_bytes()).hexdigest()[:16]
    assert stamp.encode() in VIDEO.read_bytes(), (
        "docs/demo.mp4 was rendered from an older docs/demo.html — "
        "re-run `python render_demo.py`"
    )


def test_committed_video_is_720p_h264_at_the_reel_frame_rate():
    shape = mp4_shape(VIDEO)
    assert shape["size"] == (WIDTH, HEIGHT)
    assert shape["fps"] == pytest.approx(FPS, abs=0.01)
    assert shape["frames"] == pytest.approx(shape["duration"] * FPS, abs=2)
    assert shape["stts_entries"] == 1, "variable frame rate: the renderer emits CFR"
    assert shape["faststart"], "moov must precede mdat so the file streams/embeds"


def test_poster_is_a_reel_sized_frame():
    """The poster is embedded in the README, so a wrong size shows up there."""
    assert png_size(POSTER) == (WIDTH, HEIGHT)


def test_renderer_still_points_at_the_reel():
    """Static check, so the suite needs neither a browser nor `websockets`."""
    source = RENDERER.read_text(encoding="utf-8")
    assert '"?render=1"' in source
    assert 'default=30' in source
    assert 'docs" / "demo.mp4"' in source
