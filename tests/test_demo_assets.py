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
    assert len(durations) == 8, f"expected 8 scenes, parsed {durations}"
    assert all(ms >= 1000 for ms in durations), durations
    assert mp4_shape(VIDEO)["duration"] == pytest.approx(sum(durations) / 1000, abs=0.1)


def test_committed_video_is_not_stale():
    """The video is stamped with the hash of the reel it was rendered from.

    Edit any word or timing in docs/demo.html and this fails, because the
    committed video now shows something the reel no longer claims. Line
    endings are normalized first (git stores LF, Windows checkouts are
    CRLF), so the stamp means the same bytes on every platform.
    """
    data = REEL.read_bytes().replace(b"\r\n", b"\n")
    stamp = "reel-" + hashlib.sha256(data).hexdigest()[:16]
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
    # The CI smoke job renders one frame per theme through this flag; if
    # it is renamed or removed, reel-smoke.yml starts failing blankly.
    assert "--frames" in source


# --- Light-theme coverage of the palette guard -------------------------
#
# Three pages ship dark-first HTML with a light look bolted on: the
# demo reel (both videos render from docs/demo.html), the console UI,
# and the static preview. The dark look is the default stylesheet; the
# light look exists only because theme-marked rules redefine the tokens
# and re-paint the surfaces whose colors were hardcoded for the dark
# stage. Miss one such surface and the light view opens on a black pill
# with dark text - exactly the v1.2.0 intro regression (`.badge` kept
# `#0d141d`) that shipped on demo-light.mp4. Both checks are static on
# purpose: the suite runs in CI without a browser, and a color audit is
# pure text work.

# A painted surface this dark reads as a black panel on the light
# stage. Every dark-palette surface in these pages sits far below it.
DARK_SURFACE_LUMINANCE = 0.15

SURFACE_PROPS = ("background", "background-color", "background-image")

# Page -> the selector text that marks its light-theme rules.
LIGHT_THEME_PAGES = {
    "docs/demo.html": 'data-reel-theme="light"',
    "market_intel/console.html": 'data-theme="light"',
    "docs/preview.html": 'data-theme="light"',
}


def page(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def palette_rules(relpath: str) -> tuple[list, list]:
    """(base rules, light-theme rules) for one guarded page.

    All <style> blocks are joined, comments stripped, split into rules;
    a rule belongs to the light palette when its selector carries the
    page's theme marker, to the base palette otherwise.
    """
    blocks = re.findall(r"<style>(.*?)</style>", page(relpath), re.DOTALL)
    assert blocks, f"{relpath} has no stylesheet"
    css = re.sub(r"/\*.*?\*/", "", "\n".join(blocks), flags=re.DOTALL)
    marker = LIGHT_THEME_PAGES[relpath]
    rules = css_rules(css)
    light = [rule for rule in rules if marker in rule[0]]
    assert light, f"{relpath} declares no {marker} rules"
    return [rule for rule in rules if marker not in rule[0]], light


def selector_tokens(header: str) -> frozenset[str]:
    """Selector words, ignoring combinator syntax so 'html, body' in the
    base rules matches 'html[data-reel-theme="light"] body' in the light
    rules (both reduce to the tokens html and body)."""
    return frozenset(t for t in re.split(r"[^a-zA-Z0-9_-]+", header) if t)


def srgb_luminance(hex_color: str) -> float:
    """WCAG relative luminance of a #rgb or #rrggbb color."""
    digits = hex_color[1:]
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    r, g, b = (int(digits[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def css_rules(css: str) -> list[tuple[str, str]]:
    """Every {selector: declarations} pair, including @keyframe inner rules."""
    return [
        (header.strip(), body)
        for header, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)
    ]


def dark_hexes(value: str) -> list[str]:
    hexes = re.findall(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b", value)
    return [h for h in hexes if srgb_luminance(h) < DARK_SURFACE_LUMINANCE]


def paints_a_surface(declaration: str) -> bool:
    if ":" not in declaration:
        return False
    prop = declaration.split(":", 1)[0].strip()
    return prop in SURFACE_PROPS


PALETTE_PAGES = {
    "docs/demo.html": "demo-reel",
    "docs/preview.html": "static-preview",
    "market_intel/console.html": "console",
}


@pytest.mark.parametrize("relpath", sorted(LIGHT_THEME_PAGES),
                         ids=[PALETTE_PAGES[p] for p in sorted(LIGHT_THEME_PAGES)])
def test_light_palette_repaints_every_dark_surface(relpath: str):
    """A surface painted dark for the default look must get a light repaint.

    Scope is backgrounds only (panels, terminals, pills, the stage itself):
    those are what black out and become unreadable. Accent strokes - the
    bar-fill gradient stops (#2f6f4f / #2b4a7a) and buttons - are
    deliberately two-tone (or token-painted) and are exempted by name.
    """
    base, light_rules = palette_rules(relpath)

    offenders = []
    for header, body in base:
        if "fill" in selector_tokens(header):
            continue  # data-viz fill bars: two-stop gradients, ok on both themes
        for declaration in body.split(";"):
            if ":" not in declaration:
                continue
            prop, value = (part.strip() for part in declaration.split(":", 1))
            if prop not in SURFACE_PROPS or not dark_hexes(value):
                continue
            repainted = any(
                selector_tokens(header) <= selector_tokens(light_header)
                and any(paints_a_surface(d) for d in light_body.split(";"))
                for light_header, light_body in light_rules
            )
            if not repainted:
                offenders.append(f"{header} -> {prop}: {value}")

    assert offenders == [], (
        f"{relpath}: dark surfaces with no repaint under the light theme - "
        "they would show as unreadable black panels: " + "; ".join(offenders)
    )


@pytest.mark.parametrize("relpath", sorted(LIGHT_THEME_PAGES),
                         ids=[PALETTE_PAGES[p] for p in sorted(LIGHT_THEME_PAGES)])
def test_no_dark_surface_hidden_in_inline_styles(relpath: str):
    """Inline style attributes beat even the theme's light rules, so a dark
    paint may only appear there on elements where both themes want it: the
    reel's bar fills. Anything else hides from the palette check entirely.
    JS-written styles (element.style.x = ...) are out of static reach and
    left to the browser smoke check."""
    offenders = []
    for tag in re.finditer(r"<[a-z][^>]*>", page(relpath)):
        style = re.search(r'style="([^"]*)"', tag.group(0))
        if not style or not dark_hexes(style.group(1)):
            continue
        cls = re.search(r'class="([^"]*)"', tag.group(0))
        if cls and re.search(r"\bfill\b", cls.group(1)):
            continue
        offenders.append(style.group(1))
    assert offenders == [], (
        f"{relpath}: inline dark paints survive both themes - exempt as a "
        f"fill or drop the hardcode: {offenders}"
    )
