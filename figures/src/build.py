"""Render figures/src/<name>.html to figures/<name>.pdf with headless Chrome.

  python3 figures/src/build.py [name ...]      (default: every *.html here)

Each template reads its data from figures/data/<name>.json (inlined as
`DATA`) and draws with Observable Plot / D3 (vendored in vendor/). Fonts are
TrueType, inlined as base64, so the PDF embeds them as CID TrueType; the build
fails if a font comes out as Type 3 or a plot-only figure contains a raster
image. A PNG preview goes to figures/png/ (the README shows these). Needs
Chrome (set CHROME to its binary), poppler-utils and Pillow.
"""

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent
FIG = SRC.parent
FONTS = [
    ("Source Sans 3", "SourceSans3-Regular.ttf", 400, "normal"),
    ("Source Sans 3", "SourceSans3-It.ttf", 400, "italic"),
    ("Source Sans 3", "SourceSans3-Semibold.ttf", 600, "normal"),
    ("LM Roman", "lmroman10-regular.ttf", 400, "normal"),
    ("LM Roman", "lmroman10-italic.ttf", 400, "italic"),
]
CHROME = os.environ.get("CHROME", "google-chrome")


def head():
    faces = "".join(
        f"@font-face{{font-family:'{fam}';font-weight:{w};font-style:{s};"
        f"src:url(data:font/ttf;base64,{base64.b64encode((SRC / 'fonts' / f).read_bytes()).decode()})"
        " format('truetype')}"
        for fam, f, w, s in FONTS
    )
    js = "\n".join(
        (SRC / "vendor" / f).read_text() for f in ("d3.min.js", "plot.umd.min.js")
    )
    return f"<style>{faces}\n{(SRC / 'style.css').read_text()}</style><script>{js}</script>"


def images(name, height=1000):
    """Renders a template may embed, as data URIs, downscaled to `height` px
    (about 300 dpi at the size they are printed)."""
    import io

    from PIL import Image

    out = {}
    for p in (
        sorted((FIG / "renders").glob("*.png")) if (FIG / "renders").exists() else []
    ):
        im = Image.open(p)
        if (
            p.stem == "hero"
        ):  # crop to the rider and its shadow; hop frames keep a shared canvas
            box = im.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
            pad = round(0.03 * im.height)
            im = im.crop(
                (
                    max(box[0] - pad, 0),
                    max(box[1] - pad, 0),
                    min(box[2] + pad, im.width),
                    min(box[3] + pad, im.height),
                )
            )
        if im.height > height:
            im = im.resize(
                (round(im.width * height / im.height), height), Image.LANCZOS
            )
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True)
        out[p.stem] = (
            "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        )
    return out


def build(name):
    tpl = (SRC / f"{name}.html").read_text()
    data = FIG / "data" / f"{name}.json"
    html = tpl.replace("<!--HEAD-->", head()).replace(
        "/*DATA*/",
        "const DATA = " + (data.read_text() if data.exists() else "null") + ";",
    )
    uses_images = "/*IMAGES*/" in tpl
    if uses_images:
        html = html.replace(
            "/*IMAGES*/", "const IMAGES = " + json.dumps(images(name)) + ";"
        )
    w, h = re.search(r"data-size=\"(\d+)x(\d+)\"", tpl).groups()
    html = html.replace(
        "@page { margin: 0; }", f"@page {{ size: {w}px {h}px; margin: 0; }}"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(html)
    pdf = FIG / f"{name}.pdf"
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=15000",
            f"--print-to-pdf={pdf}",
            f"file://{f.name}",
        ],
        check=True,
        capture_output=True,
    )
    fonts = subprocess.run(
        ["pdffonts", str(pdf)], capture_output=True, text=True
    ).stdout
    if "Type 3" in fonts:
        sys.exit(f"{name}: Type 3 font in PDF\n{fonts}")
    pages = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    if not re.search(r"^Pages:\s+1$", pages, re.M):
        sys.exit(f"{name}: expected one page\n{pages}")
    imgs = subprocess.run(
        ["pdfimages", "-list", str(pdf)], capture_output=True, text=True
    ).stdout
    if not uses_images and len(imgs.splitlines()) > 2:
        sys.exit(f"{name}: raster image in a plot-only figure\n{imgs}")
    (FIG / "png").mkdir(exist_ok=True)
    subprocess.run(
        [
            "pdftoppm",
            "-r",
            "220",
            "-png",
            "-singlefile",
            str(pdf),
            str(FIG / "png" / name),
        ],
        check=True,
    )
    print(f"{name}: {pdf.relative_to(FIG.parent)} ({w}x{h} px)")


if __name__ == "__main__":
    for n in sys.argv[1:] or [p.stem for p in SRC.glob("*.html")]:
        build(n)
