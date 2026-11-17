"""
Extract the average-waveform panel for each selected cluster from the
combinato overview plots and combine them into one PNG per
patient / visit / sensor (negative polarity only).

Input:
    cluster_selections_long.csv  (patient, visit, sensor, polarity, cluster)
    clustering_plots/sorted_unknown_Patient{p}_Visit{v}_sensor{s}_neg_sort_neg_ada.png

Output:
    selected_cluster_plots/Patient{p}_Visit{v}_sensor{s}_neg_selected.png

The overview plots are 2100 px wide and 300 px tall per row. Each cluster's
average panel sits in the first column with its label ("<cluster> <n> <type>")
printed underneath. Sub-cluster panels may wrap onto extra rows, so rows are not
mapped to clusters by position: the cluster number is read from the label
itself by matching rendered DejaVu Sans glyphs (the matplotlib default font).
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
from PIL import Image, ImageDraw, ImageFont

CSV = "cluster_selections_long.csv"
PLOT_DIR = "clustering_plots"
OUT_DIR = "selected_cluster_plots"

PANEL_X1 = 308          # right edge of first-column panel (black spine)
ROW_PX = 300            # figure height per row
TEXT_THRESH = 90        # max RGB value counted as label text
CHARS = "-0123456789"


def row_bounds(height):
    n = height // ROW_PX
    step = height * 0.99 / n
    off = height * 0.005
    return [(int(round(off + r * step)), int(round(off + (r + 1) * step))) for r in range(n)]


def _norm(mask, size=(16, 24)):
    ys, xs = np.where(mask)
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(size, Image.BILINEAR)) / 255.0


def build_templates(digit_height):
    font_path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
    # find a font size whose "0" matches the measured digit height
    for size in range(10, 80):
        font = ImageFont.truetype(font_path, size)
        if font.getbbox("0")[3] - font.getbbox("0")[1] >= digit_height:
            break
    templates = {}
    for ch in CHARS:
        img = Image.new("L", (size * 2, size * 2), 0)
        ImageDraw.Draw(img).text((size // 2, size // 2), ch, fill=255, font=font)
        m = np.asarray(img) > 127
        ys, xs = np.where(m)
        templates[ch] = (m[ys.min():ys.max() + 1, xs.min():xs.max() + 1], _norm(m))
    return templates


_TEMPLATE_CACHE = {}


def read_label(row_img):
    """Return the cluster number printed under the first-column panel, or None."""
    dark = row_img.max(-1) < TEXT_THRESH
    # left spine of the panel: the most consistently dark column near x=10
    spine_x = 5 + int(np.argmax(dark[:, 5:20].mean(0)))
    col = dark[:, spine_x]
    top = int(np.argmax(col))
    if not col[top]:
        return None
    # walk down the spine to find the bottom of the axes, tolerating short
    # gaps where coloured traces are drawn over it
    bottom, gap = top, 0
    for y in range(top + 1, len(col)):
        if col[y]:
            bottom, gap = y, 0
        else:
            gap += 1
            if gap > 6:
                break
    # label rows have the axes ending well above the row bottom
    if row_img.shape[0] - bottom < 25:
        return None
    txt = dark[bottom + 2:, 0:PANEL_X1 - 4]
    # drop long horizontal border lines (top spine of the next row)
    txt[txt.mean(1) > 0.5] = False
    # find the label line using only the left edge, where the cluster number
    # starts (time annotations like "19 min" can sit lower right of the axes)
    has = txt[:, :45].any(1)
    if has.sum() == 0 or txt.sum() < 80:
        return None
    # take the first contiguous run of text rows (the label line)
    start = int(np.argmax(has))
    end = start
    while end + 1 < len(has) and has[end + 1]:
        end += 1
    txt = txt[start:end + 1]

    # split columns into glyphs, stop at the first word gap
    cols = txt.any(0)
    glyphs, start, gap = [], None, 0
    for x, on in enumerate(cols):
        if on:
            if start is None:
                if glyphs and gap > 8:   # word space
                    break
                start = x
            gap = 0
        else:
            if start is not None:
                glyphs.append((start, x))
                start = None
            gap += 1
    if start is not None:
        glyphs.append((start, len(cols)))
    if not glyphs:
        return None

    # digit height from the first non-minus glyph
    heights = []
    for a, b in glyphs:
        ys = np.where(txt[:, a:b].any(1))[0]
        heights.append(ys.max() - ys.min() + 1)
    digit_h = max(heights)
    if digit_h not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[digit_h] = build_templates(digit_h)
    templates = _TEMPLATE_CACHE[digit_h]

    out = ""
    for (a, b), h in zip(glyphs, heights):
        g = txt[:, a:b]
        if h < digit_h * 0.4:  # flat glyph -> minus sign
            out += "-"
            continue
        gn = _norm(g)
        best = min((np.abs(gn - templates[c][1]).mean(), c) for c in CHARS if c != "-")
        out += best[1]
    try:
        return int(out)
    except ValueError:
        return None


def extract_panels(png_path):
    """Return {cluster_label: panel image} for every average panel in the plot."""
    im = Image.open(png_path).convert("RGB")
    a = np.asarray(im)
    panels = {}
    for y0, y1 in row_bounds(a.shape[0]):
        label = read_label(a[y0:y1, :PANEL_X1])
        if label is not None:
            panels[label] = im.crop((0, y0, PANEL_X1 + 2, y1))
    return panels


def stack(images):
    w = max(i.width for i in images)
    h = sum(i.height for i in images)
    out = Image.new("RGB", (w, h), "white")
    y = 0
    for i in images:
        out.paste(i, (0, y))
        y += i.height
    return out


def main(debug=False):
    df = pd.read_csv(CSV)
    df = df[df["polarity"] == "neg"]
    os.makedirs(OUT_DIR, exist_ok=True)
    problems = []
    n_written = 0
    for (p, v, s), grp in df.groupby(["patient", "visit", "sensor"], sort=False):
        png = os.path.join(PLOT_DIR, f"sorted_unknown_Patient{p}_Visit{v}_sensor{s}_neg_sort_neg_ada.png")
        if not os.path.exists(png):
            problems.append(f"missing plot: {png}")
            continue
        panels = extract_panels(png)
        if debug:
            print(os.path.basename(png), sorted(panels))
        wanted = list(grp["cluster"])
        missing = [c for c in wanted if c not in panels]
        if missing:
            problems.append(f"P{p} V{v} S{s}: clusters {missing} not found (found {sorted(panels)})")
        imgs = [panels[c] for c in wanted if c in panels]
        if imgs:
            stack(imgs).save(os.path.join(OUT_DIR, f"Patient{p}_Visit{v}_sensor{s}_neg_selected.png"))
            n_written += 1
    print(f"wrote {n_written} images to {OUT_DIR}/")
    for msg in problems:
        print("WARNING:", msg)


if __name__ == "__main__":
    main(debug="--debug" in sys.argv)
