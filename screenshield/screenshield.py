#!/usr/bin/env python3
"""screenshield — derive screen-protector overlays from a device photo.

Stage 1: body detection and outline derivation (--debug only).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Calibrated against the anti-aliasing halo measured on ref_1x.jpg (~14 levels
# wide, background-to-edge), with ~2x margin. This is the fallback used when
# auto_tol can't find a confident background/subject gap — it is NOT tuned to
# any single device's contrast, so it must stay low enough to still catch
# light-coloured devices (white phone, silver watch, light-grey body) against
# a white background.
FIXED_DEFAULT_TOL = 30


def sample_background_color(img, corner_frac=0.03):
    h, w = img.shape[:2]
    cy = max(1, int(h * corner_frac))
    cx = max(1, int(w * corner_frac))
    corners = [
        img[0:cy, 0:cx],
        img[0:cy, w - cx:w],
        img[h - cy:h, 0:cx],
        img[h - cy:h, w - cx:w],
    ]
    samples = np.concatenate([c.reshape(-1, c.shape[-1]) for c in corners], axis=0)
    return np.median(samples, axis=0)


def border_deviation(img, bg_color, border_frac=0.03):
    """Max-channel deviation from bg_color, sampled over an outer border band
    (wider and more representative than the tiny corner patches used to
    estimate bg_color itself)."""
    h, w = img.shape[:2]
    bh, bw = max(1, int(h * border_frac)), max(1, int(w * border_frac))
    border = np.zeros((h, w), dtype=bool)
    border[:bh, :] = True
    border[-bh:, :] = True
    border[:, :bw] = True
    border[:, -bw:] = True
    diff = np.abs(img.astype(np.int16) - bg_color.astype(np.int16)).max(axis=-1)
    return diff[border]


def auto_tol(img, bg_color, fixed_default=FIXED_DEFAULT_TOL, min_tol=10, max_tol=150,
             valley_ratio_max=0.05, border_margin_factor=1.5):
    """Pick a threshold from the actual gap in the background/subject
    deviation histogram (Otsu), instead of a fixed constant.

    Otsu alone would happily split pure sensor/JPEG noise if nothing else in
    the frame separates cleanly, so we require the histogram to actually show
    a valley at that point (few pixels sit near the cut vs. the background
    peak) before trusting it. Falls back to fixed_default otherwise.

    Returns (tol, used_auto: bool, border_p99: float) — border_p99 is only
    for diagnostics/logging.
    """
    diff = np.abs(img.astype(np.int16) - bg_color.astype(np.int16))
    dist = diff.max(axis=-1)
    dist_u8 = np.clip(dist, 0, 255).astype(np.uint8)

    otsu_val, _ = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hist = cv2.calcHist([dist_u8], [0], None, [256], [0, 256]).flatten()
    bg_peak = hist[:10].max()
    valley_idx = int(round(otsu_val))
    valley_count = hist[max(0, valley_idx - 2):valley_idx + 3].sum()

    border_p99 = float(np.percentile(border_deviation(img, bg_color), 99.5))

    confident = otsu_val > 0 and bg_peak > 0 and (valley_count / bg_peak) < valley_ratio_max
    if not confident:
        return fixed_default, False, border_p99

    tol = float(np.clip(max(otsu_val, border_p99 * border_margin_factor), min_tol, max_tol))
    return tol, True, border_p99


def device_mask(img, bg_color, tol, close_frac=0.008):
    diff = np.abs(img.astype(np.int16) - bg_color.astype(np.int16))
    dist = diff.max(axis=-1)
    mask = (dist > tol).astype(np.uint8) * 255

    h, w = img.shape[:2]
    k = max(3, int(round(close_frac * max(h, w))) | 1)  # force odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def largest_external_contour(mask, chain=cv2.CHAIN_APPROX_SIMPLE):
    # CHAIN_APPROX_SIMPLE collapses straight runs to their endpoints — fine
    # for bbox/area, but useless for line-fitting (an axis-aligned edge
    # degenerates to exactly 2 points, both corner-adjacent). Callers that
    # need real point density along the boundary must pass CHAIN_APPROX_NONE.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, chain)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def is_near_circular(contour, thresh=0.92):
    area = cv2.contourArea(contour)
    if area <= 0:
        return False
    (_, _), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * radius * radius
    if circle_area <= 0:
        return False
    return (area / circle_area) > thresh


def _fit_line_to_points(points):
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return np.array([x0, y0], dtype=np.float64), np.array([vx, vy], dtype=np.float64)


def _line_intersection(p1, d1, p2, d2):
    a = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    t, _ = np.linalg.solve(a, p2 - p1)
    return p1 + t * d1


def _vertex_indices_in_contour(contour, approx):
    contour_pts = contour.reshape(-1, 2)
    approx_pts = approx.reshape(-1, 2)
    indices = []
    for pt in approx_pts:
        d = np.sum((contour_pts - pt) ** 2, axis=1)
        indices.append(int(np.argmin(d)))
    return indices


def snap_polygon_to_lines(contour, approx, corner_margin_px=0.0, min_fit_pts=6, min_core_pts=6):
    """Refine a Douglas-Peucker polygon by least-squares-fitting a line to the
    raw contour points along each edge, then recomputing each vertex as the
    intersection of its two adjacent fitted lines. Cleans up raster
    jaggedness and erosion-rounded corners that approxPolyDP alone leaves in
    slightly the wrong place.

    corner_margin_px trims points within that Euclidean distance of either
    endpoint before fitting — a FIXED fraction of segment *point count* is
    not enough: corner rounding is a function of the erosion kernel's
    radius, not of how long the edge happens to be. A short chamfer between
    two long sides can be almost entirely inside the fillet radius; the
    caller should pass something like 1.5x the erosion radius. Edges with
    too few points left after trimming keep their original approxPolyDP
    vertex rather than fit to a handful of corner-contaminated points."""
    contour_pts = contour.reshape(-1, 2).astype(np.float64)
    idxs = _vertex_indices_in_contour(contour, approx)
    m = len(idxs)
    if m < 3:
        return approx

    lines = []
    for i in range(m):
        i0, i1 = idxs[i], idxs[(i + 1) % m]
        seg = contour_pts[i0:i1 + 1] if i1 > i0 else np.concatenate(
            [contour_pts[i0:], contour_pts[:i1 + 1]])
        if len(seg) < min_fit_pts:
            lines.append(None)
            continue
        if corner_margin_px > 0 and len(seg) > 2:
            step_len = np.linalg.norm(np.diff(seg, axis=0), axis=1)
            dist_from_start = np.concatenate([[0.0], np.cumsum(step_len)])
            dist_from_end = dist_from_start[-1] - dist_from_start
            core = seg[(dist_from_start >= corner_margin_px) & (dist_from_end >= corner_margin_px)]
        else:
            core = seg
        if len(core) < min_core_pts:
            lines.append(None)
            continue
        lines.append(_fit_line_to_points(core))

    approx_pts = approx.reshape(-1, 2).astype(np.float64)
    snapped = approx_pts.copy()
    for i in range(m):
        edge_prev, edge_curr = lines[i - 1], lines[i]
        if edge_prev is None or edge_curr is None:
            continue
        pt = _line_intersection(edge_prev[0], edge_prev[1], edge_curr[0], edge_curr[1])
        if pt is not None:
            snapped[i] = pt

    return snapped.reshape(-1, 1, 2).astype(np.float32)


def derive_outline(contour, mask_shape, inset_x_px, inset_y_px, epsilon_frac=0.004):
    """Fill contour, erode by (inset_x_px, inset_y_px), re-contour, simplify
    (or fit circle/ellipse). The erosion kernel is itself elliptical/anisotropic
    so inset_x and inset_y are honored independently — do not erode uniformly
    and patch afterwards, that can't reproduce a non-uniform true inset."""
    filled = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)

    kx = max(1, int(round(inset_x_px)) * 2 + 1)
    ky = max(1, int(round(inset_y_px)) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
    eroded = cv2.erode(filled, kernel, iterations=1)

    eroded_contour = largest_external_contour(eroded, chain=cv2.CHAIN_APPROX_NONE)
    if eroded_contour is None:
        return None, False

    if is_near_circular(eroded_contour):
        ellipse = cv2.fitEllipse(eroded_contour)
        return ellipse, True

    perimeter = cv2.arcLength(eroded_contour, True)
    epsilon = epsilon_frac * perimeter
    approx = cv2.approxPolyDP(eroded_contour, epsilon, True)
    corner_margin_px = 1.5 * max(inset_x_px, inset_y_px)
    approx = snap_polygon_to_lines(eroded_contour, approx, corner_margin_px=corner_margin_px)
    return approx, False


def ellipse_to_points(ellipse, n=72):
    (cx, cy), (major, minor), angle = ellipse
    a, b = major / 2.0, minor / 2.0
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = a * np.cos(t)
    ys = b * np.sin(t)
    theta = np.deg2rad(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = rot @ np.vstack([xs, ys])
    pts[0] += cx
    pts[1] += cy
    return pts.T.astype(np.float32).reshape(-1, 1, 2)


def sheet_offset_px(body_w, body_h, offset_pct):
    dx_pct, dy_pct = offset_pct
    return (dx_pct / 100.0) * body_w, (dy_pct / 100.0) * body_h


def outline_bounds(pts):
    pts = pts.reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def detect_body(img, tol, inset_x_pct, inset_y_pct, debug=False):
    """tol: fixed threshold to use, or None to run auto_tol.
    inset_x_pct: % of body WIDTH. inset_y_pct: % of body HEIGHT — these are
    independent because the true inset is not generally isotropic in px."""
    bg = sample_background_color(img)

    if tol is None:
        resolved_tol, used_auto, border_p99 = auto_tol(img, bg)
    else:
        resolved_tol, used_auto, border_p99 = float(tol), False, None

    mask = device_mask(img, bg, resolved_tol)
    body_contour = largest_external_contour(mask)
    if body_contour is None:
        return None

    x, y, w, h = cv2.boundingRect(body_contour)
    inset_x_px = (inset_x_pct / 100.0) * w
    inset_y_px = (inset_y_pct / 100.0) * h

    outline, is_circle = derive_outline(body_contour, mask.shape[:2], inset_x_px, inset_y_px)
    outline_pts = ellipse_to_points(outline) if is_circle else outline

    result = {
        "bg_color": bg,
        "mask": mask,
        "body_contour": body_contour,
        "body_bbox": (x, y, w, h),
        "inset_x_px": inset_x_px,
        "inset_y_px": inset_y_px,
        "outline": outline_pts,
        "is_circle": is_circle,
        "tol_used": resolved_tol,
        "tol_auto": used_auto,
        "border_p99": border_p99,
    }
    return result


def draw_debug(img, result):
    dbg = img.copy()
    cv2.drawContours(dbg, [result["body_contour"]], -1, (0, 0, 255), 2)  # red, BGR
    if result["outline"] is not None:
        cv2.drawContours(dbg, [result["outline"].astype(np.int32)], -1, (255, 0, 0), 2)  # blue
    x, y, w, h = result["body_bbox"]
    cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 255), 1)
    return dbg


# --- Stage 2: compositing, opacity model, canvas framing -------------------

# Measured directly from ref_1x.jpg, not the naive CMYK->RGB conversion of
# the .ai stroke (~#677A7B — far too dark/saturated for what a 1pt stroke
# anti-aliases down to at this scale). Two independent clean crossings over
# white background: a vertical edge landed the stroke's peak darkness in a
# single row (255->151), a horizontal edge split it evenly across two rows
# (255->186, 255->187). Summed "ink deficit" (255-pixel) matches almost
# exactly between the two (137 vs 137), confirming both are the same
# underlying anti-aliased stroke at different sub-pixel phase. Integrating
# that deficit over the spec's ~1.5px stroke width gives the true color:
# 255 - 137/1.5 ~= 164 (#A4A4A4) — lands centered in the spec's own #98-#B0
# estimate.
STROKE_COLOR_RGB = (164, 164, 164)
STROKE_WIDTH_AT_1500 = 1.5


def composite_alpha(effective_alphas):
    """Cumulative alpha after Porter-Duff 'over' compositing a stack of
    layers, given in paint order (index 0 = bottom/backmost/nearest-device,
    last = top/frontmost/furthest-offset). Pure function - no geometry, no
    color - so it can be unit-tested against the spec's reference numbers
    directly, and a compositing bug shows up immediately instead of two
    stages later as a vague "looks slightly off"."""
    a = 0.0
    for layer_a in effective_alphas:
        a = layer_a + a * (1.0 - layer_a)
    return a


def composite_layer_over(canvas_f, mask, color_rgb, alpha):
    """Alpha-blend color_rgb into canvas_f (float64 HxWx3), in place,
    wherever mask is truthy (bool array, or 0/1 coverage float array for
    anti-aliased edges). canvas_f is always treated as fully opaque."""
    if alpha <= 0:
        return
    a = mask.astype(np.float64) * alpha
    color = np.array(color_rgb, dtype=np.float64)
    canvas_f[:] = color * a[..., None] + canvas_f * (1.0 - a[..., None])


def translate_outline(outline, dx, dy):
    out = outline.reshape(-1, 2).astype(np.float64).copy()
    out[:, 0] += dx
    out[:, 1] += dy
    return out.reshape(-1, 1, 2)


def sheet_outlines(base_outline, body_w, body_h, offset_pct, step_pct, copies):
    """base_outline: the stage-1 derived outline, still in the input image's
    coordinate frame. Returns `copies` outlines, sheet 1 (nearest the
    device, smallest offset) first, each subsequent sheet stepped further
    down-right."""
    ox, oy = sheet_offset_px(body_w, body_h, offset_pct)
    sx, sy = sheet_offset_px(body_w, body_h, step_pct)
    return [translate_outline(base_outline, ox + i * sx, oy + i * sy) for i in range(copies)]


def compute_content_bbox(body_bbox, outlines):
    x, y, w, h = body_bbox
    minx, miny, maxx, maxy = float(x), float(y), float(x + w), float(y + h)
    for outline in outlines:
        l, t, r, b = outline_bounds(outline)
        minx, miny = min(minx, l), min(miny, t)
        maxx, maxy = max(maxx, r), max(maxy, b)
    return minx, miny, maxx, maxy


def compute_canvas_transform(content_bbox, canvas_size, margin_pct=4.0):
    """Uniform scale + translate that fits content_bbox into canvas_size with
    margin_pct of blank border on every side, content centered. Both outputs
    must reuse the SAME transform — the whole point of framing on the
    3-sheet extent is that the device lands identically in each file."""
    minx, miny, maxx, maxy = content_bbox
    content_w, content_h = maxx - minx, maxy - miny
    available = canvas_size * (1 - 2 * margin_pct / 100.0)
    scale = available / max(content_w, content_h)
    scaled_w, scaled_h = content_w * scale, content_h * scale
    tx = (canvas_size - scaled_w) / 2.0 - minx * scale
    ty = (canvas_size - scaled_h) / 2.0 - miny * scale
    return scale, tx, ty


def transform_outline(outline, scale, tx, ty):
    out = outline.reshape(-1, 2).astype(np.float64).copy()
    out[:, 0] = out[:, 0] * scale + tx
    out[:, 1] = out[:, 1] * scale + ty
    return out.reshape(-1, 1, 2)


def place_image_on_canvas(img, scale, tx, ty, canvas_size, bg_color=(255, 255, 255)):
    """Resize img by scale and paste it at (tx,ty) onto a canvas_size square
    canvas, cropping whatever falls outside."""
    h, w = img.shape[:2]
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    canvas = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.uint8)
    dst_x, dst_y = round(tx), round(ty)

    src_x0, src_y0 = max(0, -dst_x), max(0, -dst_y)
    dst_x0, dst_y0 = max(0, dst_x), max(0, dst_y)
    src_x1 = min(new_w, canvas_size - dst_x)
    src_y1 = min(new_h, canvas_size - dst_y)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = resized[src_y0:src_y1, src_x0:src_x1]
    return canvas


def render_sheet(canvas_f, outline_px, style_alpha, layer_alpha, stroke_width_px,
                  fill_color=(255, 255, 255), stroke_color=STROKE_COLOR_RGB):
    """Draw one sheet's fill then its stroke onto canvas_f (float64 HxWx3),
    each alpha-composited over whatever is already there (previous sheets +
    device photo underneath).

    Fill opacity = style_alpha * layer_alpha: Graphic Style 4's own fill
    renders at 50% (style_alpha) independent of the object's layer opacity,
    per the spec's opacity model — the two are kept as separate parameters
    because layer opacity varies per product line.

    Stroke opacity = layer_alpha only: nothing in the extracted Graphic
    Style 4 data reduces the stroke's own opacity, only the object/layer
    opacity does.
    """
    h, w = canvas_f.shape[:2]
    pts = outline_px.astype(np.int32)

    fill_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(fill_mask, [pts], 255)
    composite_layer_over(canvas_f, fill_mask > 0, fill_color, style_alpha * layer_alpha)

    # cv2.polylines with LINE_AA on a uint8 mask gives real 0-255 coverage
    # at the edge (not a hard binary blob) — normalize that to a [0,1]
    # coverage map and use it as a per-pixel alpha multiplier, so a
    # partially-covered edge pixel gets a partial blend instead of a jaggy.
    stroke_coverage = np.zeros((h, w), dtype=np.uint8)
    thickness = max(1, round(stroke_width_px))
    cv2.polylines(stroke_coverage, [pts], isClosed=True, color=255,
                  thickness=thickness, lineType=cv2.LINE_AA)
    composite_layer_over(canvas_f, stroke_coverage.astype(np.float64) / 255.0,
                          stroke_color, layer_alpha)


def resolve_layer_alphas(args, copies):
    if args.layer_alpha is not None:
        if len(args.layer_alpha) != copies:
            raise ValueError(
                f"--layer-alpha needs {copies} values (one per --copies), got {len(args.layer_alpha)}")
        return args.layer_alpha
    if copies == 3:
        return [0.5, 0.5, 0.8]  # nearest, middle, furthest/frontmost — spec default
    raise ValueError(f"--layer-alpha required when --copies != 3 (no default for {copies} sheets)")


def render_outputs(img, result, args):
    """Renders both outputs off the SAME canvas transform (framed on the
    3-sheet extent), per spec: the device must land identically in both
    files, not be framed independently. Returns (img_1x, img_3x) uint8."""
    body_w, body_h = result["body_bbox"][2], result["body_bbox"][3]
    base_outline = result["outline"]
    copies = args.copies

    outlines = sheet_outlines(base_outline, body_w, body_h, tuple(args.offset), tuple(args.step), copies)
    content_bbox = compute_content_bbox(result["body_bbox"], outlines)
    scale, tx, ty = compute_canvas_transform(content_bbox, args.size, margin_pct=4.0)
    stroke_width_px = STROKE_WIDTH_AT_1500 * (args.size / 1500.0)
    layer_alphas = resolve_layer_alphas(args, copies)

    base_canvas = place_image_on_canvas(img, scale, tx, ty, args.size)

    canvas_1x = base_canvas.astype(np.float64)
    sheet1_px = transform_outline(outlines[0], scale, tx, ty)
    render_sheet(canvas_1x, sheet1_px, args.style_alpha, layer_alpha=1.0,
                 stroke_width_px=stroke_width_px)

    canvas_3x = base_canvas.astype(np.float64)
    for i in range(copies):
        outline_px = transform_outline(outlines[i], scale, tx, ty)
        render_sheet(canvas_3x, outline_px, args.style_alpha, layer_alphas[i],
                     stroke_width_px=stroke_width_px)

    return (np.clip(canvas_1x, 0, 255).astype(np.uint8),
            np.clip(canvas_3x, 0, 255).astype(np.uint8))


def process_image(path: Path, args):
    img = cv2.imread(str(path))
    if img is None:
        print(f"skip {path}: could not read image", file=sys.stderr)
        return

    inset_x_pct, inset_y_pct = resolve_inset_pct(args)
    result = detect_body(img, tol=args.tol, inset_x_pct=inset_x_pct, inset_y_pct=inset_y_pct)
    if result is None:
        print(f"skip {path}: no device found", file=sys.stderr)
        return

    x, y, w, h = result["body_bbox"]
    shape = "circle/ellipse" if result["is_circle"] else f"{len(result['outline'])}-point polygon"
    tol_note = f"auto tol={result['tol_used']:.1f} (border p99={result['border_p99']:.1f})" if result["tol_auto"] \
        else f"fixed tol={result['tol_used']:.1f}"
    print(f"{path.name}: body bbox {w}x{h}px at ({x},{y}); "
          f"inset {result['inset_x_px']:.1f}x{result['inset_y_px']:.1f}px; "
          f"protector outline: {shape}; {tol_note}")

    if args.debug:
        args.outdir.mkdir(parents=True, exist_ok=True)
        dbg = draw_debug(img, result)
        out_path = args.outdir / f"{path.stem}_debug.png"
        cv2.imwrite(str(out_path), dbg)
        print(f"  wrote {out_path}")

    img_1x, img_3x = render_outputs(img, result, args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    path_1x = args.outdir / f"{path.stem}_1x.png"
    path_3x = args.outdir / f"{path.stem}_3x.png"
    cv2.imwrite(str(path_1x), img_1x)
    cv2.imwrite(str(path_3x), img_3x)
    print(f"  wrote {path_1x}")
    print(f"  wrote {path_3x}")


def resolve_inset_pct(args):
    """--inset is a convenience that sets both --inset-x and --inset-y to the
    same percentage, for devices whose inset genuinely is uniform. Given
    explicitly it overrides --inset-x/--inset-y."""
    if args.inset is not None:
        return args.inset, args.inset
    return args.inset_x, args.inset_y


def collect_inputs(path: Path):
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [path]


def build_parser():
    p = argparse.ArgumentParser(prog="screenshield")
    p.add_argument("input", type=Path, help="image file or folder")
    p.add_argument("--outdir", type=Path, default=Path("out"))
    p.add_argument("--target", choices=["body", "screen"], default="body")
    p.add_argument("--bleed", type=float, default=0.0)
    p.add_argument("--profile", type=Path, default=None)
    p.add_argument("--notch", nargs=4, type=float, action="append", default=None)
    p.add_argument("--pick", action="store_true")
    p.add_argument("--notches", choices=["profile", "auto", "none"], default="profile")
    p.add_argument("--size", type=int, default=1500)
    p.add_argument("--copies", type=int, default=3)
    p.add_argument("--inset", type=float, default=None,
                    help="convenience: sets both --inset-x and --inset-y to this "
                         "percentage, for devices with a uniform inset")
    p.add_argument("--inset-x", type=float, default=0.608, help="%% of body width")
    p.add_argument("--inset-y", type=float, default=1.441, help="%% of body height")
    p.add_argument("--offset", nargs=2, type=float, default=[7.901, 14.697])
    p.add_argument("--step", nargs=2, type=float, default=[2.75, 8.2])
    p.add_argument("--style-alpha", type=float, default=0.50)
    p.add_argument("--layer-alpha", nargs="+", type=float, default=None)
    p.add_argument("--tol", type=float, default=None,
                    help="background detection tolerance; default is auto "
                         "(histogram gap analysis), falling back to "
                         f"{FIXED_DEFAULT_TOL} if no confident gap is found")
    p.add_argument("--svg", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.target == "screen":
        print("error: --target screen is not implemented yet (build stage 4)", file=sys.stderr)
        return 1

    for path in collect_inputs(args.input):
        process_image(path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
