#!/usr/bin/env python3
"""screenshield — derive screen-protector overlays from a device photo.

Stage 1: body detection and outline derivation (--debug only).
"""
import argparse
import base64
import json
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


def offset_filled_mask(filled, offset_x_px, offset_y_px):
    """Signed Minkowski offset via an elliptical kernel: positive shrinks
    inward (erosion, body inset), negative grows outward (dilation, --fit's
    'bleed past the active area' case). A single scalar sign drives both
    axes — mixed-sign per-axis isn't a case any caller needs: body inset is
    always >=0, --fit is one signed scalar applied to both axes."""
    kx = max(1, int(round(abs(offset_x_px))) * 2 + 1)
    ky = max(1, int(round(abs(offset_y_px))) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
    if offset_x_px >= 0 and offset_y_px >= 0:
        return cv2.erode(filled, kernel, iterations=1)
    if offset_x_px <= 0 and offset_y_px <= 0:
        return cv2.dilate(filled, kernel, iterations=1)
    raise ValueError(f"mixed-sign offset not supported: ({offset_x_px}, {offset_y_px})")


def derive_outline(contour, mask_shape, inset_x_px, inset_y_px, epsilon_frac=0.004):
    """Fill contour, offset by (inset_x_px, inset_y_px) — signed: positive
    erodes/shrinks inward (body inset, or a negative --fit contracting into
    a recess), negative dilates/grows outward (a positive --fit bleeding
    past a screen) — re-contour, simplify (or fit circle/ellipse). The
    kernel is itself elliptical/anisotropic so the x and y components are
    honored independently — do not offset uniformly and patch afterwards,
    that can't reproduce a non-uniform true offset."""
    filled = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)

    offset = offset_filled_mask(filled, inset_x_px, inset_y_px)

    offset_contour = largest_external_contour(offset, chain=cv2.CHAIN_APPROX_NONE)
    if offset_contour is None:
        return None, False

    if is_near_circular(offset_contour):
        ellipse = cv2.fitEllipse(offset_contour)
        return ellipse, True

    perimeter = cv2.arcLength(offset_contour, True)
    epsilon = epsilon_frac * perimeter
    approx = cv2.approxPolyDP(offset_contour, epsilon, True)
    corner_margin_px = 1.5 * max(abs(inset_x_px), abs(inset_y_px))
    approx = snap_polygon_to_lines(offset_contour, approx, corner_margin_px=corner_margin_px)
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


# --- Stage 4: --target screen / --target recess -----------------------------

def find_active_region_contour(img, body_contour, mode):
    """Largest coherent region inside the body that's visually distinct from
    the surrounding bezel:

    screen — bright OR saturated (a lit/colorful display), per spec: "the
    largest coherent bright or saturated region inside the body contour."

    recess — the opposite polarity: a sunken feature (camera lens, a
    control knob) typically reads as a locally DARK/shadowed area rather
    than bright, so this takes the low-brightness class instead. This mode
    isn't in the original spec and has no reference device to validate
    against — same honest caveat as --notches auto: best-effort, unproven
    on real hardware, always check the debug image.

    Both use Otsu on the relevant channel restricted to body pixels — the
    same "let the histogram find the gap" principle as auto_tol and
    --notches auto, rather than a fixed brightness constant.
    """
    body_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(body_mask, [body_contour], -1, 255, thickness=cv2.FILLED)
    body_idx = body_mask > 0
    if not body_idx.any():
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v, s = hsv[..., 2], hsv[..., 1]

    if mode == "screen":
        activity = np.maximum(v, s)
        thresh, _ = cv2.threshold(activity[body_idx].astype(np.uint8), 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        region_mask = ((activity > thresh) & body_idx).astype(np.uint8) * 255
    elif mode == "recess":
        thresh, _ = cv2.threshold(v[body_idx].astype(np.uint8), 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2's Otsu convention pairs with THRESH_BINARY's "> thresh" bright
        # class; the dark class is therefore "<= thresh", not "< thresh" —
        # with a hard two-level image (e.g. exactly 10 vs 30) Otsu can place
        # thresh exactly AT the dark class's own value, and "< thresh" then
        # selects nothing at all.
        region_mask = ((v <= thresh) & body_idx).astype(np.uint8) * 255
    else:
        raise ValueError(f"unknown target mode {mode!r}")

    k = max(3, int(round(0.008 * max(img.shape[:2]))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel)
    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, kernel)

    candidate = largest_external_contour(region_mask, chain=cv2.CHAIN_APPROX_NONE)
    if candidate is None:
        return None

    # A screen/recess is a sub-region of the body (spec's own framing for
    # screen mode). On a body with no real distinct feature — a uniform
    # color, or a degenerate Otsu split on near-constant input — the
    # "region" found can end up being almost the whole body, or a sliver of
    # noise. Neither is a usable target; treat both as not found rather
    # than silently returning a wrong answer.
    body_area = float(body_idx.sum())
    candidate_area = cv2.contourArea(candidate)
    area_ratio = candidate_area / body_area if body_area > 0 else 0.0
    if area_ratio > 0.9 or area_ratio < 0.005:
        return None

    return candidate


def detect_target(img, tol, target, inset_x_pct=0.0, inset_y_pct=0.0, fit_pct=0.0):
    """Body detection is always run first (needed for the bbox regardless of
    target — canvas framing and sheet offsets are body-relative). For
    target=='body' the returned outline is body_contour eroded by
    inset_x_pct/inset_y_pct, exactly as detect_body already did. For
    target in ('screen','recess') the outline instead comes from the
    detected active region, offset by the SIGNED --fit percentage: positive
    dilates outward (bleed past a screen onto the bezel), negative erodes
    inward (contract to fit inside a recess) — see offset_filled_mask.
    fit_pct is a single scalar applied against the active region's own
    width/height (not the body's), same spirit as inset_x/inset_y being
    percentages of the body's own width/height in body mode.
    """
    result = detect_body(img, tol, inset_x_pct=inset_x_pct, inset_y_pct=inset_y_pct)
    if result is None or target == "body":
        return result

    active_contour = find_active_region_contour(img, result["body_contour"], target)
    if active_contour is None:
        result["target_found"] = False
        return result

    ax, ay, aw, ah = cv2.boundingRect(active_contour)
    fit_x_px = (fit_pct / 100.0) * aw
    fit_y_px = (fit_pct / 100.0) * ah
    # derive_outline's offset is signed positive=erode/shrink; --fit's sign
    # convention is positive=expand/bleed outward, so negate going in.
    outline, is_circle = derive_outline(active_contour, result["mask"].shape[:2], -fit_x_px, -fit_y_px)
    outline_pts = ellipse_to_points(outline) if is_circle else outline

    result["target_found"] = True
    result["active_region_contour"] = active_contour
    result["active_region_bbox"] = (ax, ay, aw, ah)
    result["fit_x_px"] = fit_x_px
    result["fit_y_px"] = fit_y_px
    result["outline"] = outline_pts
    result["is_circle"] = is_circle
    return result


def draw_debug(img, result, notch_points=None):
    dbg = img.copy()
    cv2.drawContours(dbg, [result["body_contour"]], -1, (0, 0, 255), 2)  # red, BGR
    if result.get("active_region_contour") is not None:
        cv2.drawContours(dbg, [result["active_region_contour"]], -1, (255, 0, 255), 1)  # magenta: raw screen/recess region, pre-fit
    if result["outline"] is not None:
        cv2.drawContours(dbg, [result["outline"].astype(np.int32)], -1, (255, 0, 0), 2)  # blue: final protector outline (the "detected target")
    x, y, w, h = result["body_bbox"]
    cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 255), 1)
    for pts in (notch_points or []):
        cv2.drawContours(dbg, [pts.astype(np.int32)], -1, (0, 255, 0), 2)  # green
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


def stroke_coverage_map(shape_hw, polylines_pts, stroke_width_px, supersample=4):
    """Anti-aliased stroke coverage in [0,1] at a true sub-pixel width.

    cv2.polylines only accepts an integer thickness — rounding a 1.5px
    target up to 2px draws a stroke that's ~33% thicker than intended, a
    real, measurable difference (cross-checked against ref_1x.jpg: a clean
    axis-aligned stroke crossing showed a pixel at deficit 104 out of 255,
    which puts a hard lower bound of ~104 on the stroke's true peak-darkness
    value — that alone rules out the width implied by a 2px-equivalent
    stroke, since it would require every pixel's deficit to stay under ~68).
    So: supersample the polyline at an *integer* thickness scaled up by
    `supersample`, then area-downsample back to native resolution — the
    box-filter downsample reconstructs the fractional/sub-pixel width
    accurately instead of rounding it away.
    """
    h, w = shape_hw
    ss_thickness = max(1, round(stroke_width_px * supersample))
    big = np.zeros((h * supersample, w * supersample), dtype=np.uint8)
    for pts in polylines_pts:
        big_pts = (pts.reshape(-1, 1, 2).astype(np.float64) * supersample).astype(np.int32)
        cv2.polylines(big, [big_pts], isClosed=True, color=255,
                      thickness=ss_thickness, lineType=cv2.LINE_AA)
    coverage = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    return coverage.astype(np.float64) / 255.0


def render_sheet(canvas_f, outline_px, style_alpha, layer_alpha, stroke_width_px,
                  fill_color=(255, 255, 255), stroke_color=STROKE_COLOR_RGB,
                  notch_rects_px=None):
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

    notch_rects_px: optional list of (4,1,2) point arrays, already in this
    sheet's final pixel space (offset + canvas-transformed). Cut from the
    fill as even-odd holes, with the sheet's own stroke traced around each
    hole boundary too — matching a real compound path with holes.
    """
    h, w = canvas_f.shape[:2]
    pts = outline_px.astype(np.int32)
    notches = [n.astype(np.int32) for n in (notch_rects_px or [])]

    fill_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(fill_mask, [pts], 255)
    for notch_pts in notches:
        cv2.fillPoly(fill_mask, [notch_pts], 0)
    composite_layer_over(canvas_f, fill_mask > 0, fill_color, style_alpha * layer_alpha)

    stroke_coverage = stroke_coverage_map((h, w), [pts] + notches, stroke_width_px)
    composite_layer_over(canvas_f, stroke_coverage, stroke_color, layer_alpha)


# --- Stage 3: cutouts / notches ---------------------------------------------

def notch_pct_to_points(notch_pct, body_bbox):
    """Corners of a notch rect in the SAME (un-offset, device-relative)
    coordinate frame as the base derived outline — NOT sheet-relative. Each
    sheet's own offset gets applied to this afterward, same as the outline
    itself, so the notch stays in the same place relative to the sheet's own
    shape (these are copies of the same protector model, not registered to
    one physical device underneath)."""
    bx, by, bw, bh = body_bbox
    x = bx + notch_pct["x_pct"] / 100.0 * bw
    y = by + notch_pct["y_pct"] / 100.0 * bh
    w = notch_pct["w_pct"] / 100.0 * bw
    h = notch_pct["h_pct"] / 100.0 * bh
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                     dtype=np.float64).reshape(-1, 1, 2)


def notch_points_to_pct(points, body_bbox):
    """Inverse of notch_pct_to_points — round-trips exactly."""
    bx, by, bw, bh = body_bbox
    x0, y0, x1, y1 = outline_bounds(points)
    return {
        "x_pct": (x0 - bx) / bw * 100.0,
        "y_pct": (y0 - by) / bh * 100.0,
        "w_pct": (x1 - x0) / bw * 100.0,
        "h_pct": (y1 - y0) / bh * 100.0,
    }


TEMPLATE_MARGIN_FRAC = 0.15


def crop_notch_template(img, notch_points, margin_frac=TEMPLATE_MARGIN_FRAC):
    """Small image patch around a notch (with a margin for match context),
    to be cached in the profile and later relocated by template matching on
    a new shot of the same/similar model."""
    x0, y0, x1, y1 = outline_bounds(notch_points)
    w, h = x1 - x0, y1 - y0
    mx, my = w * margin_frac, h * margin_frac
    ih, iw = img.shape[:2]
    px0, py0 = max(0, int(round(x0 - mx))), max(0, int(round(y0 - my)))
    px1, py1 = min(iw, int(round(x1 + mx))), min(ih, int(round(y1 + my)))
    return img[py0:py1, px0:px1]


def encode_template(patch):
    if patch is None or patch.size == 0:
        return None
    ok, buf = cv2.imencode(".png", patch)
    return base64.b64encode(buf).decode("ascii") if ok else None


def decode_template(b64):
    if not b64:
        return None
    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def attach_notch_templates(notches_pct, img, body_bbox):
    """Adds a cached template patch to each notch dict, plus the body size
    it was captured against (needed to scale the template if a later photo
    has the device at a different pixel size). Mutates and returns the same
    list of dicts."""
    _, _, body_w, body_h = body_bbox
    for notch in notches_pct:
        points = notch_pct_to_points(notch, body_bbox)
        patch = crop_notch_template(img, points)
        b64 = encode_template(patch)
        if b64:
            notch["template_png_b64"] = b64
            notch["template_margin_frac"] = TEMPLATE_MARGIN_FRAC
            notch["captured_body_w"] = body_w
            notch["captured_body_h"] = body_h
    return notches_pct


def relocate_notches_by_template(img, body_bbox, cached_notches,
                                  search_margin_factor=1.0, min_match_score=0.6):
    """Relocate cached notches (with template patches from a prior --notch
    or --pick run) on a new photo, instead of searching the whole frame for
    unknown features. Each template is scaled by the ratio of the new
    body's size to the size it was captured against, then matched only
    within a small window around where the profile's percentage position
    predicts it should be — this is what makes it tractable where the
    brightness heuristic isn't: it's relocating a known thing, not
    discovering an unknown one.

    Returns (notches_pct, report). For a notch whose template doesn't match
    confidently, the profile's percentage-predicted position is used as-is
    (still a reasonable answer) and flagged in the report rather than
    dropped or handed off to the unrelated brightness scan.
    """
    bx, by, bw, bh = body_bbox
    notches_pct = []
    matched, low_confidence, no_template = 0, 0, 0
    scores = []

    for cached in cached_notches:
        b64 = cached.get("template_png_b64")
        template = decode_template(b64)
        cap_w = cached.get("captured_body_w")
        cap_h = cached.get("captured_body_h")
        margin_frac = cached.get("template_margin_frac", TEMPLATE_MARGIN_FRAC)

        if template is None or not cap_w or not cap_h:
            no_template += 1
            notches_pct.append({k: cached[k] for k in ("x_pct", "y_pct", "w_pct", "h_pct")})
            continue

        rx, ry = bw / cap_w, bh / cap_h
        th0, tw0 = template.shape[:2]
        new_tw, new_th = max(1, round(tw0 * rx)), max(1, round(th0 * ry))
        template_scaled = cv2.resize(template, (new_tw, new_th), interpolation=cv2.INTER_LINEAR)

        expected_points = notch_pct_to_points(cached, body_bbox)
        ex0, ey0, ex1, ey1 = outline_bounds(expected_points)
        # expected rect is the un-padded notch; the template includes the
        # margin, so the search window needs the same margin plus slack.
        ew, eh = ex1 - ex0, ey1 - ey0
        pad_x, pad_y = ew * margin_frac, eh * margin_frac
        slack_x, slack_y = new_tw * search_margin_factor, new_th * search_margin_factor

        ih, iw = img.shape[:2]
        sx0 = max(0, int(round(ex0 - pad_x - slack_x)))
        sy0 = max(0, int(round(ey0 - pad_y - slack_y)))
        sx1 = min(iw, int(round(ex1 + pad_x + slack_x)))
        sy1 = min(ih, int(round(ey1 + pad_y + slack_y)))
        search_region = img[sy0:sy1, sx0:sx1]

        if search_region.shape[0] < new_th or search_region.shape[1] < new_tw:
            low_confidence += 1
            scores.append(0.0)
            notches_pct.append({k: cached[k] for k in ("x_pct", "y_pct", "w_pct", "h_pct")})
            continue

        result_map = cv2.matchTemplate(search_region, template_scaled, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result_map)
        scores.append(float(max_val))

        if max_val < min_match_score:
            low_confidence += 1
            notches_pct.append({k: cached[k] for k in ("x_pct", "y_pct", "w_pct", "h_pct")})
            continue

        # matched top-left of the (padded) template, in image coords -> the
        # notch itself sits inset by the padding within that template.
        notch_w_in_template = new_tw / (1 + 2 * margin_frac)
        notch_h_in_template = new_th / (1 + 2 * margin_frac)
        inset_x = (new_tw - notch_w_in_template) / 2.0
        inset_y = (new_th - notch_h_in_template) / 2.0
        nx0 = sx0 + max_loc[0] + inset_x
        ny0 = sy0 + max_loc[1] + inset_y
        points = np.array([[nx0, ny0], [nx0 + notch_w_in_template, ny0],
                            [nx0 + notch_w_in_template, ny0 + notch_h_in_template],
                            [nx0, ny0 + notch_h_in_template]],
                           dtype=np.float64).reshape(-1, 1, 2)
        matched += 1
        notches_pct.append(notch_points_to_pct(points, body_bbox))

    report = {
        "matched": matched,
        "low_confidence": low_confidence,
        "no_template": no_template,
        "scores": scores,
    }
    return notches_pct, report


def load_profile(path):
    if path is None or not Path(path).exists():
        return {"notches": []}
    with open(path) as f:
        return json.load(f)


def save_profile(path, profile):
    path = Path(path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)


def compute_display_scale(img_shape, max_display=900):
    """Scale factor to fit an image within a max_display x max_display
    window for interactive picking — most device photos (1500px+) don't fit
    on screen at native size."""
    h, w = img_shape[:2]
    return min(1.0, max_display / max(h, w))


def rois_to_notches_pct(rois, display_scale, body_bbox):
    """Convert cv2.selectROIs output — (x,y,w,h) tuples in the SCALED
    display window's pixel space — to device-relative percentages.

    Kept separate from the actual GUI call (pick_notches) specifically so
    it's unit-testable: feed it synthetic ROI tuples and a known
    display_scale, no window required. The display scale must be divided
    back out here — selectROIs has no idea the image it saw was shrunk to
    fit the screen, it just reports pixel coordinates in that shrunk image.
    """
    notches = []
    for (x, y, w, h) in rois:
        if w <= 0 or h <= 0:
            continue
        fx, fy = x / display_scale, y / display_scale
        fw, fh = w / display_scale, h / display_scale
        points = np.array([[fx, fy], [fx + fw, fy], [fx + fw, fy + fh], [fx, fy + fh]],
                           dtype=np.float64).reshape(-1, 1, 2)
        notches.append(notch_points_to_pct(points, body_bbox))
    return notches


def pick_notches(img, body_bbox, profile_path=None, max_display=900):
    """Interactive: opens a cv2.selectROIs window (image scaled to fit the
    screen), lets the user drag a box over each cutout, and writes the
    result into --profile using the exact same schema --notch produces —
    so --notch and --pick are interchangeable routes to the same profile.
    Can't be exercised headlessly; rois_to_notches_pct above is where the
    actual conversion math lives and is unit-tested directly."""
    display_scale = compute_display_scale(img.shape, max_display)
    h, w = img.shape[:2]
    disp = cv2.resize(img, (max(1, round(w * display_scale)), max(1, round(h * display_scale))))
    rois = cv2.selectROIs("screenshield: drag a box over each cutout, ENTER per box, ESC when done", disp)
    cv2.destroyAllWindows()

    notches = rois_to_notches_pct(rois, display_scale, body_bbox)
    attach_notch_templates(notches, img, body_bbox)
    if profile_path:
        profile = load_profile(profile_path)
        profile["notches"] = notches
        save_profile(profile_path, profile)
        print(f"  wrote {len(notches)} notch(es) to {profile_path}")
    return notches


def detect_notches_auto(img, body_contour, body_bbox, min_area_pct=0.05, max_area_pct=5.0,
                         collar_frac=0.02):
    """Best-effort cutout detection: features inside the body that aren't
    bezel-dark. NOT reliable — a naive version of this found 5 candidates on
    the reference device where there are 2, because on-screen UI elements
    (icons, status bar contents) look like bezel features once the screen
    itself is removed. Always verify against the debug image (cutouts drawn
    in green) before trusting an auto-detected batch.

    Method: Otsu-threshold brightness within the body to separate bezel-dark
    from everything else (screen + any physical buttons), remove the
    largest such region (the display) plus a dilated collar around it so
    its own anti-aliased edge doesn't get picked up as extra candidates,
    then filter what's left by area.

    Returns (notches_pct, report). report always describes what happened —
    found/kept/rejected counts and why — so confidence in a given run can
    be judged from the printed summary without opening the debug image.
    """
    bx, by, bw, bh = body_bbox
    body_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(body_mask, [body_contour], -1, 255, thickness=cv2.FILLED)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    body_pixels = gray[body_mask > 0]
    if body_pixels.size == 0:
        return [], {"found": 0, "kept": 0, "rejected_area": 0, "screen_removed": False}

    dark_thresh, _ = cv2.threshold(body_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    non_bezel = ((gray > dark_thresh) & (body_mask > 0)).astype(np.uint8) * 255

    contours, _ = cv2.findContours(non_bezel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], {"found": 0, "kept": 0, "rejected_area": 0, "screen_removed": False}

    # largest non-bezel region = the display. Remove it + a collar.
    screen_contour = max(contours, key=cv2.contourArea)
    screen_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(screen_mask, [screen_contour], -1, 255, thickness=cv2.FILLED)
    collar_px = max(1, int(round(collar_frac * max(bw, bh))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (collar_px * 2 + 1, collar_px * 2 + 1))
    screen_mask = cv2.dilate(screen_mask, kernel)

    candidates_mask = non_bezel.copy()
    candidates_mask[screen_mask > 0] = 0

    cand_contours, _ = cv2.findContours(candidates_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    body_area = float(bw) * bh
    min_area = min_area_pct / 100.0 * body_area
    max_area = max_area_pct / 100.0 * body_area

    kept_contours = []
    rejected_area = 0
    for c in cand_contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            rejected_area += 1
            continue
        kept_contours.append(c)

    notches_pct = []
    for c in kept_contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        points = np.array([[cx, cy], [cx + cw, cy], [cx + cw, cy + ch], [cx, cy + ch]],
                           dtype=np.float64).reshape(-1, 1, 2)
        notches_pct.append(notch_points_to_pct(points, body_bbox))

    report = {
        "found": len(cand_contours),
        "kept": len(kept_contours),
        "rejected_area": rejected_area,
        "screen_removed": True,
    }
    return notches_pct, report


def resolve_notches(args, img=None, result=None):
    """Returns a list of notch dicts {x_pct,y_pct,w_pct,h_pct}, device-relative
    (percentages of the body bounding box). Storing relative percentages
    rather than pixels is what lets a profile survive any resolution or crop
    (spec). CLI --notch values take precedence and get written back to
    --profile if one is given, so the next shot of the same model is
    zero-input. img/result are only needed for --pick and --notches auto."""
    if args.notches == "none":
        return []

    if getattr(args, "pick", False):
        return pick_notches(img, result["body_bbox"], args.profile)

    if args.notch:
        notches = [{"x_pct": x, "y_pct": y, "w_pct": w, "h_pct": h} for x, y, w, h in args.notch]
        if img is not None and result is not None:
            attach_notch_templates(notches, img, result["body_bbox"])
        if args.profile:
            profile = load_profile(args.profile)
            profile["notches"] = notches
            save_profile(args.profile, profile)
            print(f"  wrote {len(notches)} notch(es) to {args.profile}")
        return notches

    if args.notches == "profile":
        if args.profile:
            return load_profile(args.profile).get("notches", [])
        return []

    if args.notches == "auto":
        cached = load_profile(args.profile).get("notches", []) if args.profile else []
        cached_with_templates = [n for n in cached if n.get("template_png_b64")]

        if cached_with_templates:
            notches, report = relocate_notches_by_template(img, result["body_bbox"], cached_with_templates)
            scores_str = ", ".join(f"{s:.2f}" for s in report["scores"])
            print(f"  auto (template relocate): {report['matched']} matched confidently, "
                  f"{report['low_confidence']} low-confidence (kept profile position), "
                  f"{report['no_template']} had no cached template — scores: [{scores_str}]")
            if report["low_confidence"]:
                print("  WARNING: some cached cutouts did not relocate confidently — "
                      "check the debug image before trusting this on a batch.")
            return notches

        notches, report = detect_notches_auto(img, result["body_contour"], result["body_bbox"])
        print("  WARNING: --notches auto is best-effort and known to produce false "
              "positives from on-screen UI — check the debug image (green boxes) "
              "before trusting this on a batch. (No cached notch templates found in "
              "--profile to relocate instead — define cutouts once via --notch or "
              "--pick and this will use template relocation on the next shot.)")
        print(f"  auto-detect (brightness fallback): {report['found']} raw candidate(s) "
              f"after removing the screen region, {report['rejected_area']} rejected by "
              f"area filter (0.05%-5% of body area), {report['kept']} kept")
        return notches

    return []


def resolve_layer_alphas(args, copies):
    if args.layer_alpha is not None:
        if len(args.layer_alpha) != copies:
            raise ValueError(
                f"--layer-alpha needs {copies} values (one per --copies), got {len(args.layer_alpha)}")
        return args.layer_alpha
    if copies == 3:
        return [0.5, 0.5, 0.8]  # nearest, middle, furthest/frontmost — spec default
    raise ValueError(f"--layer-alpha required when --copies != 3 (no default for {copies} sheets)")


def compute_layout(result, args, notches_pct=None):
    """Geometry shared by PNG and SVG output — computed exactly once so the
    two representations can never diverge (spec: 'SVG coordinates must
    match the PNG output exactly'). Framed on the 3-sheet extent, per spec:
    the device must land identically in both PNG files, not be framed
    independently — same scale/tx/ty reused for every sheet and for SVG.

    Returns a dict: scale/tx/ty, per-sheet canvas-space outline points
    (sheet_outlines_px[i]) and per-sheet canvas-space notch hole points
    (sheet_notches_px[i], one list of point-arrays per sheet), and the
    canvas-space stroke width.
    """
    body_w, body_h = result["body_bbox"][2], result["body_bbox"][3]
    base_outline = result["outline"]
    copies = args.copies

    outlines = sheet_outlines(base_outline, body_w, body_h, tuple(args.offset), tuple(args.step), copies)
    content_bbox = compute_content_bbox(result["body_bbox"], outlines)
    scale, tx, ty = compute_canvas_transform(content_bbox, args.size, margin_pct=4.0)
    stroke_width_px = STROKE_WIDTH_AT_1500 * (args.size / 1500.0)

    base_notch_points = [notch_pct_to_points(n, result["body_bbox"]) for n in (notches_pct or [])]
    ox, oy = sheet_offset_px(body_w, body_h, tuple(args.offset))
    sx, sy = sheet_offset_px(body_w, body_h, tuple(args.step))

    sheet_outlines_px, sheet_notches_px = [], []
    for i in range(copies):
        sheet_outlines_px.append(transform_outline(outlines[i], scale, tx, ty))
        dx, dy = ox + i * sx, oy + i * sy
        sheet_notches_px.append([transform_outline(translate_outline(pts, dx, dy), scale, tx, ty)
                                  for pts in base_notch_points])

    return {
        "scale": scale, "tx": tx, "ty": ty,
        "stroke_width_px": stroke_width_px,
        "sheet_outlines_px": sheet_outlines_px,
        "sheet_notches_px": sheet_notches_px,
    }


def render_outputs(img, result, args, notches_pct=None, layout=None):
    """Renders both PNG outputs off the SAME layout (see compute_layout),
    per spec: the device must land identically in both files, not be framed
    independently. Returns (img_1x, img_3x) uint8."""
    if layout is None:
        layout = compute_layout(result, args, notches_pct)
    scale, tx, ty = layout["scale"], layout["tx"], layout["ty"]
    stroke_width_px = layout["stroke_width_px"]
    copies = args.copies
    layer_alphas = resolve_layer_alphas(args, copies)

    base_canvas = place_image_on_canvas(img, scale, tx, ty, args.size)

    canvas_1x = base_canvas.astype(np.float64)
    render_sheet(canvas_1x, layout["sheet_outlines_px"][0], args.style_alpha, layer_alpha=1.0,
                 stroke_width_px=stroke_width_px, notch_rects_px=layout["sheet_notches_px"][0])

    canvas_3x = base_canvas.astype(np.float64)
    for i in range(copies):
        outline_px = layout["sheet_outlines_px"][i]
        render_sheet(canvas_3x, outline_px, args.style_alpha, layer_alphas[i],
                     stroke_width_px=stroke_width_px, notch_rects_px=layout["sheet_notches_px"][i])

    return (np.clip(canvas_1x, 0, 255).astype(np.uint8),
            np.clip(canvas_3x, 0, 255).astype(np.uint8))


# --- Stage 5: SVG export -----------------------------------------------------

def points_to_svg_subpath(points):
    """One closed SVG path subpath ('M x,y L x,y ... Z') from an (N,1,2)
    point array — used both for the outer outline and for each notch hole,
    so a compound path is just these strings concatenated (see
    compound_path_d)."""
    pts = points.reshape(-1, 2)
    if len(pts) == 0:
        return ""
    d = f"M {pts[0][0]:.2f},{pts[0][1]:.2f}"
    for x, y in pts[1:]:
        d += f" L {x:.2f},{y:.2f}"
    return d + " Z"


def compound_path_d(outline_points, notch_points_list):
    """Outer outline plus each notch as its own closed subpath, in one path
    'd' string. fill-rule=evenodd (set by the caller) turns the notch
    subpaths into holes regardless of winding direction — this is what
    spec's 'compound paths with cutouts as even-odd holes' means."""
    subpaths = [points_to_svg_subpath(outline_points)]
    subpaths.extend(points_to_svg_subpath(n) for n in notch_points_list)
    return " ".join(subpaths)


def render_svg(layout, size, copies):
    """One SVG with two named layers/groups, protector_1x and protector_3x,
    each holding one compound path per sheet — coordinates come from the
    exact same layout dict compute_layout produces for the PNG raster, so
    they can't diverge (spec: 'SVG coordinates must match the PNG output
    exactly').

    Fill/stroke here are just enough to make the shapes visible on open —
    this is meant as an Illustrator escape hatch (spec): select the sheets
    there and apply Graphic Style 4 for the real appearance, don't rely on
    this file's own paint.
    """
    stroke_w = layout["stroke_width_px"]

    def sheet_path(i):
        d = compound_path_d(layout["sheet_outlines_px"][i], layout["sheet_notches_px"][i])
        return (f'    <path d="{d}" fill-rule="evenodd" fill="#ffffff" fill-opacity="0.5" '
                f'stroke="#a4a4a4" stroke-width="{stroke_w:.2f}"/>')

    layer_1x = '  <g id="protector_1x">\n' + sheet_path(0) + '\n  </g>'
    layer_3x = ('  <g id="protector_3x">\n'
                + "\n".join(sheet_path(i) for i in range(copies))
                + '\n  </g>')

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">\n'
        f'{layer_1x}\n{layer_3x}\n'
        '</svg>\n'
    )


def process_image(path: Path, args):
    img = cv2.imread(str(path))
    if img is None:
        print(f"skip {path}: could not read image", file=sys.stderr)
        return

    inset_x_pct, inset_y_pct = resolve_inset_pct(args)
    result = detect_target(img, tol=args.tol, target=args.target,
                            inset_x_pct=inset_x_pct, inset_y_pct=inset_y_pct, fit_pct=args.fit)
    if result is None:
        print(f"skip {path}: no device found", file=sys.stderr)
        return
    if args.target != "body" and not result.get("target_found", True):
        print(f"skip {path}: body found but no {args.target} region detected inside it "
              f"(try --target body, or --debug to see why)", file=sys.stderr)
        return

    x, y, w, h = result["body_bbox"]
    shape = "circle/ellipse" if result["is_circle"] else f"{len(result['outline'])}-point polygon"
    tol_note = f"auto tol={result['tol_used']:.1f} (border p99={result['border_p99']:.1f})" if result["tol_auto"] \
        else f"fixed tol={result['tol_used']:.1f}"
    if args.target == "body":
        print(f"{path.name}: body bbox {w}x{h}px at ({x},{y}); "
              f"inset {result['inset_x_px']:.1f}x{result['inset_y_px']:.1f}px; "
              f"protector outline: {shape}; {tol_note}")
    else:
        ax, ay, aw, ah = result["active_region_bbox"]
        print(f"{path.name}: body bbox {w}x{h}px at ({x},{y}); "
              f"target={args.target} region {aw}x{ah}px at ({ax},{ay}); "
              f"fit {result['fit_x_px']:+.1f}x{result['fit_y_px']:+.1f}px; "
              f"protector outline: {shape}; {tol_note}")

    notches_pct = resolve_notches(args, img=img, result=result)
    if notches_pct:
        print(f"  {len(notches_pct)} notch(es) ({args.notches})")
    notch_points = [notch_pct_to_points(n, result["body_bbox"]) for n in notches_pct]

    if args.debug:
        args.outdir.mkdir(parents=True, exist_ok=True)
        dbg = draw_debug(img, result, notch_points)
        out_path = args.outdir / f"{path.stem}_debug.png"
        cv2.imwrite(str(out_path), dbg)
        print(f"  wrote {out_path}")

    layout = compute_layout(result, args, notches_pct)
    img_1x, img_3x = render_outputs(img, result, args, notches_pct, layout=layout)
    args.outdir.mkdir(parents=True, exist_ok=True)
    path_1x = args.outdir / f"{path.stem}_1x.png"
    path_3x = args.outdir / f"{path.stem}_3x.png"
    cv2.imwrite(str(path_1x), img_1x)
    cv2.imwrite(str(path_3x), img_3x)
    print(f"  wrote {path_1x}")
    print(f"  wrote {path_3x}")

    if args.svg:
        svg_text = render_svg(layout, args.size, args.copies)
        svg_path = args.outdir / f"{path.stem}.svg"
        svg_path.write_text(svg_text, encoding="utf-8")
        print(f"  wrote {svg_path}")


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
    p.add_argument("--target", choices=["body", "screen", "recess"], default="body")
    p.add_argument("--fit", type=float, default=0.0,
                    help="screen/recess only: signed %% of the detected active region's own "
                         "width/height. Positive dilates outward (bleed past a screen onto "
                         "the bezel); negative erodes inward (contract to fit inside a recess)")
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

    for path in collect_inputs(args.input):
        process_image(path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
