from pathlib import Path

import cv2
import numpy as np

import screenshield as ss

REF_1X = Path(__file__).parent / "ref_1x.jpg"

# Measured directly from ref_1x.jpg (not from the .ai file, per spec: rendered reference wins).
TRUE_BODY_BBOX = (61, 325, 1234, 694)  # x, y, w, h
TRUE_SHEET_EDGES = (166, 437, 1385, 1112)  # left, top, right, bottom, from stroke pixels
# 8.5 / 16.1 from PROMPT.md's rounded measurement does not match ref_1x.jpg's
# actual stroke pixels. Solved directly from clean pixel measurements of body
# edges (61,325)-(1295,1019) and sheet edges (166,437)-(1385,1111), each
# independently re-verified against raw pixels (not just taken on faith):
#   sheet_left  = body_left  + inset_x + offset_x  -> inset_x + offset_x = 105
#   sheet_right = body_right - inset_x + offset_x  -> -inset_x + offset_x = 90
#   sheet_top   = body_top   + inset_y + offset_y  -> inset_y + offset_y = 112
#   sheet_bot   = body_bottom - inset_y + offset_y -> -inset_y + offset_y = 92
# Solving: inset_x=7.5, offset_x=97.5, inset_y=10, offset_y=102.
# The inset is anisotropic (7.5px/side horizontally, 10px/side vertically) —
# not a single value applied uniformly by a circular kernel. See
# --inset-x/--inset-y in screenshield.py.
DEFAULT_OFFSET_PCT = (7.901, 14.697)
DEFAULT_INSET_X_PCT = 0.608  # 7.5 / 1234 * 100
DEFAULT_INSET_Y_PCT = 1.441  # 10 / 694 * 100

# ref_1x.jpg is a CONTAMINATED test image: it's a rendered output, so the
# protector sheet's own stroke is baked into it. That stroke crosses over the
# device and back onto the white background; its diff-from-white (~79-103)
# is enough to bridge into the device's contour at low/mid tol and inflate
# the detected body bbox by tens of px. tol=120 clears that specific bridge
# and reproduces the true body bbox exactly.
#
# This is NOT the production default. It's overfit to this one contaminated
# reference and would miss light-coloured devices entirely on real
# (clean, protector-free) input photos — see test_synthetic_light_grey_device
# and test_tol_120_breaks_on_light_device below for the case it silently
# breaks. The production default is auto_tol(), with FIXED_DEFAULT_TOL as
# its own conservative fallback.
REFERENCE_IMAGE_OVERRIDE_TOL = 120

# A SECOND, separate contamination effect, found while calibrating the
# outline-placement test below: the device's own edge isn't a hard
# black-to-white step, it has a genuine ~10px anti-aliased transition band
# (diff-from-white ~72-118 measured directly at x=700 on the bottom edge).
# tol=120 sits inside that band and truncates it, so the body CONTOUR is
# locally ~10px short of the true edge along flat runs — even though the
# bbox extreme (governed by a different point, e.g. a corner) still happens
# to land exactly on target. That makes tol=120 fine for a bbox check but
# wrong for anything that depends on the actual contour shape, like erosion.
# tol=80 sits below the transition band everywhere it was sampled, so the
# body contour tracks the true edge (verified directly: x=700 lands on
# y=1018, matching a clean pixel scan of the bezel), at the cost of a few px
# of bbox slop from a residual stroke-bridge nub the morphological open
# doesn't fully clean at this tol. No single tol satisfies both checks on
# this contaminated image — that's a property of the test fixture, not
# something to paper over by picking one number and hiding the other cost.
REFERENCE_IMAGE_SHAPE_TOL = 80


def _detect_reference(tol):
    img = cv2.imread(str(REF_1X))
    return ss.detect_body(img, tol=tol,
                           inset_x_pct=DEFAULT_INSET_X_PCT, inset_y_pct=DEFAULT_INSET_Y_PCT)


def test_body_bbox_matches_reference():
    result = _detect_reference(REFERENCE_IMAGE_OVERRIDE_TOL)
    x, y, w, h = result["body_bbox"]
    tx, ty, tw, th = TRUE_BODY_BBOX
    assert abs(x - tx) <= 2, f"body x off by {x - tx}"
    assert abs(y - ty) <= 2, f"body y off by {y - ty}"
    assert abs(w - tw) <= 2, f"body w off by {w - tw}"
    assert abs(h - th) <= 2, f"body h off by {h - th}"


def test_protector_outline_matches_reference_sheet():
    """The derived protector outline, placed at the sheet-1 offset, must land
    on the stroke edges measured directly from ref_1x.jpg within 2px.

    Uses REFERENCE_IMAGE_SHAPE_TOL, not REFERENCE_IMAGE_OVERRIDE_TOL: this
    test depends on the actual contour shape along each edge (erosion acts
    locally), not just the bbox extremes, and tol=120 corrupts that shape
    on this image. See the comment above REFERENCE_IMAGE_SHAPE_TOL."""
    result = _detect_reference(REFERENCE_IMAGE_SHAPE_TOL)
    assert not result["is_circle"], "reference device body is rectangular"

    _, _, body_w, body_h = result["body_bbox"]
    dx, dy = ss.sheet_offset_px(body_w, body_h, DEFAULT_OFFSET_PCT)

    placed = result["outline"].reshape(-1, 2).astype(np.float64)
    placed[:, 0] += dx
    placed[:, 1] += dy

    left, top, right, bottom = ss.outline_bounds(placed.reshape(-1, 1, 2))
    t_left, t_top, t_right, t_bottom = TRUE_SHEET_EDGES

    assert abs(left - t_left) <= 2, f"left edge off by {left - t_left:.1f}px"
    assert abs(top - t_top) <= 2, f"top edge off by {top - t_top:.1f}px"
    assert abs(right - t_right) <= 2, f"right edge off by {right - t_right:.1f}px"
    assert abs(bottom - t_bottom) <= 2, f"bottom edge off by {bottom - t_bottom:.1f}px"


def _synthetic_light_grey_device(diff_from_white=33, blur_sigma=0.8):
    """A clean (no protector baked in) synthetic photo: a light-grey device
    body on a white background, close enough in tone that a naive high tol
    would miss it. Meant to stand in for a white phone / silver watch / light
    grey camera body — exactly the case that overfitting tol to ref_1x.jpg's
    contamination would silently break."""
    canvas = np.full((900, 900, 3), 255, dtype=np.uint8)
    color = (255 - diff_from_white,) * 3
    true_box = (200, 250, 500, 400)  # x, y, w, h
    x, y, w, h = true_box
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness=-1)
    canvas = cv2.GaussianBlur(canvas, (5, 5), blur_sigma)  # soften edges like real photography
    return canvas, true_box


def test_synthetic_light_grey_device():
    """Production default (auto tol) must still find a low-contrast device."""
    canvas, true_box = _synthetic_light_grey_device()
    result = ss.detect_body(canvas, tol=None,
                             inset_x_pct=DEFAULT_INSET_X_PCT, inset_y_pct=DEFAULT_INSET_Y_PCT)
    assert result is not None, "auto tol failed to find the light-grey device at all"

    x, y, w, h = result["body_bbox"]
    tx, ty, tw, th = true_box
    assert abs(x - tx) <= 3, f"body x off by {x - tx}"
    assert abs(y - ty) <= 3, f"body y off by {y - ty}"
    assert abs(w - tw) <= 3, f"body w off by {w - tw}"
    assert abs(h - th) <= 3, f"body h off by {h - th}"


def test_tol_120_breaks_on_light_device():
    """Regression guard: the contaminated-reference override tol must NOT
    become the production default. At tol=120 the light-grey device (diff
    ~33 from background) is entirely below threshold and detection fails."""
    canvas, _ = _synthetic_light_grey_device()
    result = ss.detect_body(canvas, tol=REFERENCE_IMAGE_OVERRIDE_TOL,
                             inset_x_pct=DEFAULT_INSET_X_PCT, inset_y_pct=DEFAULT_INSET_Y_PCT)
    assert result is None, "tol=120 unexpectedly detected the light-grey device — test fixture drifted"


def test_auto_tol_falls_back_on_featureless_image():
    """No background/subject gap exists on a blank canvas — auto_tol must
    fall back to FIXED_DEFAULT_TOL rather than inventing a threshold from
    pure noise."""
    blank = np.full((400, 400, 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(0)
    blank = np.clip(blank.astype(np.int16) + rng.integers(-2, 3, blank.shape), 0, 255).astype(np.uint8)
    bg = ss.sample_background_color(blank)
    tol, used_auto, _ = ss.auto_tol(blank, bg)
    assert used_auto is False
    assert tol == ss.FIXED_DEFAULT_TOL


# --- Stage 2: opacity model -------------------------------------------------

# Spec's required test (PROMPT.md "Opacity" section): sampling ref_3x.jpg
# over the black bezel in each overlap region gives these four alpha
# values. Reproduce all four within 0.005.
REFERENCE_ALPHA_SHEET1_ALONE = 0.251
REFERENCE_ALPHA_SHEETS_1_2 = 0.439
REFERENCE_ALPHA_ALL_THREE = 0.663
REFERENCE_ALPHA_1X_SINGLE = 0.502

STYLE_ALPHA = 0.50
LAYER_ALPHAS_3X = [0.5, 0.5, 0.8]  # nearest, middle, furthest/frontmost


def test_composite_alpha_matches_reference_numbers():
    """Pure math check on the opacity model itself (no geometry, no pixels)
    — isolates the compositing formula so a bug here can't hide behind a
    geometry bug in the pixel-level test below."""
    effective = [STYLE_ALPHA * la for la in LAYER_ALPHAS_3X]

    sheet1_alone = ss.composite_alpha(effective[:1])
    sheets_1_2 = ss.composite_alpha(effective[:2])
    all_three = ss.composite_alpha(effective)
    single_1x = ss.composite_alpha([STYLE_ALPHA * 1.0])

    assert abs(sheet1_alone - REFERENCE_ALPHA_SHEET1_ALONE) <= 0.005
    assert abs(sheets_1_2 - REFERENCE_ALPHA_SHEETS_1_2) <= 0.005
    assert abs(all_three - REFERENCE_ALPHA_ALL_THREE) <= 0.005
    assert abs(single_1x - REFERENCE_ALPHA_1X_SINGLE) <= 0.005


def _sample_alpha_over_black(pixel_rgb):
    """White fill over a pure-black bezel: alpha recovers linearly."""
    return float(pixel_rgb[0]) / 255.0


def test_render_sheet_pixel_alpha_matches_reference_numbers():
    """End-to-end version of the same test: actually calls render_sheet (not
    just the abstract formula) on a black canvas with staggered rectangular
    sheets and samples real output pixels, so a bug in the mask/compositing
    code — not just the math — would show up here too."""
    canvas = np.zeros((400, 400, 3), dtype=np.float64)  # black bezel stand-in

    def rect_outline(x0, y0, x1, y1):
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64).reshape(-1, 1, 2)

    sheets = [
        rect_outline(50, 50, 250, 250),
        rect_outline(80, 80, 280, 280),
        rect_outline(110, 110, 310, 310),
    ]
    for outline, layer_alpha in zip(sheets, LAYER_ALPHAS_3X):
        ss.render_sheet(canvas, outline, STYLE_ALPHA, layer_alpha, stroke_width_px=0)

    sheet1_alone_px = canvas[60, 60]     # inside sheet1 only
    sheets_1_2_px = canvas[90, 90]       # inside sheet1+2, outside sheet3
    all_three_px = canvas[150, 150]      # inside all three

    assert abs(_sample_alpha_over_black(sheet1_alone_px) - REFERENCE_ALPHA_SHEET1_ALONE) <= 0.005
    assert abs(_sample_alpha_over_black(sheets_1_2_px) - REFERENCE_ALPHA_SHEETS_1_2) <= 0.005
    assert abs(_sample_alpha_over_black(all_three_px) - REFERENCE_ALPHA_ALL_THREE) <= 0.005

    canvas_1x = np.zeros((400, 400, 3), dtype=np.float64)
    ss.render_sheet(canvas_1x, sheets[0], STYLE_ALPHA, layer_alpha=1.0, stroke_width_px=0)
    single_1x_px = canvas_1x[60, 60]
    assert abs(_sample_alpha_over_black(single_1x_px) - REFERENCE_ALPHA_1X_SINGLE) <= 0.005


def test_canvas_transform_frames_three_sheet_extent_identically():
    """Both outputs must reuse the same transform — device lands at the same
    position/scale in the 1x and 3x file. Framing independently per-output
    (e.g. tightest-fit around just the 1x content) is exactly what the spec
    says not to do."""
    body_bbox = (100, 100, 400, 300)
    outlines = ss.sheet_outlines(
        np.array([[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.float64).reshape(-1, 1, 2),
        body_w=400, body_h=300, offset_pct=(8, 15), step_pct=(3, 8), copies=3,
    )
    content_bbox = ss.compute_content_bbox(body_bbox, outlines)
    scale, tx, ty = ss.compute_canvas_transform(content_bbox, canvas_size=1500, margin_pct=4.0)

    # sanity: transformed content bbox sits within canvas bounds with ~4% margin
    minx, miny, maxx, maxy = content_bbox
    left = minx * scale + tx
    top = miny * scale + ty
    right = maxx * scale + tx
    bottom = maxy * scale + ty
    assert left >= 0 and top >= 0 and right <= 1500 and bottom <= 1500
    # content is wider than tall, so width hits the ~4% margin tightly on
    # both sides; height (uniformly scaled, then centered) gets more margin
    # — that's correct aspect-preserving fit, not a bug.
    assert abs(left - 1500 * 0.04) < 1500 * 0.01
    assert abs((1500 - right) - 1500 * 0.04) < 1500 * 0.01
    assert top > 1500 * 0.04 and (1500 - bottom) > 1500 * 0.04


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
