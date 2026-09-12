import re
from pathlib import Path

import cv2
import numpy as np

import screenshield as ss

REF_1X = Path(__file__).parent / "ref_1x.jpg"
CLEAN_DEVICE = Path(__file__).parent / "input" / "test_clean_device.png"

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


# --- Stage 3: cutouts / notches ---------------------------------------------

# Measured directly from ref_1x.jpg, absolute pixels: two small rectangular
# cutouts on the device's right edge, visible through the sheet. These are
# SHEET-1 positions (the sheet is offset from the device by
# DEFAULT_OFFSET_PCT), not device positions — subtract sheet-1's own offset
# before converting to device-relative percentages, or the profile stores
# the wrong location for every sheet.
REFERENCE_CUTOUTS_SHEET_PX = [
    (1309, 521, 35, 15),
    (1309, 552, 35, 16),
]


def _sheet1_offset_px(body_bbox):
    _, _, body_w, body_h = body_bbox
    return ss.sheet_offset_px(body_w, body_h, DEFAULT_OFFSET_PCT)


def test_notch_pct_round_trips_reference_cutouts():
    """Convert the reference cutouts (measured on sheet 1) to device-relative
    percentages against TRUE_BODY_BBOX, then back to pixels, and confirm both
    the device-space rect AND the re-offset sheet-1 rect reproduce the
    original measurement exactly — this is the check that catches an
    off-by-the-offset bug rather than just an off-by-a-constant one."""
    ox, oy = _sheet1_offset_px(TRUE_BODY_BBOX)

    for sx, sy, sw, sh in REFERENCE_CUTOUTS_SHEET_PX:
        device_rect = (sx - ox, sy - oy, sw, sh)
        dx0, dy0, dw, dh = device_rect
        device_points = np.array(
            [[dx0, dy0], [dx0 + dw, dy0], [dx0 + dw, dy0 + dh], [dx0, dy0 + dh]],
            dtype=np.float64).reshape(-1, 1, 2)

        pct = ss.notch_points_to_pct(device_points, TRUE_BODY_BBOX)
        round_tripped = ss.notch_pct_to_points(pct, TRUE_BODY_BBOX)

        rx0, ry0, rx1, ry1 = ss.outline_bounds(round_tripped)
        assert abs(rx0 - dx0) < 1e-6
        assert abs(ry0 - dy0) < 1e-6
        assert abs((rx1 - rx0) - dw) < 1e-6
        assert abs((ry1 - ry0) - dh) < 1e-6

        # re-apply sheet-1's own offset and confirm we land back on the
        # ORIGINAL sheet-space measurement.
        resheeted = ss.translate_outline(round_tripped, ox, oy)
        sx0, sy0, sx1, sy1 = ss.outline_bounds(resheeted)
        assert abs(sx0 - sx) < 1e-6
        assert abs(sy0 - sy) < 1e-6
        assert abs((sx1 - sx0) - sw) < 1e-6
        assert abs((sy1 - sy0) - sh) < 1e-6


def test_notch_profile_json_round_trip(tmp_path):
    """--notch args written to --profile, then read back on a fresh 'next
    run' via --notches profile (no --notch given), must reproduce the exact
    same percentages — and, combined with the offset math above, the exact
    same absolute sheet-space position."""
    profile_path = tmp_path / "device.json"
    bx, by, bw, bh = TRUE_BODY_BBOX
    ox, oy = _sheet1_offset_px(TRUE_BODY_BBOX)

    device_rects_pct = []
    for sx, sy, sw, sh in REFERENCE_CUTOUTS_SHEET_PX:
        dx, dy = sx - ox, sy - oy
        device_rects_pct.append((
            (dx - bx) / bw * 100.0,
            (dy - by) / bh * 100.0,
            sw / bw * 100.0,
            sh / bh * 100.0,
        ))

    class Args:
        pass

    args = Args()
    args.notches = "profile"
    args.notch = [list(r) for r in device_rects_pct]
    args.profile = profile_path
    written = ss.resolve_notches(args)
    assert profile_path.exists()

    args2 = Args()  # fresh "next shot of the same model" — zero-input
    args2.notches = "profile"
    args2.notch = None
    args2.profile = profile_path
    read_back = ss.resolve_notches(args2)

    assert read_back == written
    for got, expected in zip(read_back, device_rects_pct):
        assert abs(got["x_pct"] - expected[0]) < 1e-9
        assert abs(got["y_pct"] - expected[1]) < 1e-9
        assert abs(got["w_pct"] - expected[2]) < 1e-9
        assert abs(got["h_pct"] - expected[3]) < 1e-9

    for notch_pct, (sx, sy, sw, sh) in zip(read_back, REFERENCE_CUTOUTS_SHEET_PX):
        device_points = ss.notch_pct_to_points(notch_pct, TRUE_BODY_BBOX)
        resheeted = ss.translate_outline(device_points, ox, oy)
        rx0, ry0, rx1, ry1 = ss.outline_bounds(resheeted)
        assert abs(rx0 - sx) < 1e-6
        assert abs(ry0 - sy) < 1e-6
        assert abs((rx1 - rx0) - sw) < 1e-6
        assert abs((ry1 - ry0) - sh) < 1e-6


def test_notches_none_mode_ignores_profile(tmp_path):
    """--notches none must return no notches even if a profile with saved
    notches exists — an explicit opt-out, not just 'no --notch given'."""
    profile_path = tmp_path / "device.json"
    ss.save_profile(profile_path, {"notches": [{"x_pct": 1, "y_pct": 1, "w_pct": 1, "h_pct": 1}]})

    class Args:
        pass

    args = Args()
    args.notches = "none"
    args.notch = None
    args.profile = profile_path
    assert ss.resolve_notches(args) == []


# --- Stage 3: --pick (interactive ROI selection) ----------------------------

def test_pick_rois_round_trip_reference_cutouts():
    """Feed synthetic ROI tuples — what cv2.selectROIs would return, in the
    SCALED display window's pixel space — directly into the conversion
    function they flow into, and confirm they land on the same percentages
    as the manual --notch path for the same physical locations. --pick
    operates on the input device photo directly (device-space already, no
    sheet offset involved), unlike the ref_1x.jpg-derived measurements
    above which were sheet-1 positions."""
    display_scale = 0.6
    ox, oy = _sheet1_offset_px(TRUE_BODY_BBOX)
    bx, by, bw, bh = TRUE_BODY_BBOX

    expected_pct = []
    synthetic_rois = []
    for sx, sy, sw, sh in REFERENCE_CUTOUTS_SHEET_PX:
        dx, dy = sx - ox, sy - oy  # device-space location of the real cutout
        expected_pct.append((
            (dx - bx) / bw * 100.0,
            (dy - by) / bh * 100.0,
            sw / bw * 100.0,
            sh / bh * 100.0,
        ))
        # what selectROIs would report after the display was shrunk by scale
        synthetic_rois.append((dx * display_scale, dy * display_scale,
                                sw * display_scale, sh * display_scale))

    got = ss.rois_to_notches_pct(synthetic_rois, display_scale, TRUE_BODY_BBOX)

    assert len(got) == len(expected_pct)
    for notch, (ex, ey, ew, eh) in zip(got, expected_pct):
        assert abs(notch["x_pct"] - ex) < 1e-6
        assert abs(notch["y_pct"] - ey) < 1e-6
        assert abs(notch["w_pct"] - ew) < 1e-6
        assert abs(notch["h_pct"] - eh) < 1e-6


def test_pick_display_scale_must_be_divided_out():
    """Regression guard for the specific bug the spec warned about: if the
    display-scale division were dropped, the same picked pixel ROI would
    silently produce a different (wrong) percentage depending on window
    size. Assert scale=1.0 and scale=0.5 on the identical ROI give the
    correctly-differing percentages, not the same wrong one."""
    body_bbox = (0, 0, 1000, 1000)
    roi = [(100, 100, 50, 50)]

    full_scale = ss.rois_to_notches_pct(roi, 1.0, body_bbox)[0]
    half_scale = ss.rois_to_notches_pct(roi, 0.5, body_bbox)[0]

    assert abs(full_scale["x_pct"] - 10.0) < 1e-9   # 100/1000*100
    assert abs(half_scale["x_pct"] - 20.0) < 1e-9   # (100/0.5)/1000*100
    assert full_scale != half_scale


def test_pick_writes_same_profile_schema_as_notch(tmp_path, monkeypatch):
    """--pick must write into the same profile JSON schema --notch produces,
    so the two routes are interchangeable. Exercises pick_notches end to
    end (not just the math) by monkeypatching cv2.selectROIs — no display
    needed, this is what makes it testable at all."""
    profile_pick = tmp_path / "pick.json"
    profile_notch = tmp_path / "notch.json"
    body_bbox = (0, 0, 1000, 800)
    display_scale = 0.5

    roi_px = (100, 100, 50, 40)  # real device-space rect this simulates
    synthetic_roi = (roi_px[0] * display_scale, roi_px[1] * display_scale,
                      roi_px[2] * display_scale, roi_px[3] * display_scale)

    monkeypatch.setattr(ss, "compute_display_scale", lambda shape, max_display=900: display_scale)
    monkeypatch.setattr(ss.cv2, "selectROIs", lambda *a, **k: np.array([synthetic_roi]))
    monkeypatch.setattr(ss.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(ss.cv2, "resize", lambda img, size, **k: img)  # skip the real resize

    fake_img = np.zeros((800, 1000, 3), dtype=np.uint8)
    picked = ss.pick_notches(fake_img, body_bbox, profile_path=profile_pick)

    class Args:
        pass

    args = Args()
    args.notches = "profile"
    args.notch = [[roi_px[0] / body_bbox[2] * 100.0, roi_px[1] / body_bbox[3] * 100.0,
                    roi_px[2] / body_bbox[2] * 100.0, roi_px[3] / body_bbox[3] * 100.0]]
    args.profile = profile_notch
    via_notch = ss.resolve_notches(args)

    assert set(ss.load_profile(profile_pick).keys()) == set(ss.load_profile(profile_notch).keys())
    assert len(picked) == len(via_notch) == 1
    for key in ("x_pct", "y_pct", "w_pct", "h_pct"):
        assert abs(picked[0][key] - via_notch[0][key]) < 1e-6


# --- Stage 3: --notches auto -------------------------------------------------

def test_notches_auto_reports_confidence_and_warns(capsys):
    """--notches auto must always print a warning to check the debug image,
    plus a confidence signal (found/kept/rejected counts) — required
    regardless of how accurate detection turns out to be on a given photo,
    since the method is explicitly best-effort (spec)."""
    img = cv2.imread(str(CLEAN_DEVICE))
    result = ss.detect_body(img, tol=None, inset_x_pct=DEFAULT_INSET_X_PCT, inset_y_pct=DEFAULT_INSET_Y_PCT)

    class Args:
        pass

    args = Args()
    args.notches = "auto"
    args.notch = None
    args.profile = None
    args.pick = False

    ss.resolve_notches(args, img=img, result=result)
    out = capsys.readouterr().out.lower()
    assert "warning" in out
    assert "check" in out and "debug" in out
    assert "kept" in out and "rejected" in out


def test_notches_auto_report_counts_are_consistent():
    """found == kept + rejected_area always, and the report dict always has
    the same keys — callers print these directly, they must never be
    missing. Also documents actual, honest behavior on the reference
    device: the two real cutouts are low-contrast (similar brightness to
    the surrounding grey bezel panel) and are NOT found by this brightness-
    based heuristic, while bright unrelated features (printed logo text,
    the top edge highlight) are — this is exactly the kind of false
    positive/false negative the spec warns is expected from auto mode."""
    img = cv2.imread(str(CLEAN_DEVICE))
    result = ss.detect_body(img, tol=None, inset_x_pct=DEFAULT_INSET_X_PCT, inset_y_pct=DEFAULT_INSET_Y_PCT)
    notches, report = ss.detect_notches_auto(img, result["body_contour"], result["body_bbox"])

    assert report["found"] == report["kept"] + report["rejected_area"]
    assert len(notches) == report["kept"]


# --- Stage 3: --notches auto, template relocation (better than brightness) --

def _checker_patch(size=(16, 20), c1=(40, 180, 230), c2=(230, 60, 40)):
    """A small distinctive 2x2 checkerboard patch — enough texture that
    template matching can't just latch onto a flat color anywhere."""
    h, w = size
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[: h // 2, : w // 2] = c1
    patch[h // 2:, w // 2:] = c1
    patch[: h // 2, w // 2:] = c2
    patch[h // 2:, : w // 2] = c2
    return patch


def _synthetic_capture(body_bbox, notch_rect_px, canvas_size=(300, 400)):
    """A synthetic 'photo': body_bbox on a black canvas with a distinctive
    checker patch stamped at notch_rect_px (device-space pixels)."""
    h, w = canvas_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    nx, ny, nw, nh = notch_rect_px
    img[ny:ny + nh, nx:nx + nw] = _checker_patch((nh, nw))
    return img


def test_relocate_by_template_finds_shifted_notch():
    """Cache a template from one synthetic photo, shift the same marker by
    a few px in a second synthetic photo (simulating 'position shifted
    slightly'), and confirm relocate finds the NEW position with a
    confident score — not just repeats the old profile-predicted one."""
    body_bbox = (50, 50, 300, 200)
    original_rect = (150, 120, 20, 16)  # nx, ny, nw, nh — device px

    captured_img = _synthetic_capture(body_bbox, original_rect)
    notch_points = np.array(
        [[original_rect[0], original_rect[1]],
         [original_rect[0] + original_rect[2], original_rect[1]],
         [original_rect[0] + original_rect[2], original_rect[1] + original_rect[3]],
         [original_rect[0], original_rect[1] + original_rect[3]]],
        dtype=np.float64).reshape(-1, 1, 2)
    cached_notch = ss.notch_points_to_pct(notch_points, body_bbox)
    ss.attach_notch_templates([cached_notch], captured_img, body_bbox)
    assert cached_notch.get("template_png_b64")

    shifted_rect = (158, 113, 20, 16)  # shifted +8, -7
    new_img = _synthetic_capture(body_bbox, shifted_rect)

    notches, report = ss.relocate_notches_by_template(new_img, body_bbox, [cached_notch])

    assert report["matched"] == 1
    assert report["low_confidence"] == 0
    assert report["scores"][0] > 0.9  # near-exact pixel match, should be very confident

    relocated_points = ss.notch_pct_to_points(notches[0], body_bbox)
    rx0, ry0, rx1, ry1 = ss.outline_bounds(relocated_points)
    assert abs(rx0 - shifted_rect[0]) <= 1
    assert abs(ry0 - shifted_rect[1]) <= 1
    assert abs((rx1 - rx0) - shifted_rect[2]) <= 1
    assert abs((ry1 - ry0) - shifted_rect[3]) <= 1


def test_relocate_by_template_low_confidence_keeps_profile_position():
    """If the cached marker is nowhere near the search window in the new
    photo (e.g. the cutout genuinely isn't there), relocation must not
    invent a confident-looking wrong answer — it should report low
    confidence and fall back to the profile's percentage-predicted
    position rather than the unrelated best-available match."""
    body_bbox = (50, 50, 300, 200)
    original_rect = (150, 120, 20, 16)

    captured_img = _synthetic_capture(body_bbox, original_rect)
    notch_points = np.array(
        [[original_rect[0], original_rect[1]],
         [original_rect[0] + original_rect[2], original_rect[1]],
         [original_rect[0] + original_rect[2], original_rect[1] + original_rect[3]],
         [original_rect[0], original_rect[1] + original_rect[3]]],
        dtype=np.float64).reshape(-1, 1, 2)
    cached_notch = ss.notch_points_to_pct(notch_points, body_bbox)
    ss.attach_notch_templates([cached_notch], captured_img, body_bbox)

    blank_img = np.zeros((300, 400, 3), dtype=np.uint8)  # marker absent entirely
    notches, report = ss.relocate_notches_by_template(blank_img, body_bbox, [cached_notch])

    assert report["matched"] == 0
    assert report["low_confidence"] == 1
    # falls back to the original profile position, unchanged
    assert abs(notches[0]["x_pct"] - cached_notch["x_pct"]) < 1e-9
    assert abs(notches[0]["y_pct"] - cached_notch["y_pct"]) < 1e-9


def test_relocate_by_template_scales_with_body_size():
    """A later photo of the same model at a different resolution/crop —
    the captured body size differs from the current one — must scale the
    cached template by that ratio before matching, not assume identical
    pixel scale."""
    capture_body_bbox = (50, 50, 300, 200)
    original_rect = (150, 120, 20, 16)
    captured_img = _synthetic_capture(capture_body_bbox, original_rect)
    notch_points = np.array(
        [[original_rect[0], original_rect[1]],
         [original_rect[0] + original_rect[2], original_rect[1]],
         [original_rect[0] + original_rect[2], original_rect[1] + original_rect[3]],
         [original_rect[0], original_rect[1] + original_rect[3]]],
        dtype=np.float64).reshape(-1, 1, 2)
    cached_notch = ss.notch_points_to_pct(notch_points, capture_body_bbox)
    ss.attach_notch_templates([cached_notch], captured_img, capture_body_bbox)

    # new photo: everything scaled up 2x (bigger body, bigger canvas)
    new_body_bbox = (100, 100, 600, 400)
    new_rect = (300, 240, 40, 32)  # same relative position/size, 2x scale
    new_img = _synthetic_capture(new_body_bbox, new_rect, canvas_size=(600, 800))

    notches, report = ss.relocate_notches_by_template(new_img, new_body_bbox, [cached_notch])

    assert report["matched"] == 1
    relocated_points = ss.notch_pct_to_points(notches[0], new_body_bbox)
    rx0, ry0, rx1, ry1 = ss.outline_bounds(relocated_points)
    assert abs(rx0 - new_rect[0]) <= 2
    assert abs(ry0 - new_rect[1]) <= 2
    assert abs((rx1 - rx0) - new_rect[2]) <= 2
    assert abs((ry1 - ry0) - new_rect[3]) <= 2


def test_notches_auto_uses_template_relocation_when_available(tmp_path, capsys):
    """resolve_notches('auto') must prefer template relocation over the
    brightness heuristic when the profile has cached templates."""
    body_bbox = (50, 50, 300, 200)
    original_rect = (150, 120, 20, 16)
    captured_img = _synthetic_capture(body_bbox, original_rect)
    notch_points = np.array(
        [[original_rect[0], original_rect[1]],
         [original_rect[0] + original_rect[2], original_rect[1]],
         [original_rect[0] + original_rect[2], original_rect[1] + original_rect[3]],
         [original_rect[0], original_rect[1] + original_rect[3]]],
        dtype=np.float64).reshape(-1, 1, 2)
    cached_notch = ss.notch_points_to_pct(notch_points, body_bbox)
    ss.attach_notch_templates([cached_notch], captured_img, body_bbox)

    profile_path = tmp_path / "device.json"
    ss.save_profile(profile_path, {"notches": [cached_notch]})

    shifted_rect = (152, 118, 20, 16)
    new_img = _synthetic_capture(body_bbox, shifted_rect)

    class FakeResult(dict):
        pass

    result = {"body_bbox": body_bbox, "body_contour": np.array(
        [[[50, 50]], [[350, 50]], [[350, 250]], [[50, 250]]], dtype=np.int32)}

    class Args:
        pass

    args = Args()
    args.notches = "auto"
    args.notch = None
    args.profile = profile_path
    args.pick = False

    notches = ss.resolve_notches(args, img=new_img, result=result)
    out = capsys.readouterr().out.lower()

    assert "template relocate" in out
    assert "brightness" not in out.split("no cached")[0]  # brightness fallback text not used
    assert len(notches) == 1


def test_notches_auto_falls_back_to_brightness_without_cached_templates(tmp_path, capsys):
    """No profile, or a profile with no cached templates, must fall back to
    the original brightness heuristic exactly as before — it's the
    fallback, not replaced."""
    img = cv2.imread(str(CLEAN_DEVICE))
    result = ss.detect_body(img, tol=None, inset_x_pct=DEFAULT_INSET_X_PCT, inset_y_pct=DEFAULT_INSET_Y_PCT)

    class Args:
        pass

    args = Args()
    args.notches = "auto"
    args.notch = None
    args.profile = None
    args.pick = False

    ss.resolve_notches(args, img=img, result=result)
    out = capsys.readouterr().out.lower()
    assert "brightness fallback" in out
    assert "template relocate" not in out


# --- Stage 4: --target screen / --target recess, signed --fit ---------------

def _synthetic_round_watch(body_r=150, active_r=100, bright=True,
                            center=(220, 220), canvas=440):
    """A round body (dark bezel) with a concentric round active region —
    bright (screen-like) or dark (recess-like) — for exercising the
    circle/ellipse-fitting branch, which the rectangular reference device
    never touches."""
    img = np.full((canvas, canvas, 3), 255, dtype=np.uint8)
    cv2.circle(img, center, body_r, (30, 30, 30), -1)
    color = (230, 200, 80) if bright else (10, 10, 10)
    cv2.circle(img, center, active_r, color, -1)
    return img


def _outline_center_radius(outline):
    pts = outline.reshape(-1, 2)
    cx, cy = pts.mean(axis=0)
    r = float(np.mean(np.linalg.norm(pts - [cx, cy], axis=1)))
    return cx, cy, r


def test_target_screen_stays_round_and_fit_zero_matches_region():
    """Spec: 'a round watch crystal must stay round' — must fit an ellipse,
    not force a low-vertex polygon. fit=0 should reproduce the detected
    region's own radius (no offset)."""
    img = _synthetic_round_watch(active_r=100, bright=True)
    result = ss.detect_target(img, tol=None, target="screen", fit_pct=0.0)

    assert result["target_found"] is True
    assert result["is_circle"] is True
    _, _, r = _outline_center_radius(result["outline"])
    assert abs(r - 100) < 3


def test_fit_sign_convention_positive_bleeds_negative_contracts():
    """Positive --fit dilates outward past the detected region (bleed onto
    the bezel, screen use case); negative erodes inward (contract to fit
    inside, recess use case). Both signs measured against the SAME region
    so only the sign/magnitude of --fit explains the radius change."""
    img = _synthetic_round_watch(active_r=100, bright=True)

    r0 = _outline_center_radius(ss.detect_target(img, tol=None, target="screen", fit_pct=0.0)["outline"])[2]
    r_pos = _outline_center_radius(ss.detect_target(img, tol=None, target="screen", fit_pct=15.0)["outline"])[2]
    r_neg = _outline_center_radius(ss.detect_target(img, tol=None, target="screen", fit_pct=-15.0)["outline"])[2]

    assert r_pos > r0 > r_neg
    # active region bbox width ~199px (diameter 200, minus AA/morphology
    # slop) -> fit_x_px = 15% of that ~30px, added/subtracted directly to
    # the radius by the dilate/erode kernel -> +-30px, roughly.
    assert abs((r_pos - r0) - 30) < 6
    assert abs((r0 - r_neg) - 30) < 6


def test_target_recess_detects_dark_region_not_bright():
    """recess is the opposite polarity from screen: a locally DARK region
    inside the body (e.g. a sunken camera lens), not a bright one. This
    also regression-guards the Otsu boundary bug found during development —
    a hard two-level image (exactly 10 vs 30) put the Otsu threshold AT the
    dark class's own value, and a strict '<' comparison found nothing."""
    dark_img = _synthetic_round_watch(active_r=100, bright=False)
    result = ss.detect_target(dark_img, tol=None, target="recess", fit_pct=0.0)
    assert result["target_found"] is True
    _, _, r = _outline_center_radius(result["outline"])
    assert abs(r - 100) < 3

    # and screen mode must NOT find this same dark region (wrong polarity)
    result_wrong_mode = ss.detect_target(dark_img, tol=None, target="screen", fit_pct=0.0)
    if result_wrong_mode["target_found"]:
        _, _, r_wrong = _outline_center_radius(result_wrong_mode["outline"])
        assert abs(r_wrong - 100) > 20  # did not find the same 100px recess


def test_target_screen_not_found_on_uniform_body():
    """A body with no actual distinct screen/recess feature (uniform color)
    must report target_found=False, not silently return the whole body (or
    a noise speck) as if it were a legitimate detection."""
    uniform = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.circle(uniform, (150, 150), 100, (30, 30, 30), -1)
    result = ss.detect_target(uniform, tol=None, target="screen", fit_pct=0.0)
    assert result is not None  # body itself was still found
    assert result.get("target_found") is False


def test_target_body_mode_unaffected_by_detect_target_refactor():
    """detect_target(target='body') must behave identically to the
    original detect_body — this is a regression guard for the refactor
    that generalized derive_outline to a signed offset."""
    img = cv2.imread(str(CLEAN_DEVICE))
    via_detect_body = ss.detect_body(img, tol=None, inset_x_pct=DEFAULT_INSET_X_PCT,
                                      inset_y_pct=DEFAULT_INSET_Y_PCT)
    via_detect_target = ss.detect_target(img, tol=None, target="body",
                                          inset_x_pct=DEFAULT_INSET_X_PCT, inset_y_pct=DEFAULT_INSET_Y_PCT)
    assert via_detect_body["body_bbox"] == via_detect_target["body_bbox"]
    np.testing.assert_array_equal(via_detect_body["outline"], via_detect_target["outline"])
    assert via_detect_body["is_circle"] == via_detect_target["is_circle"]


# --- Stage 5: SVG export -----------------------------------------------------

def test_svg_matches_png_layout_exactly():
    """SVG path coordinates must come from the exact same geometry as the
    PNG (spec: 'SVG coordinates must match the PNG output exactly') — parse
    the emitted path data back out and compare against the layout dict
    render_outputs uses for the raster, point for point, not just eyeball
    it. Also checks the required two-layer structure and one compound path
    per sheet with cutouts included as extra subpaths."""
    img = cv2.imread(str(CLEAN_DEVICE))
    profile_path = Path(__file__).parent / "profiles" / "chigee_test.json"
    args = ss.build_parser().parse_args(
        [str(CLEAN_DEVICE), "--profile", str(profile_path), "--svg"])

    result = ss.detect_target(img, tol=args.tol, target=args.target,
                               inset_x_pct=DEFAULT_INSET_X_PCT, inset_y_pct=DEFAULT_INSET_Y_PCT,
                               fit_pct=args.fit)
    notches_pct = ss.resolve_notches(args, img=img, result=result)
    layout = ss.compute_layout(result, args, notches_pct)
    svg_text = ss.render_svg(layout, args.size, args.copies)

    assert svg_text.count('<g id="protector_1x">') == 1
    assert svg_text.count('<g id="protector_3x">') == 1
    assert svg_text.count("<path") == 1 + args.copies  # one in 1x layer, one per sheet in 3x layer
    assert f'width="{args.size}"' in svg_text and f'height="{args.size}"' in svg_text
    assert 'fill-rule="evenodd"' in svg_text
    assert len(notches_pct) == 2  # sanity: the profile's real cutouts loaded, so holes are exercised below

    all_paths_d = re.findall(r'<path d="([^"]+)"', svg_text)
    assert len(all_paths_d) == 1 + args.copies

    def parse_points(d):
        return [(float(x), float(y)) for x, y in re.findall(r"(-?\d+\.\d+),(-?\d+\.\d+)", d)]

    def expected_points(sheet_index):
        outline = layout["sheet_outlines_px"][sheet_index].reshape(-1, 2)
        notches = [n.reshape(-1, 2) for n in layout["sheet_notches_px"][sheet_index]]
        return np.vstack([outline] + notches) if notches else outline

    # path 0 = the 1x layer's single sheet (sheet 0)
    svg_pts = parse_points(all_paths_d[0])
    expected = expected_points(0)
    assert len(svg_pts) == len(expected)
    for (sx, sy), (ex, ey) in zip(svg_pts, expected):
        assert abs(sx - ex) < 0.02
        assert abs(sy - ey) < 0.02

    # paths 1..copies = the 3x layer's sheets, in order
    for i in range(args.copies):
        svg_pts = parse_points(all_paths_d[1 + i])
        expected = expected_points(i)
        assert len(svg_pts) == len(expected)
        for (sx, sy), (ex, ey) in zip(svg_pts, expected):
            assert abs(sx - ex) < 0.02
            assert abs(sy - ey) < 0.02


def test_svg_written_only_with_flag(tmp_path):
    """--svg is opt-in — no .svg file without it."""
    img = cv2.imread(str(CLEAN_DEVICE))
    args = ss.build_parser().parse_args([str(CLEAN_DEVICE), "--outdir", str(tmp_path)])
    assert args.svg is False
    ss.process_image(CLEAN_DEVICE, args)
    assert not any(tmp_path.glob("*.svg"))
    assert any(tmp_path.glob("*_1x.png"))  # sanity: it did actually run


def test_compound_path_holes_are_evenodd_subpaths():
    """A notch hole must be its own closed subpath (separate 'M...Z'), not
    merged into the outer outline's point list — that's what makes
    fill-rule=evenodd treat it as a hole."""
    outline = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64).reshape(-1, 1, 2)
    notch = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float64).reshape(-1, 1, 2)
    d = ss.compound_path_d(outline, [notch])
    assert d.count("M") == 2
    assert d.count("Z") == 2


# --- Stroke sub-pixel width (post-stage-5 follow-up) -------------------------

def test_stroke_width_is_subpixel_accurate_not_rounded():
    """Regression guard for the fix: cv2.polylines only accepts integer
    thickness, so a naive round(1.5) draws a 2px-thick stroke (~33% too
    thick). stroke_coverage_map must integrate to close to the requested
    sub-pixel width instead. Measured against ref_1x.jpg (see CLAUDE.md): a
    clean axis-aligned stroke crossing showed a pixel at deficit 104/255,
    putting a hard lower bound on the true width that rules out anything
    close to 2px — the reference is closer to 1.5 than 2, per the spec's
    own stated stroke width."""
    pts = np.array([[50, 10], [50, 90]], dtype=np.float64).reshape(-1, 1, 2)

    integrated_15 = ss.stroke_coverage_map((100, 100), [pts], stroke_width_px=1.5)[50, :].sum()
    integrated_20 = ss.stroke_coverage_map((100, 100), [pts], stroke_width_px=2.0)[50, :].sum()

    # within ~5% of the requested width, not rounded up to the next integer
    assert abs(integrated_15 - 1.5) < 0.1
    assert abs(integrated_20 - 2.0) < 0.1
    # and a 1.5 target must land closer to 1.5 than a naive round-to-2 would
    assert abs(integrated_15 - 1.5) < abs(integrated_15 - 2.0)


# --- Stage 6: batch mode -----------------------------------------------------

def _write_synthetic_device_png(path):
    canvas, _ = _synthetic_light_grey_device()
    cv2.imwrite(str(path), canvas)


def _batch_args(input_path, outdir, **overrides):
    argv = [str(input_path), "--outdir", str(outdir)]
    args = ss.build_parser().parse_args(argv)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def test_process_image_skips_when_outputs_already_exist(tmp_path):
    """Second run over the same file, without --force, must not redo the
    work — this is what makes 'rerun just the one that failed' cheap
    instead of reprocessing the whole batch."""
    src = tmp_path / "device.png"
    _write_synthetic_device_png(src)
    outdir = tmp_path / "out"

    args = _batch_args(src, outdir)
    first = ss.process_image(src, args)
    assert first["status"] == "ok"
    mtime_1x = (outdir / "device_1x.png").stat().st_mtime

    second = ss.process_image(src, args)
    assert second["status"] == "skipped"
    assert (outdir / "device_1x.png").stat().st_mtime == mtime_1x  # untouched


def test_force_reprocesses_existing_outputs(tmp_path):
    src = tmp_path / "device.png"
    _write_synthetic_device_png(src)
    outdir = tmp_path / "out"

    ss.process_image(src, _batch_args(src, outdir))
    forced = ss.process_image(src, _batch_args(src, outdir, force=True))
    assert forced["status"] == "ok"


def test_outputs_exist_accounts_for_debug_and_svg_flags(tmp_path):
    """Re-running with --debug or --svg added, when only the plain PNGs
    exist from a prior run, must NOT be treated as already-done — those
    specific files genuinely don't exist yet."""
    src = tmp_path / "device.png"
    _write_synthetic_device_png(src)
    outdir = tmp_path / "out"

    ss.process_image(src, _batch_args(src, outdir))  # plain run: _1x/_3x only
    assert not ss.outputs_exist(src, _batch_args(src, outdir, debug=True))
    assert not ss.outputs_exist(src, _batch_args(src, outdir, svg=True))
    assert ss.outputs_exist(src, _batch_args(src, outdir))


def test_batch_one_failure_does_not_abort_the_rest(tmp_path, capsys):
    """A folder of several files where one fails (e.g. a blank image with
    no detectable device) must still process the others, exit non-zero,
    and let a rerun of the same command redo only the failed one (the
    others get skipped via outputs-already-exist)."""
    folder = tmp_path / "batch"
    folder.mkdir()
    _write_synthetic_device_png(folder / "a_good.png")
    _write_synthetic_device_png(folder / "c_good.png")
    cv2.imwrite(str(folder / "b_blank.png"), np.full((300, 300, 3), 255, dtype=np.uint8))
    outdir = tmp_path / "out"

    exit_code = ss.main([str(folder), "--outdir", str(outdir)])

    assert exit_code == 1
    assert (outdir / "a_good_1x.png").exists()
    assert (outdir / "c_good_1x.png").exists()
    assert not (outdir / "b_blank_1x.png").exists()

    out = capsys.readouterr().out
    assert "3 file(s): 2 ok, 0 skipped, 1 failed" in out
    assert "b_blank.png" in out

    # rerun: the two good ones are skipped (already exist), only the
    # failure is retried — and fails again for the same reason, not a crash.
    exit_code_2 = ss.main([str(folder), "--outdir", str(outdir)])
    out2 = capsys.readouterr().out
    assert exit_code_2 == 1
    assert "3 file(s): 0 ok, 2 skipped, 1 failed" in out2


def test_batch_summary_lists_ok_and_failed_files(tmp_path, capsys):
    folder = tmp_path / "batch2"
    folder.mkdir()
    _write_synthetic_device_png(folder / "good.png")
    cv2.imwrite(str(folder / "bad.png"), np.full((300, 300, 3), 255, dtype=np.uint8))
    outdir = tmp_path / "out2"

    exit_code = ss.main([str(folder), "--outdir", str(outdir)])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "2 file(s): 1 ok, 0 skipped, 1 failed" in out
    assert "bad.png: no device found" in out


def test_unexpected_exception_in_one_file_does_not_abort_batch(tmp_path, monkeypatch, capsys):
    """A genuine crash (not just a detection failure) partway through one
    file must be caught by main(), recorded as failed, and not stop the
    rest of the batch."""
    folder = tmp_path / "batch3"
    folder.mkdir()
    _write_synthetic_device_png(folder / "a_good.png")
    _write_synthetic_device_png(folder / "z_good.png")
    outdir = tmp_path / "out3"

    real_render_outputs = ss.render_outputs

    # simplest reliable trigger: fail deterministically for the first file
    # processed (alphabetical order -> a_good.png), succeed for the rest.
    call_count = {"n": 0}

    def flaky_render_outputs(img, result, args, notches_pct=None, layout=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom")
        return real_render_outputs(img, result, args, notches_pct, layout=layout)

    monkeypatch.setattr(ss, "render_outputs", flaky_render_outputs)

    exit_code = ss.main([str(folder), "--outdir", str(outdir)])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert not (outdir / "a_good_1x.png").exists()
    assert (outdir / "z_good_1x.png").exists()
    assert "2 file(s): 1 ok, 0 skipped, 1 failed" in out
    assert "a_good.png: RuntimeError: boom" in out


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
