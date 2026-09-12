# Automate

## screenshield/

CLI tool that derives screen-protector overlay images from a device photo. Full spec: `screenshield/PROMPT.md`. Build proceeds in stages, stopping after each for review (see PROMPT.md "Build order"). **Stage 1 (body detection + outline derivation) is settled** as of this writing; stage 2 (compositing/opacity/canvas framing) has not started.

Everything below was derived empirically against `screenshield/ref_1x.jpg` and `ref_3x.jpg` and is settled — don't re-derive it from scratch in a fresh session, extend from it.

### Solved geometry constants

```
inset_x = 7.5px   = 0.608% of body width   (--inset-x default)
inset_y = 10px    = 1.441% of body height  (--inset-y default)
offset_x = 97.5px = 7.901% of body width   (--offset default, x)
offset_y = 102px  = 14.697% of body height (--offset default, y)
```

`PROMPT.md`'s own stated defaults (inset 0.57% of width uniformly, offset 8.5%/16.1%) do **not** match the rendered reference — they were rounded/approximate. Per the spec's own rule, the rendered reference wins.

**The inset is anisotropic** — not one value applied uniformly. A single `inset_x_px` eroded via a circular kernel reproduced the reference body bbox but left the derived sheet ~3-4px short vertically; splitting into `inset_x`/`inset_y` and eroding with an **elliptical** kernel sized `(2·inset_x+1, 2·inset_y+1)` (in `derive_outline`) fixed it exactly. Erode anisotropically at the kernel — don't erode uniformly and patch the result afterward, that cannot reproduce a non-uniform true inset.

**How the four constants were derived:** not by guessing and checking, but by solving two linear equations per axis directly from clean pixel measurements (independently re-verified against raw pixel scans, not taken from any single derived intermediate number):

```
sheet_left  = body_left  + inset_x + offset_x
sheet_right = body_right - inset_x + offset_x
sheet_top   = body_top   + inset_y + offset_y
sheet_bottom= body_bottom - inset_y + offset_y
```

With body edges `(61,325)-(1295,1019)` and sheet edges `(166,437)-(1385,1112)` (all measured directly off `ref_1x.jpg` pixels, not inferred from bounding boxes of a derived outline), solving both equations per axis gives the constants above. This is the pattern to reuse if any of these numbers ever need re-calibrating against a new/updated reference: measure both edges of both shapes directly and solve, don't back-out one unknown while holding an assumed value for the other.

Regression test: `screenshield/test_screenshield.py::test_protector_outline_matches_reference_sheet` — places the derived outline at the sheet-1 offset and asserts all four edges land within 2px of the measured reference. No per-edge special-casing.

### auto_tol — why it exists, and why tol=120 must never be a default

`--tol` (background/subject threshold) defaults to `None`, which triggers `auto_tol()`: Otsu's threshold on the whole-image background-deviation histogram, trusted only if there's a genuine valley there (few pixels sit near the cut vs. the background peak — checked, not assumed). If no confident gap is found (e.g. a blank/textured image with no real subject), it falls back to `FIXED_DEFAULT_TOL = 30`, calibrated with ~2x margin over the anti-aliasing halo measured on the reference (~14 levels wide).

`tol=120` came out of an earlier calibration pass against `ref_1x.jpg` and produced an *exact* body-bbox match — but that image is contaminated (see below), and 120 is overfit to it. **It must never become the production default.** At that threshold, a light-coloured device (white phone, silver watch, light-grey camera body — diff from a white background well under 120) is missed entirely. Guarded by `test_tol_120_breaks_on_light_device`, which asserts detection fails at tol=120 against a synthetic light-grey device that `auto_tol` handles fine.

### Why ref_1x.jpg needs two different override tolerances for testing

`ref_1x.jpg` is a *rendered output*, not a clean input photo — the protector sheet's own stroke is baked into it. This causes two separate, unrelated problems depending on tol:

1. **Stroke bridging (low/mid tol).** The sheet's stroke crosses over the device and back onto the white background; its diff-from-white (~79-103) is enough to bridge into the device's contour and inflate the detected body by tens of px, unless tol clears ~103.
2. **Anti-aliased edge truncation (high tol).** The device's own edge isn't a hard step — there's a genuine ~10px transition band (diff-from-white measured ~72-118 at one sampled column). `tol=120` sits inside that band and truncates it: the body *contour* ends up ~10px short of the true edge along flat runs, even though the *bounding-box extreme* still happens to land on target (governed by some other point, e.g. a corner). So tol=120 gives an exact bbox by coincidence while corrupting the actual contour shape that erosion depends on.

No single tol on this contaminated image satisfies both "exact bbox" and "shape-accurate contour" at once — that's a property of the fixture, not something to hide behind one convenient number. Split accordingly in `test_screenshield.py`:

- `REFERENCE_IMAGE_OVERRIDE_TOL = 120` — used only by `test_body_bbox_matches_reference`. Clears the stroke bridge, exact bbox match.
- `REFERENCE_IMAGE_SHAPE_TOL = 80` — used by `test_protector_outline_matches_reference_sheet`. Sits below the anti-aliased transition band everywhere it was sampled (verified: contour at x=700 lands on y=1018, matching a direct clean pixel scan of the bezel), at the cost of a few px of bbox slop from a residual stroke-bridge nub morphological opening doesn't fully clean at this tol. Outline-shape fidelity matters more than bbox exactness here because erosion acts on the local contour, not the bbox extremes.

Neither value belongs anywhere near production code paths — both are contaminated-reference-only test constants.

### CHAIN_APPROX_NONE is required for the eroded contour

`largest_external_contour()` takes a `chain` parameter, `cv2.CHAIN_APPROX_SIMPLE` by default (fine for body detection — only bbox/area needed there). But the eroded contour, used for outline simplification and the line-fit-snap corner refinement, **must** be re-contoured with `cv2.CHAIN_APPROX_NONE`. `CHAIN_APPROX_SIMPLE` collapses straight runs down to just their endpoints — a perfectly vertical/horizontal edge degenerates to exactly 2 points, both already corner-adjacent, leaving nothing for a corner-trimmed line fit to use. This was a real bug found mid-session: switching to `CHAIN_APPROX_NONE` for `derive_outline`'s re-contour call gave the dense, uniform-density point cloud line-fitting actually needs.

### Line-fit-snap

`snap_polygon_to_lines()`: after `approxPolyDP` simplifies the eroded contour to a polygon, each edge gets a least-squares line fit (`cv2.fitLine`) to the raw dense contour points along it, and each vertex is recomputed as the intersection of its two adjacent fitted lines. This recovers the crisp corner that erosion's rounding destroys.

The corner-exclusion margin is a **fixed Euclidean distance** (`corner_margin_px = 1.5 × max(inset_x_px, inset_y_px)`), not a fraction of segment point count — corner rounding radius is a function of the erosion kernel, not of how long an edge happens to be. A short chamfer between two long sides can be almost entirely inside the fillet radius; trimming by point-count fraction under-trimmed it and pulled fitted corners several px off. Edges with too few points left after trimming keep their original `approxPolyDP` vertex rather than fit to corner-contaminated noise.

### Current test suite (`screenshield/test_screenshield.py`), all passing

- `test_body_bbox_matches_reference` — exact body bbox on the contaminated reference, tol=120.
- `test_protector_outline_matches_reference_sheet` — all 4 placed-outline edges within 2px, tol=80.
- `test_synthetic_light_grey_device` — auto_tol finds a clean, low-contrast synthetic device.
- `test_tol_120_breaks_on_light_device` — regression guard, tol=120 fails on the same synthetic device.
- `test_auto_tol_falls_back_on_featureless_image` — no gap in histogram → falls back to `FIXED_DEFAULT_TOL`.

Run with `py -3 -m pytest screenshield/test_screenshield.py -v` (this machine has no bare `python`/`python3` on PATH — use the `py` launcher).
