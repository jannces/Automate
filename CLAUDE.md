# Automate

## screenshield/

CLI tool that derives screen-protector overlay images from a device photo. Full spec: `screenshield/PROMPT.md`. Build proceeds in stages, stopping after each for review (see PROMPT.md "Build order"). **Stages 1-4 are settled** as of this writing (body detection, compositing/opacity/canvas framing, cutouts, `--target screen`/`recess`); stage 5 (SVG export) has not started.

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

### Stage 3: cutouts/notches

`--notch`/`--profile`/`--pick` are the reliable, spec-required paths — notches stored as percentages of the body bbox in the profile JSON, offset math verified against real measured cutout positions (see `test_notch_pct_round_trips_reference_cutouts` — note the reference cutouts were measured on the *sheet*, not the device, so sheet-1's offset has to be subtracted before storing).

`--notches auto` has two paths: brightness heuristic (spec's original description — best-effort, known false positives, e.g. picks up printed logo text; **missed both real cutouts entirely** on the reference device since they're similar brightness to the surrounding bezel panel, not bright/saturated) and, on top of that, **template relocation**: when a profile has cutouts previously defined via `--notch`/`--pick`, a small cached image patch of each is matched locally (constrained to a small window around the profile's predicted position, not a blind scan) on the new photo, scaled by the ratio of body sizes to handle a resolution/crop difference.

**Template relocation's value is unproven on real data.** The only real-photo demo was a rigid whole-image shift (translate the whole photo a few px) — body-bbox re-detection already tracks that correctly on its own via percentage placement, so that demo didn't actually exercise the scenario relocation is *for* (a cutout moving independently of the body, e.g. manufacturing tolerance or non-rigid drift). That scenario is only validated with synthetic checkerboard-patch tests, not a real photo pair. If real repeat-model photos show relocation and plain percentage placement always agreeing, consider removing template relocation as unnecessary complexity — `--profile` alone may already be sufficient.

### Stage 4: `--target screen` / `--target recess`, signed `--fit`

`--target recess` and `--fit` are not in `PROMPT.md` — added on request, design choices made without a reference device for either (same "best-effort, unvalidated" caveat as auto-notches):

- `--fit` replaces `--bleed`, signed, applied against the detected active region's own width/height (not the body's): **positive dilates outward** (bleed past a screen onto the bezel), **negative erodes inward** (contract to fit inside a recess). Implemented by generalizing `derive_outline`'s erosion to `offset_filled_mask()` — positive offset erodes, negative dilates, single elliptical kernel either way.
- `screen` detection: largest coherent bright-or-saturated region inside the body (spec's own description). `recess` detection: the opposite polarity — largest coherent **dark** region (a sunken feature like a camera lens reads as locally shadowed, not bright). Both via Otsu restricted to body pixels, same "let the histogram find the gap" pattern as `auto_tol`.
- **Real bug hit and fixed**: cv2's Otsu pairs with `THRESH_BINARY`'s "> thresh" bright class, so the dark class is `<= thresh`, not `< thresh`. On a hard two-level test image (e.g. exactly value 10 vs 30) Otsu placed the threshold *at* the dark class's own value, and `<` selected nothing at all. Regression-guarded by `test_target_recess_detects_dark_region_not_bright`.
- A body with no real screen/recess feature (uniform color, or a degenerate Otsu split) must not silently return the whole body or a noise speck as if it were a real detection — `find_active_region_contour` rejects candidates outside a `[0.5%, 90%]` share of the body's area and reports `target_found: False` instead.
- Circle/ellipse fitting (`is_near_circular`/`cv2.fitEllipse`, built in stage 1) was never actually exercised until stage 4 — the reference device is rectangular. Covered now via a synthetic round-watch fixture in `test_screenshield.py`.

### Test suite (`screenshield/test_screenshield.py`)

26 tests, all passing, covering stages 1-4: body bbox/outline accuracy against the reference (split contaminated-image tolerances, see above), `auto_tol` behavior, the opacity/alpha-compositing math and canvas framing (stage 2), notch percentage/profile round-trips + `--pick` ROI math + template relocation (stage 3), and `--target screen`/`recess` + signed `--fit` including the Otsu boundary regression (stage 4). Test names are self-describing; read the file rather than this list, which will go stale faster than the code does.

Two permanent test-image fixtures beyond `ref_1x.jpg`/`ref_3x.jpg`: `screenshield/input/test_clean_device.png` (the reference device cropped to its body bbox and pasted on white — the protector's own edges fall outside that crop, so this is genuinely uncontaminated, used to verify `auto_tol`/detection without any override tolerance) and `screenshield/profiles/chigee_test.json` (a real notch profile with cached templates, built from it).

Run with `py -3 -m pytest screenshield/test_screenshield.py -v` (this machine has no bare `python`/`python3` on PATH — use the `py` launcher).
