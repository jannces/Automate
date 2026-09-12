# Automate

## screenshield/

CLI tool that derives screen-protector overlay images from a device photo. Full spec: `screenshield/PROMPT.md`. Build proceeds in stages, stopping after each for review (see PROMPT.md "Build order"). **All 6 build-order stages are settled** as of this writing (body detection, compositing/opacity/canvas framing, cutouts, `--target screen`/`recess`, SVG export, batch mode). Nothing left in the original build order; remaining work would be follow-ups (e.g. the stroke-color/width finding below, or acting on the "how to validate" notes for template relocation / `--target recess`).

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

### Stroke width and color: solved together as one coupled system

`STROKE_WIDTH_AT_1500 = 1.317` and `STROKE_COLOR_RGB = (151,151,151)` are solved *together* from two invariants measured on `ref_1x.jpg`, not from one assumed and the other backed out — that was the original stage-2 bug (assumed width=1.5 from spec's stated value, solved color=164 from it) and it's why changing one without the other breaks the match again (see verification below).

The two invariants:
1. **Total ink deficit is invariant to sub-pixel phase.** Two independent clean axis-aligned crossings over white background (a vertical edge and a horizontal edge) both sum to 137 (`255-pixel`, summed across the full perpendicular crossing) — this equals `width * D` where `D` is the stroke's true full-coverage-per-pixel deficit, for any phase.
2. **A real pixel's deficit can never exceed `D`.** The vertical edge's peak pixel is 151 (deficit 104), identical at 3 widely-separated points along the edge (not JPEG noise). That alone proves `D ≥ 104` — already enough to rule out the old assumption (width=1.5 implies `D=91.33 < 104`, impossible).

Taking the peak as at-or-near saturation (`D=104`) and combining both invariants solves the pair at once: `W = 137/104 ≈ 1.317px`, `color = 255-104 = 151` (`#979797`) — not two separate guesses. 151 lands almost exactly on the *lower* edge of the spec's own independent eyeballed estimate (`#98`–`#B0`), where the old 164 sat mid-range.

**Verified by rendering, not just by the math:** re-rendered `test_clean_device.png` and re-sampled its own stroke crossing (a *different* image than `ref_1x.jpg`, so a different sub-pixel phase — its peak split across 2 pixels, 186/182, rather than landing on one). Total ink deficit there: 142 — close to the reference's 137 (~3.6% over), attributable to `stroke_coverage_map`'s own small known residual (~0.045-0.05px over target at supersample=32, see below), not a color or width error. Shipping color=151 with the *old* width=1.5 would have given ~161 instead — a 17% miss, not 3.6% — which is why both constants had to move together.

### Stage 3: cutouts/notches

`--notch`/`--profile`/`--pick` are the reliable, spec-required paths — notches stored as percentages of the body bbox in the profile JSON, offset math verified against real measured cutout positions (see `test_notch_pct_round_trips_reference_cutouts` — note the reference cutouts were measured on the *sheet*, not the device, so sheet-1's offset has to be subtracted before storing).

`--notches auto` has two paths: brightness heuristic (spec's original description — best-effort, known false positives, e.g. picks up printed logo text; **missed both real cutouts entirely** on the reference device since they're similar brightness to the surrounding bezel panel, not bright/saturated) and, on top of that, **template relocation**: when a profile has cutouts previously defined via `--notch`/`--pick`, a small cached image patch of each is matched locally (constrained to a small window around the profile's predicted position, not a blind scan) on the new photo, scaled by the ratio of body sizes to handle a resolution/crop difference.

**Template relocation's value is unproven on real data.** The only real-photo demo was a rigid whole-image shift (translate the whole photo a few px) — body-bbox re-detection already tracks that correctly on its own via percentage placement, so that demo didn't actually exercise the scenario relocation is *for* (a cutout moving independently of the body, e.g. manufacturing tolerance or non-rigid drift). That scenario is only validated with synthetic checkerboard-patch tests, not a real photo pair. If real repeat-model photos show relocation and plain percentage placement always agreeing, consider removing template relocation as unnecessary complexity — `--profile` alone may already be sufficient.

**How to validate:** shoot the same device model twice (or more) under normal shoot-to-shoot variation — not a synthetic/rigid shift, an actual second photo session. Run `--notches auto` (template path) and compare against plain `--notches profile` (percentage-only) on the second shot. If the two agree within a px or two every time, percentage placement alone is sufficient — remove template relocation. If they diverge (the cutout moved *relative to the body bbox*, not just the whole frame moved), that divergence is the evidence relocation is earning its keep.

### Stage 4: `--target screen` / `--target recess`, signed `--fit`

`--target recess` and `--fit` are not in `PROMPT.md` — added on request, design choices made without a reference device for either (same "best-effort, unvalidated" caveat as auto-notches):

- `--fit` replaces `--bleed`, signed, applied against the detected active region's own width/height (not the body's): **positive dilates outward** (bleed past a screen onto the bezel), **negative erodes inward** (contract to fit inside a recess). Implemented by generalizing `derive_outline`'s erosion to `offset_filled_mask()` — positive offset erodes, negative dilates, single elliptical kernel either way.
- `screen` detection: largest coherent bright-or-saturated region inside the body (spec's own description). `recess` detection: the opposite polarity — largest coherent **dark** region (a sunken feature like a camera lens reads as locally shadowed, not bright). Both via Otsu restricted to body pixels, same "let the histogram find the gap" pattern as `auto_tol`.
- **Real bug hit and fixed**: cv2's Otsu pairs with `THRESH_BINARY`'s "> thresh" bright class, so the dark class is `<= thresh`, not `< thresh`. On a hard two-level test image (e.g. exactly value 10 vs 30) Otsu placed the threshold *at* the dark class's own value, and `<` selected nothing at all. Regression-guarded by `test_target_recess_detects_dark_region_not_bright`.
- A body with no real screen/recess feature (uniform color, or a degenerate Otsu split) must not silently return the whole body or a noise speck as if it were a real detection — `find_active_region_contour` rejects candidates outside a `[0.5%, 90%]` share of the body's area and reports `target_found: False` instead.
- Circle/ellipse fitting (`is_near_circular`/`cv2.fitEllipse`, built in stage 1) was never actually exercised until stage 4 — the reference device is rectangular. Covered now via a synthetic round-watch fixture in `test_screenshield.py`.

**`--target recess`'s "largest dark region" is a design choice, not derived from real hardware, and is unvalidated against an actual recessed device.** No reference device has a recess at all, so every number and every branch of `find_active_region_contour`'s recess path (the Otsu split, the `<=` boundary, the 0.5%-90% area guard) has only been checked against synthetic hard-edged circles, never a real shadowed/sunken feature under real lighting.

There's a specific, concrete reason to doubt it before even trying: on the *actual reference device*, the two real button cutouts turned out to be similar brightness to the surrounding bezel panel — a brightness-based heuristic (`--notches auto`'s brightness path) completely missed them for exactly that reason. A physical recess (sunken camera lens, dial) is exactly the same kind of feature — small, on a dark bezel, possibly not much darker than the bezel itself, maybe *reflective* rather than shadowed depending on the lens/material and lighting angle. The recess heuristic is vulnerable to the identical failure mode already observed once on this project, just not yet checked against a real example of it.

**How to validate:** shoot an actual device with a real recessed feature (sunken camera lens, control knob, dial) under normal product-photography lighting — diffuse/even lighting, not a hard directional light that would create an artificial shadow the algorithm could get lucky on. Run `--target recess --debug` and look at the magenta (raw detected region) box in the debug image:
- If it lands on the recess: check *why* — was the recess actually distinctly darker than the bezel in that lighting, or did it luck into the area/position guards? Try the same photo with flatter lighting to see if it still holds.
- If it misses (magenta box empty/wrong, `target_found: False`, or it grabs something else) — that confirms the suspicion above. The fix is likely the same fix `--notches auto` needed: brightness alone isn't enough for a low-contrast feature on a similarly-dark bezel; something distance-from-local-background-based, or manual `--fit`-only body-mode-derived approach, would be needed instead.

### Stage 5: SVG export

`compute_layout()` is the key structural piece: it computes the canvas scale/transform and every sheet's canvas-space outline + notch-hole points *once*, and both `render_outputs` (PNG) and `render_svg` consume that same dict — this is what makes "SVG coordinates must match the PNG output exactly" (spec) true by construction rather than something that could drift between two independent implementations. `process_image` computes `layout` once and passes it to both.

Each sheet is one `<path>` with `fill-rule="evenodd"`: the outer outline as one `M...Z` subpath, each notch as its own separate `M...Z` subpath appended after it — evenodd treats every additional subpath as a hole regardless of winding direction, no need to reverse-wind hole contours. `test_svg_matches_png_layout_exactly` parses the emitted path data back out and diffs it against `layout` point-for-point (not just eyeballed). The fill/stroke paint SVG ships with is arbitrary (semi-transparent white, `#a4a4a4` stroke) — per spec this file is an Illustrator escape hatch, the user selects the sheets there and applies Graphic Style 4 for the real appearance.

**Numerically cross-checked against a real SVG renderer** (not just the coordinate-parsing test): `cairosvg` doesn't run on this Windows machine (needs the native `libcairo-2.dll`, not present), but `skia-python` ships its own bundled binary and worked. Rasterized the SVG at 1500×1500 with skia, and separately built a reference raster by calling `render_sheet` directly on a blank white canvas with `style_alpha=0` (stroke only, same code path the PNG uses) — then compared the two stroke masks via distance transform. Result: every skia-rendered stroke pixel had a same-position match in the reference raster at distance 0; the reference raster's stroke was wider (max distance 2.0px, in the direction of the *extra* pixels only). At the time, `render_sheet` still rounded stroke width to an integer pixel count (`round(1.5) = 2px`) while SVG/skia rendered the continuous 1.5px width directly — this was the first concrete evidence of the rounding bug fixed later (see "Stroke width and color" above). Confirmed visually with a 10x-magnified overlay: both traces shared the identical centerline and corner position, the grey (PNG) line was just thicker — **no positional offset, a width-rounding difference only**. `skia-python` was pip-installed only for this one-off verification — it is not a project dependency and screenshield.py does not import it. (This comparison predates the sub-pixel stroke fix and the corrected color constant; re-run it if SVG/PNG visual parity ever needs re-checking after further stroke changes.)

### Stage 6: batch mode

`process_image()` now returns `{"path", "status", "reason"}` (`status` is `"ok"`, `"skipped"`, or `"failed"`) instead of `None` — `main()` collects these across every file in a folder and prints a summary line (`N file(s): X ok, Y skipped, Z failed`, plus the filename+reason for each skipped/failed one) rather than the per-file prints being the only feedback.

- **Skip-if-exists, `--force` to override**: `outputs_exist()` checks the *specific* files this invocation would produce (`_1x.png`/`_3x.png` always, plus `_debug.png`/`.svg` only if `--debug`/`--svg` are actually passed) — re-running with `--debug` added when only the plain PNGs exist from a prior run does NOT count as already-done, since that file genuinely doesn't exist yet. `test_outputs_exist_accounts_for_debug_and_svg_flags` guards this.
- **One bad file can't abort the batch**: `main()` wraps each `process_image()` call in `try/except` — both "expected" failures (unreadable image, no device found, no target region — these return a `"failed"` status normally, no exception) and genuine unexpected crashes (caught at the `main()` level, recorded with the exception message) leave the rest of the batch untouched. Exit code is 1 if anything failed, 0 otherwise.
- **The actual workflow this enables** (verified live, not just unit-tested): run a folder of 3, one fails (blank image, no device) → 2 ok, 1 failed, exit 1. Re-run the identical command → the 2 that succeeded are `skipped` (files already exist), only the failure is retried. Fix the bad input, re-run again → the fixed one now succeeds, the other two are still skipped, exit 0. A batch of twenty only ever pays for reprocessing the ones that actually need it.

### Test suite (`screenshield/test_screenshield.py`)

36 tests, all passing, covering all 6 build-order stages: body bbox/outline accuracy against the reference (split contaminated-image tolerances, see above), `auto_tol` behavior, the opacity/alpha-compositing math and canvas framing (stage 2), notch percentage/profile round-trips + `--pick` ROI math + template relocation (stage 3), `--target screen`/`recess` + signed `--fit` including the Otsu boundary regression (stage 4), SVG/PNG coordinate parity (stage 5), and skip-if-exists/`--force`/one-failure-doesn't-abort-the-batch (stage 6). Test names are self-describing; read the file rather than this list, which will go stale faster than the code does.

Two permanent test-image fixtures beyond `ref_1x.jpg`/`ref_3x.jpg`: `screenshield/input/test_clean_device.png` (the reference device cropped to its body bbox and pasted on white — the protector's own edges fall outside that crop, so this is genuinely uncontaminated, used to verify `auto_tol`/detection without any override tolerance) and `screenshield/profiles/chigee_test.json` (a real notch profile with cached templates, built from it).

Run with `py -3 -m pytest screenshield/test_screenshield.py -v` (this machine has no bare `python`/`python3` on PATH — use the `py` launcher).
