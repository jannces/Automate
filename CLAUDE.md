# Automate

## screenshield/

CLI tool that derives screen-protector overlay images from a device photo. Full spec: `screenshield/PROMPT.md`. Build proceeds in stages, stopping after each for review (see PROMPT.md "Build order"). **All 6 build-order stages are settled** as of this writing (body detection, compositing/opacity/canvas framing, cutouts, `--target screen`/`recess`, SVG export, batch mode). Nothing left in the original build order. The follow-up design questions are also settled: the peel fold's gradient placement, the chamfer shrinking with the inset, the reference-calibrated canvas framing, and uniform sheet steps (each marked **SETTLED** below). A proposed `--illustrator` mode was **declined by the user — don't build it**. The only open work is acting on the "how to validate" notes for template relocation / `--target recess`, both of which need real photos that don't exist yet.

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

### Parametric primitive outline (default, `--shape primitive`)

The protector is generated from fitted parameters, not the traced contour: rotated rectangle `(cx, cy, w, h, theta)` plus ONE shared corner — `chamfer` (45° legs) or `round` (radius). `fit_best_primitive` does a Huber-IRLS Levenberg-Marquardt fit of signed distance to the body contour (`CHAIN_APPROX_NONE`, pixel centres +0.5, then grown 0.5px so edges sit on true pixel boundaries), tries both kinds, keeps the lower cost. The inset is applied analytically by `offset_primitive` (sides move by inset_x/inset_y; corner re-solved from the elliptical kernel's support at 45°) — verified equal within 1px to fitting a mask actually eroded with the same kernel.
- Rotation that moves the far end of the longest side <1px is snapped to 0 and refit — the reference device's bottom/left/right edges are exactly axis-aligned, only its top edge drifts ~1px, which alone produced a spurious 0.027° fit.
- If >2% of contour points are >2px off the best primitive, it falls back to the traced path (`derive_outline`'s old erode/approxPolyDP/line-snap) and prints a WARNING. Circles/ellipses keep the ellipse fit.
- On the reference device: chamfer beats round clearly (0.2-0.4% vs 3.4% outliers). Per-corner independent fits show the device's own chamfers are 45.6/41.5/43.2/43.8px and the reference SHEET's are 39.5 (BR)/41.7 (TR)/44.4 (BL) — not identical in either.
- **SETTLED (user): the chamfer shrinks with the inset, the way Illustrator's Offset Path does.** The reference sheet's unequal corners are hand-drawn variance, not a spec to reproduce — the primitive gives one shared 39.0px on all four (body ~44 minus the anisotropic inset shrink). `offset_primitive` already implemented exactly this; confirmed empirically against Illustrator's own `Object > Path > Offset Path` (miter join) on an 800×500 octagon with 60px legs: offsets 7.5/10/20 give legs 55.6064/54.1421/48.2842, and `k + d(√2 − 2)` predicts 55.6066/54.1421/48.2843. Our anisotropic form (`k + hypot(dx,dy) − dx − dy`, the elliptical support at 45°) reduces to that identically when dx = dy. Offset Path can't express an anisotropic inset at all, which is why the generalisation is ours. Guarded by `test_chamfer_shrinks_with_the_inset_like_offset_path`.
- **The chamfer now feeds the fold position, so the two are coupled.** Gradient length is `(W + H − k_TL − k_BR)/√2`, so the chamfer decision moves the peel's fold: keeping the body's 44px instead of the shrunk 39px shifts the fold 2.26 along x+y (1.6px across the sheet — further than the fold's own 1.1px width, so visible). Don't revisit one of these without the other. Guarded by `test_chamfer_shrink_feeds_the_fold_position`.
- `render_sheet` rasterises fill with supersampled area coverage at sub-pixel vertices (`fill_coverage_map`); it used to truncate vertices to int for both fill and stroke.

### Canvas framing — calibrated to the references, and deliberately so

**SETTLED (user): framing is a deliberate composition choice, not an artifact.** The user framed the references by hand because that is the layout they want, and new output has to drop into the existing catalogue with no visible scale or position shift. So unlike the gradient placement — where the reference and a correct fresh apply turned out to agree — here the references simply win over PROMPT.md's "4% margin, centred", which does not reproduce them.

Derived from **both** references, and this is the rule now in `compute_canvas_transform`:

```
CANVAS_MARGIN_PCT = 3.60448     # a side, on the LONG axis of the 3-sheet extent
CANVAS_SHIFT_PCT  = (0.46219, 1.66042)   # then nudged down-right off centre, % of canvas
```

- **Calibration criterion:** re-feeding a reference render must be the identity — scale 1.0, tx = ty = 0. The references *are* 1500px canvases with the device at a known place, so this is exact rather than a fit. It gives the long axis filling 92.791% of the canvas, and margins L/R 61.0/47.1, T/B 325.0/275.2.
- **What each reference contributes.** `ref_1x` alone fixes the scale and position. `ref_3x` is what proves the *basis* is the 3-sheet extent and not the 1-sheet one: it carries far more content (ink bbox to 1454,1226 vs 1385,1112) yet places the device on the very same pixel (60,325 in both). Framing per-output would have moved it.
- **Verified end to end:** the clean device through the real pipeline lands at (61.00, 325.02)–(1294.94, 1018.99) against the reference's (61,325)–(1295,1019) — 0.06px, scale error 0.005%. `test_framing_reproduces_the_reference_layout`.
- The shift is a composition offset, not slop: it is smaller than the margin on both axes, so content can never leave the canvas even when both axes bind (`test_canvas_shift_can_never_push_content_off_canvas`).
- Content extent is a fixed multiple of the body bbox, so the rule transfers to any device: `1.12793 × body_w` by `1.29656 × body_h`.

**SETTLED (user): sheet steps stay uniform.** `ref_3x`'s own steps are uneven — 1→2 (+33.5, +58.5), 2→3 (+35.0, +55.2) — but that is hand-placement jitter, not intent. Default `--step` (2.75%, 8.2%) = (33.9, 56.9) is their average within 0.3px. Independent corroboration that the sheets are plain copies: each sheet's fold tracks it exactly, the three folds sitting at x+y = 1140.5 / 1232.5 / 1322.9, spacings 92.0 and 90.4 against the measured step sums 92.0 and 90.2. Guarded by `test_sheet_steps_are_uniform`.

A clean device image can be recovered from ref_1x by dividing out the sheet with the reference's own gradient placement (`t = 0.0005425*((1385+1112)-(x+y)) - 0.0569`, pixel-index coords) and inpainting stroke + step line; bezel comes back to level 1.

### Line-fit-snap (trace fallback only)

`snap_polygon_to_lines()`: after `approxPolyDP` simplifies the eroded contour to a polygon, each edge gets a least-squares line fit (`cv2.fitLine`) to the raw dense contour points along it, and each vertex is recomputed as the intersection of its two adjacent fitted lines. This recovers the crisp corner that erosion's rounding destroys.

The corner-exclusion margin is a **fixed Euclidean distance** (`corner_margin_px = 1.5 × max(inset_x_px, inset_y_px)`), not a fraction of segment point count — corner rounding radius is a function of the erosion kernel, not of how long an edge happens to be. A short chamfer between two long sides can be almost entirely inside the fillet radius; trimming by point-count fraction under-trimmed it and pulled fitted corners several px off. Edges with too few points left after trimming keep their original `approxPolyDP` vertex rather than fit to corner-contaminated noise.

### Graphic Style 4 — extracted from the .ai and confirmed by Illustrator itself

`screenshield-gs.ai` is PDF-compatible; the native data is 8 `/AIPrivateDataN` streams joined, after a `%AI24_ZStandard_Data` header, **zstd**-compressed (not zlib). Decompressing needs the `zstandard` pip package (installed to a throwaway dir for this, not a project dependency). Illustrator 2022 (26.0.1) is installed and scriptable via COM (`Illustrator.Application.26`, `DoJavaScriptFile`); probe results below came from running ExtendScript against a *copy* of the .ai, never the original.

**The gradient IS the peel effect — this is the design intent, stated by the user.** The 135° gradient's hard stop is the *fold line* of the lifted top-left corner: flat sheet at 50%, thinning to 40% as it approaches the fold, then a hard step up to a 60% lifted flap. It is not a subtle sheen, and the fold's **position and sharpness are the visible design feature** — treat them as load-bearing, not as incidental shading. One fold per sheet is plainly visible in both references (three parallel ones in `ref_3x`).

**Fill = "Unnamed gradient 386"**, linear, 135°, white at all three stops. Re-extracted verbatim from the .ai and unchanged — stops as `(ramp, opacity, midpoint)`: `(56.367808%, 50%, 50)`, `(67.817290%, 40%, 50)`, `(67.903610%, 60%, 13)`.
- **The two stops at the hard stop are NOT at identical positions** — they sit `0.086320` percentage points apart, i.e. 0.000863 of the gradient length. On a reference-sized sheet (length ≈1284) that is a **1.11px ramp measured across the sheet**, perpendicular to the fold. So the fold is a one-pixel ramp by design, not a mathematical discontinuity; anything much wider is wrong, and so is forcing it to zero width.
- **Midpoint** belongs to the segment from that stop to the *next* stop (verified by render), so the 13 on the last stop does nothing. The curve is `s^(ln .5/ln m)` for m≥0.5, mirrored for m<0.5 (fit to renders at 13/50/80).
- **Placement when the style is applied — fitted to the shape's bounds in the GRADIENT'S OWN ROTATED FRAME, not to its axis-aligned bounding box.** Length = the shape's extent along the 135° axis; centred on that rotated-frame bbox; origin then shifted `0.047488 × length` toward the end stop. At 135° the axis points straight at the top-left and bottom-right corners, so **chamfering those corners shortens the gradient**: length = `(W + H − k_TL − k_BR)/√2`. Only those two corners matter; the other two do not affect it at all.

  Measured against Illustrator on a 1219×675 bbox: plain rectangle → 1339.260, 40px-chamfer octagon → 1282.692, 150px → 1127.128, asymmetric (k_TL=30, k_BR=50) → 1282.692. `gradient_vector` reproduces all of these to 1e-6, and their origins to within Illustrator's whole-unit rounding. The perpendicular component of the origin also comes from the rotated-frame centre, not the bbox centre (bbox-centre is off by 3.5px on the asymmetric octagon; rotated-frame centre by 0.6px, within rounding).

- **CORRECTION — the earlier "`ref_1x.jpg` does not use fresh-apply placement" claim is refuted.** It *is* fresh-apply; the ~20px discrepancy was a bug in our model, which fitted the gradient to the **bounding box**. The sheet is chamfered, so the bbox overstates the gradient length by ~4%, which drags the fold ~19 along x+y (~13px across the sheet).

  With the corrected rule the fold lands at x+y = **1139.5** against the reference's measured **1140.3–1140.5**; the old bbox rule put it at 1121.8. The reference fold was measured two independent ways that agree: mean brightness drop per x+y diagonal over the device, and extrapolating the 50%→40% ramp off the bottom bezel (6051 px, rms 0.0024, which independently gives length 1286.6 vs the rule's 1284.0). There is no hand-placement and no unrefit copy — **do not "correct" the renderer back toward the reference with an offset constant.**

- **Why the earlier check missed it:** the validating test used only **plain rectangles**, and for an unrotated rectangle the bounding box and the extent along the 135° axis are *the same number*. Rectangles cannot distinguish the two rules. `illustrator_gs4_octagon.png` (asymmetric chamfered octagon, same bbox, exported from the same RGB document, stroke removed) is the fixture that can, and `test_gradient_fits_shape_extent_not_bounding_box` pins the rule directly.

- **The fold renders sharp; the "3px strip" is Illustrator's own antialiasing, not a blur we introduced.** In Illustrator's export the step spans 3 diagonals (s=605→607); ours point-samples the gradient at pixel centres and resolves it in **2**, i.e. marginally sharper, with the midpoint within 0.31 of Illustrator's. Away from those diagonals the two renders agree to ≤1.5 levels per pixel. Because the gradient is constant along x+y, every pixel on a diagonal gets the identical value, so the fold is a perfectly straight 45° edge with no staircase. Guarded by `test_fold_renders_as_a_hard_edge_not_a_blur`, which asserts ours is never *wider* than Illustrator's. Do not add supersampling to the gradient sampling — it would soften the fold for no gain.

- The template path in the .ai (Illustrator-reported gradient length 1602.27 for a bbox projecting to 1664.75) fits the same rule: it is a chamfered shape, and 1602.27 is exactly what `(W + H − k_TL − k_BR)/√2` gives for ~44px chamfers. Not an unrefit gradient either.
- **CMYK vs RGB document matters.** The .ai is CMYK. Illustrator's PNG export from it renders 50% white over black as 0.586, not 0.500 (dot-gain in the profile conversion). After converting the document to RGB it renders exactly 0.502/0.600, matching the references. So the references were not exported from a CMYK document.

**Stroke = 1pt CMYK 54.6/46.1/45.7/11.1, butt cap, miter join** → `#787878` at 1px on a 1500px canvas. **Not inferred from pixels — read off the file twice over:** the .ai declares `1 w` (and `0 J 0 j`), and Illustrator's DOM reports `strokeWidth = 1` with `strokeColor` RGB (120,120,120) directly, in the RGB document the references were exported from.

The earlier constants (1.317px, `#979797`) were solved by assuming the darkest stroke pixel was saturated. Refuted — but note *why* a straight edge can't settle it: total ink across a crossing is `width × (255 − colour)` and is **phase-independent**, so a straight edge reveals only that product. (1.317, 151) and (1.0, 120) give ≈the same product, hence look identical there.

**Solved independently from each source; they agree, and the export wins:**

| source | straight-edge ink | can it separate w from c? | result |
|---|---|---|---|
| Illustrator PNG export | **135.0** exactly, one pixel at 120 | yes — half-pixel placement fully covers one pixel, so that pixel *is* the colour | w = 1.000, c = 120 |
| `ref_1x.jpg` | **137** core (JPEG tails inflate it to 142-145) | only via diagonal extremes, which JPEG corrupts | w ≈ 1.015 with c = 120 |

They agree to ~1.5%, and the export is much more trustworthy:

- `ref_1x` is **JPEG**. The 142-143 figure previously recorded here counted ringing tails (`252/254/253` either side of the real stroke pixels); the two genuine pixels total 137. Worse, `ref_1x`'s chamfers contain pixels at **116 — darker than the stroke colour 120, which is physically impossible** for a #787878 stroke over white. JPEG undershoot is distorting precisely the statistic the (w, c) split depends on. It still refutes `#979797` decisively (151 vs an observed 116 is far beyond JPEG noise), but it cannot calibrate.
- The export is lossless, at geometry we chose (so the sub-pixel phase is known and coverage is computable per pixel), from the same renderer and the same graphic style — the references minus the lossy transport.

**The renderer's ~141 was a bug, not a calibration.** `stroke_coverage_map` supersampled `cv2.polylines`, and cv2 renders a thickness-`t` line as `2·floor((t+1)/2) + 1` pixels — **always odd**. At supersample 32 a 1.0px target could only land on 0.96875 or 1.03125; it drew 1.031, plus `LINE_AA` softening on top (unnecessary — the area-downsample already antialiases), giving an effective **1.055px** and ~142 ink. Raising the supersample only halves the gap and costs memory quadratically: the approach could not hit the target even in principle.

Replaced with **analytic coverage** — exact distance from each pixel centre to each segment, then the exact 1-D overlap of the band with the pixel. Total ink is now exactly `width` at every width and sub-pixel phase (tested to 1e-9), straight edges are **pixel-identical to Illustrator's export**, and it dropped the suite from ~36s to ~23s with no supersampled buffers. Don't reintroduce supersampling here.

- **Diagonals are deliberately not matched.** Illustrator's own 45° antialiasing is heavier than true area coverage: it saturates the centre pixel *and* puts 51/135 either side — 237 ink per row where the geometry says `√2 × 135 = 191`. Ours lands at 214, between the two (max 11 levels apart, on the chamfers only; mean whole-image difference 0.014 levels). That gap is Illustrator's rasteriser, not the artwork, so it is recorded rather than papered over with an angle-dependent fudge. Joins are round (distance-to-segment) against Illustrator's miter — sub-pixel at 1px.

(Graphic Style 3 is identical except for a 1.5pt stroke; the template path on the artboard uses Style 3, not 4.)

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

**Numerically cross-checked against a real SVG renderer** (not just the coordinate-parsing test): `cairosvg` doesn't run on this Windows machine (needs the native `libcairo-2.dll`, not present), but `skia-python` ships its own bundled binary and worked. Rasterized the SVG at 1500×1500 with skia, and separately built a reference raster by calling `render_sheet` directly on a blank white canvas with `style_alpha=0` (stroke only, same code path the PNG uses) — then compared the two stroke masks via distance transform. Result: every skia-rendered stroke pixel had a same-position match in the reference raster at distance 0; the reference raster's stroke was wider (max distance 2.0px, in the direction of the *extra* pixels only). At the time, `render_sheet` still rounded stroke width to an integer pixel count (`round(1.5) = 2px`) while SVG/skia rendered the continuous 1.5px width directly — this was the first concrete evidence of the rounding bug fixed later (see "Stroke width and color" above). Confirmed visually with a 10x-magnified overlay: both traces shared the identical centerline and corner position, the grey (PNG) line was just thicker — **no positional offset, a width-rounding difference only**. `skia-python` was pip-installed only for this one-off verification — it is not a project dependency and screenshield.py does not import it. (This comparison predates BOTH the corrected colour constant and the analytic-coverage rewrite that replaced supersampling entirely; the width difference it saw is gone. Re-run it if SVG/PNG visual parity ever needs re-checking after further stroke changes.)

### Stage 6: batch mode

`process_image()` now returns `{"path", "status", "reason"}` (`status` is `"ok"`, `"skipped"`, or `"failed"`) instead of `None` — `main()` collects these across every file in a folder and prints a summary line (`N file(s): X ok, Y skipped, Z failed`, plus the filename+reason for each skipped/failed one) rather than the per-file prints being the only feedback.

- **Skip-if-exists, `--force` to override**: `outputs_exist()` checks the *specific* files this invocation would produce (`_1x.png`/`_3x.png` always, plus `_debug.png`/`.svg` only if `--debug`/`--svg` are actually passed) — re-running with `--debug` added when only the plain PNGs exist from a prior run does NOT count as already-done, since that file genuinely doesn't exist yet. `test_outputs_exist_accounts_for_debug_and_svg_flags` guards this.
- **One bad file can't abort the batch**: `main()` wraps each `process_image()` call in `try/except` — both "expected" failures (unreadable image, no device found, no target region — these return a `"failed"` status normally, no exception) and genuine unexpected crashes (caught at the `main()` level, recorded with the exception message) leave the rest of the batch untouched. Exit code is 1 if anything failed, 0 otherwise.
- **The actual workflow this enables** (verified live, not just unit-tested): run a folder of 3, one fails (blank image, no device) → 2 ok, 1 failed, exit 1. Re-run the identical command → the 2 that succeeded are `skipped` (files already exist), only the failure is retried. Fix the bad input, re-run again → the fixed one now succeeds, the other two are still skipped, exit 0. A batch of twenty only ever pays for reprocessing the ones that actually need it.

### Test suite (`screenshield/test_screenshield.py`)

56 tests, all passing, covering all 6 build-order stages plus the parametric primitive fit and Graphic Style 4's gradient/stroke (against Illustrator's own exports and the references), including the peel fold's placement rule and its sharpness: body bbox/outline accuracy against the reference (split contaminated-image tolerances, see above), `auto_tol` behavior, the opacity/alpha-compositing math and the reference-calibrated canvas framing (stage 2), notch percentage/profile round-trips + `--pick` ROI math + template relocation (stage 3), `--target screen`/`recess` + signed `--fit` including the Otsu boundary regression (stage 4), SVG/PNG coordinate parity (stage 5), and skip-if-exists/`--force`/one-failure-doesn't-abort-the-batch (stage 6). Test names are self-describing; read the file rather than this list, which will go stale faster than the code does.

Permanent test-image fixtures beyond `ref_1x.jpg`/`ref_3x.jpg`: `screenshield/illustrator_gs4_render.png`, `screenshield/illustrator_gs4_octagon.png` and `screenshield/illustrator_gs4_stroke.png` (Illustrator's own PNG exports of Graphic Style 4 — on a plain rectangle, on an asymmetric chamfered octagon of the same bbox, and stroke-only over white. The octagon pins the gradient-fitting rule and the fold's sharpness; the stroke export pins stroke width and colour), `screenshield/input/test_clean_device.png` (the reference device cropped to its body bbox and pasted on white — the protector's own edges fall outside that crop, so this is genuinely uncontaminated, used to verify `auto_tol`/detection without any override tolerance) and `screenshield/profiles/chigee_test.json` (a real notch profile with cached templates, built from it).

Run with `py -3 -m pytest screenshield/test_screenshield.py -v` (this machine has no bare `python`/`python3` on PATH — use the `py` launcher).

### Known limitations — what will break, and why

Per spec's "Acceptance" section, an honest rundown, mechanism-level not just a list:

- **Angled and perspective shots.** Every geometric constant in this tool (`--inset`, `--offset`, `--step`, `--fit`) is a percentage of the body's pixel width/height, computed on the assumption that the photo is a fronto-parallel (flat, non-perspective) view — "7.5px inset" means the same physical distance everywhere only if the device's true edges map to straight, undistorted lines in the image. Under perspective, edges converge toward a vanishing point; `line-fit-snap`'s least-squares fit will still happily fit a straight line to a slightly-curved/converging edge, just the *wrong* line, silently. Inset/offset math has no way to detect or correct for this — expect systematic corner and edge-position errors that scale with the perspective angle, with no error or warning.

- **Colored devices on colored backgrounds.** `device_mask`/`auto_tol` are built entirely around one background color sampled from the four image corners, separated from the device by per-channel deviation thresholding. This is the exact same mechanism that already failed once on this project (the two real button cutouts on the reference device were missed by the brightness-based `--notches auto` heuristic because they were too close in tone to their surroundings) — a device whose color is close to its background (white phone on cream backdrop, black device on dark grey seamless) hits the identical failure mode: `auto_tol`'s Otsu step won't find a confident valley, it falls back to `FIXED_DEFAULT_TOL=30`, and if the true contrast is below that, detection silently misses part or all of the device rather than erroring out.

- **Curved-edge (2.5D/fully curved) phones needing a smaller protector than the glass.** The whole method (spec's own "derive the shape, don't template it") erodes the *outer silhouette* by a fixed inset. A curved-glass edge needs the protector to stop well short of where the glass starts curving down — but that boundary is a property of the glass's 3D shape, invisible in the 2D silhouette the erosion operates on. There's no signal in a body/screen photo that says "this is where flat becomes curved"; a uniform inset can't target it. Expect the derived protector to overhang the curve and not sit flush.

- **Glossy screens where reflections fragment detection.** `find_active_region_contour`'s screen mode takes the *largest* coherent bright/saturated connected region as the screen. Specular glare under studio lighting can blow out patches to near-white and simultaneously darken reflected-object patches, splitting the true screen into two or more disconnected lobes — `MORPH_CLOSE` only heals small gaps, not a reflection streak crossing the middle of a display. `largest_external_contour` then picks whichever lobe happens to be bigger, not the true screen extent — the derived protector will be sized and shaped for a fragment of the display, not the whole thing.

- **`--target recess` — unvalidated against any real recessed device.** No reference device has a recess at all; every branch of its "largest dark region" heuristic has only been checked against synthetic hard-edged circles. There's a specific, concrete reason to doubt it, not just an absence of testing: the same brightness-based approach already failed on this project's own real low-contrast cutouts. See the dedicated section above ("`--target recess`'s 'largest dark region'...") for the full reasoning and exactly what a real test photo would need to show to settle it either way.

- **Template relocation (`--notches auto` with a cached profile) — unvalidated against any real repeat photo pair.** The only real-photo test was a rigid whole-image shift, which plain percentage placement already handles on its own — it didn't exercise the actual scenario relocation exists for (a cutout moving independently of the body, e.g. manufacturing tolerance). See "Template relocation's value is unproven on real data" above for what a real validation shoot would need to show.

None of these fail loudly — they degrade silently into a wrong-shaped or wrong-positioned protector. `--debug` (red body / magenta raw target region / blue final outline / green cutouts) is the only real check; always look at it on a device model before trusting a batch run against that model.
