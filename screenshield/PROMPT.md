# screenshield — build spec

Build a CLI tool called `screenshield` that automates a repetitive product-image job.

## The job

I sell screen protectors. For every device I stock I have to make two listing images by hand in Illustrator:

1. The device image with **one** translucent protector sheet floating over it, offset down-right
2. The same device, identical position and scale, with **three** sheets stacked in a staggered fan

I want to drop in a clean device image and get both outputs automatically.

## Files in this folder

- `ref_1x.jpg` — reference output, one sheet. This is the target look.
- `ref_3x.jpg` — reference output, three sheets
- `screenshield-gs.ai` — my Illustrator file containing the protector style, saved as "Graphic Style 4"

Measure the reference images. Do not guess. Where the `.ai` file and the rendered references disagree, **the rendered references win** — they are what I actually ship.

## Core principle: derive the shape, don't template it

Do **not** use a fixed template shape. The protector outline must be derived from the device in the input image, so edges, corner chamfers and radii match every device automatically.

Method:

1. Sample the background colour from the image corners, threshold to get a device mask
2. Morphological close to heal glare and highlight gaps, then take the largest external contour as the device body
3. Fill it, then **erode** by the inset distance
4. Re-contour the eroded mask and simplify — that outline is the protector

Erosion rather than a mathematical offset, because it follows any silhouette — chamfers, tapers, radii — with no per-model assumptions. On my reference device an 8-point chamfered body correctly yields an 8-point chamfered sheet.

## Two detection targets

Add `--target body|screen`, defaulting to `body`.

**`body`** — the method above. Correct when the glass covers nearly the whole front face: phones, tablets, head units, dash cams, e-readers.

**`screen`** — detects the display panel instead and derives the protector from that. Needed for watches, cameras, laptops and handhelds, where the screen is a sub-region of the body and `body` mode would produce a protector shaped like the whole device including straps or grips. Detect it as the largest coherent bright or saturated region inside the body contour, take its contour, then expand outward by a `--bleed` percentage, since protectors usually overhang the active area onto the bezel.

Both modes must handle **non-polygonal outlines**. Do not force a low vertex count when simplifying — a round watch crystal must stay round. Detect near-circular contours by comparing contour area to the area of the enclosing circle, and fit a circle or ellipse rather than polygonising.

`--debug` draws the detected target in blue in both modes, so I can tell at a glance which mode a device needs.

## Measured geometry — use as defaults

Reference device body is 1234 × 694 px. Output canvas is **exactly 1500 × 1500 px**.

| Parameter | Value |
|---|---|
| Inset from body edge | 0.57% of body width (~7 px) |
| Sheet 1 offset from device top-left | +8.5% of body W, +16.1% of body H |
| Per-sheet step for extra sheets | +2.75% of body W, +8.2% of body H |
| Stroke | ~1.5 px at 1500 px canvas |
| Canvas margin | 4% |
| Sheets in stacked version | 3 |

Two layout rules that are easy to miss:

- The canvas is framed on the **3-sheet extent** in *both* outputs, so the device lands in the identical position and scale in each file. Do not frame each output independently.
- Sheet ordering matters for opacity assignment (below), not just for drawing.

## Opacity — this is the part to get exactly right

Graphic Style 4's fill renders as **50% white**. It is flat: I sampled alpha along the black bezel from x=340 to x=1220 in `ref_1x.jpg` and it is 0.502 at every point with no falloff along either axis.

The `.ai` file describes the fill as a 135° linear gradient named "Unnamed gradient 386". Its stops must be near-identical, because it renders flat. If you find the stops differ meaningfully, tell me — but match the rendered reference regardless.

Layer opacity multiplies on top of the 50% style fill:

**1-sheet output** — one sheet at 100% layer opacity → effective alpha **0.50**

**3-sheet output:**

| Sheet | Layer opacity | Effective alpha |
|---|---|---|
| Nearest the device (smallest offset) | 50% | 0.25 |
| Middle | 50% | 0.25 |
| Furthest from device (largest offset, frontmost) | 80% | 0.40 |

Keep **layer opacity and style-fill opacity as separate parameters**, not one baked-in number. I may change layer opacities per product line.

### Required test

Sampling `ref_3x.jpg` over the black bezel in the regions where each number of sheets overlaps gives:

| Region | Alpha |
|---|---|
| Sheet 1 alone | 0.251 |
| Sheets 1+2 | 0.439 |
| All three | 0.663 |
| `ref_1x.jpg`, single sheet | 0.502 |

Write a test that reproduces all four within 0.005. A compositing bug then shows up immediately instead of as a vague "looks slightly off" two stages later.

## Stroke

The `.ai` gives the stroke as solid CMYK 54.6 / 46.1 / 45.7 / 11.1 at 1pt, butt cap, miter join. That converts naively to roughly `#677A7B`, but the stroke in `ref_1x.jpg` measures around `#98`–`#B0` grey — anti-aliasing on a thin line accounts for the difference.

Sample the actual stroke pixels in the reference and use the measured value for raster output. The CMYK value is correct only for the SVG and Illustrator path.

## Cutouts / notches

Devices have sensor windows, LEDs and buttons the protector has holes for. My reference device has two small rectangles on the right edge.

Fully automatic detection is not reliable — a naive approach finds 5 candidates on my reference where there are 2, because on-screen UI elements look like bezel features. Support three routes:

- `--notch X% Y% W% H%` — repeatable, as percentages of the device bounding box, written to a profile JSON so the next shot of the same model is zero-input
- `--pick` — OpenCV `selectROIs` window, drag a box over each cutout, saves to the same profile
- `--notches auto` — best-effort: features inside the body that aren't bezel-dark, with the largest non-bezel region (the display) plus a collar removed first, then filtered by area and constrained to the bezel band. Print a warning that it needs checking.

Default to reading a profile if one is given, otherwise no cutouts. Store profiles as relative percentages so they survive any resolution or crop.

## CLI

```
screenshield <image-or-folder> [options]

--outdir DIR          default: out
--target body|screen  default: body
--bleed PCT           screen mode only: outward expansion past the active area
--profile FILE        notch profile JSON for this device model
--notch X Y W H       cutout as % of body box, repeatable
--pick                click cutouts in a GUI window
--notches MODE        profile | auto | none
--size N              canvas px, default 1500
--copies N            sheets in the stacked version, default 3
--inset PCT           % of body width
--offset X Y          sheet 1 offset, % of body W/H
--step X Y            per-sheet step, % of body W/H
--style-alpha F       style fill opacity, default 0.50
--layer-alpha ...     per-sheet layer opacity, default 1.0 single / 0.5 0.5 0.8 stacked
--tol N               background detection tolerance, default 8
--svg                 also export vector outlines
--debug               dump detection overlay
```

Outputs `<stem>_1x.png` and `<stem>_3x.png`, both 1500×1500. Folder input processes every image in it.

`--debug` writes `<stem>_debug.png` with the detected body in red, protector outline in blue, cutouts in green. I need this to sanity-check a new model before trusting a batch.

`--svg` writes one SVG with two named layers, `protector_1x` and `protector_3x`, each holding correctly-placed compound paths with cutouts as even-odd holes. This is my Illustrator escape hatch: I open it, select the sheets, and apply Graphic Style 4 to keep the real appearance. SVG coordinates must match the PNG output exactly.

## Stack

Python 3, `opencv-python` and `numpy` only. No other dependencies. Single file is fine if it stays readable; split it if it doesn't.

## Build order

Stop after each stage so I can check before you continue.

1. `body` detection and outline derivation, with `--debug` working, verified against `ref_1x.jpg`
2. Compositing, opacity model and canvas framing — then run the four-number alpha test and show me the results
3. Cutouts: manual and profile first, auto last
4. `--target screen` with circle/ellipse fitting
5. SVG export
6. Batch mode

## Acceptance

Run the tool on a device crop taken from `ref_1x.jpg` and show me the result beside the original. Body dimensions, sheet offsets and canvas framing should land within a few pixels, and the four alpha numbers within 0.005. Tell me where they don't rather than adjusting the constants to hide it.

Then tell me honestly which cases will break: angled and perspective shots, coloured devices on coloured backgrounds, curved-edge phones where the protector must be smaller than the glass, and glossy screens where reflections fragment screen detection.
