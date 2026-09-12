# screenshield

Drop in a device photo, get two listing images: one protector sheet, and three stacked. Run these from inside this folder.

## First time with a new device model

1. Run with `--debug` before trusting anything:

   ```
   py -3 screenshield.py photo.jpg --debug
   ```

   Open `out/photo_debug.png`. Red = device body it found, blue = the protector outline it derived. If that doesn't trace the device correctly, see "When detection looks wrong" below.

2. If the device has cutouts (camera, buttons, sensors), mark them once and save to a profile named for the model:

   ```
   py -3 screenshield.py photo.jpg --pick --profile profiles/modelname.json
   ```

   (`--pick` opens a window — drag a box over each cutout, Enter to confirm each, Esc when done.)

3. Check `out/photo_1x.png` and `out/photo_3x.png`. That's the deliverable.

## Next time you shoot the same model

One command — cutouts come from the profile automatically:

```
py -3 screenshield.py new_photo.jpg --profile profiles/modelname.json
```

## A folder of photos at once

```
py -3 screenshield.py photos/ --outdir out --profile profiles/modelname.json
```

Files that already have outputs in `out/` are skipped. If one fails, fix it and run the exact same command again — only the failure gets redone, everything else is skipped:

```
py -3 screenshield.py photos/ --outdir out --profile profiles/modelname.json
```

Force everything to redo anyway:

```
py -3 screenshield.py photos/ --outdir out --profile profiles/modelname.json --force
```

You'll get a summary line at the end either way: how many succeeded, skipped, or failed, and why.

## When detection looks wrong

Start with `--debug`, look at `out/<name>_debug.png`.

- **Watch / camera / laptop / handheld** (screen is only part of the device): add `--target screen`.
- **Sunken feature** (camera lens, dial) instead of a flat screen: `--target recess` — less proven, double-check it.
- **Missed the device entirely, or traced the wrong thing**: try `--tol 40` (or another number) to override automatic background detection.
- **Cutouts in the wrong place or missing**: redo with `--pick` (or `--notch X Y W H`, percentages of the device box), save over the same profile.
- **`--notches auto` looks wrong**: expected, it's best-effort. Define cutouts once by hand instead.
- **Outline or sheet position looks off**: `--inset`, `--offset`, `--step` control that; the defaults were tuned on one reference device and may need adjusting for something very differently shaped.

## Known trouble spots

Angled/perspective shots, a device close in color to its background, curved-edge glass, and glossy screens with reflections can all throw off detection — check the debug image when in doubt. Full detail in `CLAUDE.md`.
