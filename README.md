# AE_Feedback
Feedback Effect for Adobe After Effects

## Using it

- The trail stays where the layer left it in the comp, so a moving layer leaves its trail behind.
- A 3D layer keeps its trail on the layer, which goes through the camera with it. To work on what a
  3D layer looks like through the camera, precompose it with the camera and put Feedback on the
  precomp, or put Feedback on an adjustment layer above it.
- Map Layer is read where it is in the comp. Where the map is transparent, nothing moves. Left at
  None, nothing is displaced.
- Map Layer set to the effect's own layer, at Effects & Masks, uses the trail as the map.
- The blur spreads a copy in step with its age: a small Blur Radius leaves new copies almost sharp,
  and older ones spread further each frame, with soft, long-tailed edges.
- Each group (Alpha, Blur, Levels, HSL, Transform, Displacement and Edge) has a Mask Layer. Its
  luminance at each pixel is how much of the group applies there: black leaves the trail as it was,
  white applies the whole group, and transparent parts apply none of it. A grey Transform mask moves
  a pixel part of the way; a grey Blur mask blurs it by that share of the radius. Like Map Layer, a
  mask is read where it is in the comp, and the effect's own layer at Effects & Masks uses the
  trail. Left at None, the group applies everywhere. Where an Alpha mask is dark the trail doesn't
  fade, so jumping to a frame recalculates the trail from its start, or from Pre-Roll Limit.
- Update, under Sampling, is on by default. Turned off, the layer stops adding to the trail, as if it
  were empty, while the trail goes on moving, fading and changing as usual. Keyframe it on and off to
  stamp the layer into a smoothly moving trail, in time with music for instance.
- Transform's Scale goes down to 0, which shrinks the whole trail away in a single frame.
- Pixel Sampling, under Sampling, sets how the trail is moved between pixels. Bilinear, the default,
  blends the four nearest pixels: a trail moved by fractions of a pixel (by Transform, displacement or
  Sub-steps, which divide each frame's move) softens a little with every move, like motion blur.
  Bicubic (Catmull-Rom, held to the range of the nearest pixels so it can't add fringes) keeps such a
  trail about three times crisper, and suits detail moved by Scale or Rotation. Pixel moves the trail by whole pixels and carries the fraction left over on to the next move, so it
  stays exactly as sharp as it went in while moving just as far: for pixel art and hard-edged
  graphics. At Pixel, Scale, Rotation and displacement apply once a frame, as with no sub-steps, and
  near Centre a gentle Scale can leave pixels in place, as copying whole pixels does. At reduced
  preview resolution a move is fewer pixels, so the trail steps more coarsely than in a full render.
- The trail moves on once per frame of the comp its layer is in. Nested in a comp at another frame
  rate, or stretched, it still looks as it does in its own comp at those frames, and between them it
  goes part of the way to the next. A time-reversed layer leaves its trail through the comp's earlier
  frames too.
- Jumping to a frame recalculates the whole trail before it, so the trail is always complete, though a
  long one that never fades takes a while. Turn on Limit Pre-Roll to recalculate only the last
  Pre-Roll Limit frames instead (300 by default): quicker, but a long trail can look shorter after a
  jump. Playing forward keeps the full trail either way.
- Cut-off Precision, under Sampling, sets how faint the oldest frames may get before recalculating
  leaves them out, and how far the blur's faint tails are followed: below what 8 bit, 16 bit or 32 bit
  float can show. The trail itself is always worked in float. 8 bit is quickest, and right for most
  work; a deeper setting recalculates further back, for a project at that depth.
- Work is kept to where the trail can be: a small layer's trail is worked on, and its maps read, only
  around it. Blur, large displacement and Transform can carry a trail anywhere, so with those it
  covers the whole comp, as before.

## Building

Requires macOS on Apple silicon, Xcode and CMake.

1. Download the After Effects SDK from Adobe and copy it into this folder as `AfterEffectsSDK/`, so that `AfterEffectsSDK/Examples/Headers` exists. The SDK isn't included in this repository.
2. Build the plugin to `Build/Feedback.plugin`:

   ```sh
   cmake -S . -B Build
   cmake --build Build
   ```

3. Link it into After Effects 2027, using an admin account or `sudo`, then restart After Effects:

   ```sh
   scripts/install.sh
   ```

## Testing

With After Effects closed, and the build configured with `-DFEEDBACK_LOG=ON`:

```sh
tests/run_all.sh LABEL [BASELINE]
```

This rebuilds the test projects and runs a live session in After Effects, renders everything with `aerender` into `tests/out/LABEL`, and checks the results. With a `BASELINE` label, it also lists the frames that changed most since that run.
