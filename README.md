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
- Update Every (Frames), under Timing, moves the trail on only every so many frames, counted from the
  layer's in-point, each time by that many frames' worth, so the trail keeps its length in seconds.
  In between, the trail holds while the layer plays over it. Sub-steps divide each update.
- Jumping to a frame recalculates the whole trail before it, so the trail is always complete, though a
  long one that never fades takes a while. Turn on Limit Pre-Roll to recalculate only the last
  Pre-Roll Limit frames instead: quicker, but a long trail can look shorter after a jump. Playing
  forward keeps the full trail either way.

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
