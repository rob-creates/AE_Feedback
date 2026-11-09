# AE_Feedback
Feedback Effect for Adobe After Effects

## Building

Requires macOS on Apple silicon, Xcode and CMake.

1. Download the After Effects SDK from Adobe and copy it into this folder as `AfterEffectsSDK/`, so that `AfterEffectsSDK/Examples/Headers` exists. The SDK isn't included in this repository.
2. Build the plugin to `Build/Feedback.plugin`:

   ```sh
   cmake -S . -B Build
   cmake --build Build
   ```

3. Link it into After Effects 2026, using an admin account or `sudo`, then restart After Effects:

   ```sh
   scripts/install.sh
   ```

## Testing

With After Effects closed, and the build configured with `-DFEEDBACK_LOG=ON`:

```sh
tests/run_all.sh LABEL [BASELINE]
```

This rebuilds the test projects and runs a live session in After Effects, renders everything with `aerender` into `tests/out/LABEL`, and checks the results. With a `BASELINE` label, it also lists the frames that changed most since that run.
