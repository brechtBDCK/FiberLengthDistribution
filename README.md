## Fiber analysis

```bash
uv run python -m fiber_analysis analyze --input data/images --output results \
  --methods adaptive_skeleton,multiscale_ridge,line_segments \
  --config config/default.yaml --units-per-pixel 0.0 --unit-name um
```

Lengths are centerline/geodesic skeleton lengths in pixels (not fitted curves).
Images above `max_analysis_pixels` are downscaled for detection; reported pixel
lengths, widths, centroids, and endpoints are scaled back to source pixels.
Set `--units-per-pixel` only with a real calibration. `unet` safely reports
skipped until PyTorch and trained weights configured under `unet.weights` exist.
Each image/method produces two PNG graphs. Titles show accepted fiber count;
bars show each fiber's centerline length and axial orientation. Orientation is
`0° = horizontal`, `90° = vertical`, and always lies in `[0, 180)`.

Pass `--debug` to also save masks, detector responses, labels, and skeletons
as `.npy` arrays. `--units-per-pixel 0.0` means no calibration, so graph lengths
remain source-image pixels regardless of `--unit-name`.
