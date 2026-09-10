from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, skeletonize


input_path = Path(
    "/home/bdck/PROJECTS_WSL/FiberLengthDistribution/"
    "data_webp/Glasvezels 1- 4 mm/beker_1/"
    "250429-Z50-B1-P1-F1`.webp"
)

output_dir = input_path.parent / "processed"
output_dir.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# Parameters to tune
# ------------------------------------------------------------------

BACKGROUND_RADIUS = 50
NOISE_RADIUS = 0.7

# Ignore components smaller than this many foreground pixels.
MIN_COMPONENT_AREA = 15

# Minimum centerline length for an isolated fiber.
MIN_SKELETON_LENGTH = 15

# A fiber should normally be substantially longer than it is wide.
MIN_ELONGATION = 2.5

# Large connected fiber networks should not be rejected merely because
# crossings reduce the overall component elongation.
LARGE_NETWORK_LENGTH = 100

# ------------------------------------------------------------------
# Load and preprocess
# ------------------------------------------------------------------

with Image.open(input_path) as pil_image:
    gray_pil = ImageOps.grayscale(pil_image)

gray = np.asarray(gray_pil, dtype=np.uint8)

# Very small blur: suppress pixel-scale noise without erasing fibers.
denoised_pil = gray_pil.filter(
    ImageFilter.GaussianBlur(radius=NOISE_RADIUS)
)

# Massive blur estimates the slowly varying background.
background_pil = denoised_pil.filter(
    ImageFilter.GaussianBlur(radius=BACKGROUND_RADIUS)
)

denoised = np.asarray(denoised_pil, dtype=np.uint8)
background = np.asarray(background_pil, dtype=np.uint8)


# ------------------------------------------------------------------
# One-sided background subtraction
# ------------------------------------------------------------------

# Correct for bright fibers:
#
# Positive differences are retained.
# Negative differences are clipped to zero instead of becoming a halo.
foreground = cv2.subtract(denoised, background)

# Robust contrast stretch for visualization.
low, high = np.percentile(foreground, (1.0, 99.8))

if high > low:
    foreground_display = np.clip(
        (foreground.astype(np.float32) - low) * 255.0 / (high - low),
        0,
        255,
    ).astype(np.uint8)
else:
    foreground_display = foreground.copy()


# ------------------------------------------------------------------
# Threshold
# ------------------------------------------------------------------

# Otsu produces a starting threshold automatically.
otsu_threshold, binary_u8 = cv2.threshold(
    foreground_display,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)

binary = binary_u8 > 0

# Remove only extremely small components here.
binary = remove_small_objects(
    binary,
    min_size=MIN_COMPONENT_AREA,
    connectivity=2,
)


# ------------------------------------------------------------------
# Reject isolated round specks
# ------------------------------------------------------------------

component_labels = label(binary, connectivity=2)
clean_mask = np.zeros_like(binary, dtype=bool)

for region in regionprops(component_labels):
    component = component_labels == region.label
    component_skeleton = skeletonize(component)

    skeleton_length = int(component_skeleton.sum())
    major_length = float(region.major_axis_length)
    minor_length = max(float(region.minor_axis_length), 1.0)
    elongation = major_length / minor_length

    looks_like_fiber = (
        skeleton_length >= MIN_SKELETON_LENGTH
        and elongation >= MIN_ELONGATION
    )

    # Crossing fibers can form a large component whose global aspect ratio
    # is relatively low. Preserve sufficiently large skeleton networks.
    looks_like_fiber_network = (
        skeleton_length >= LARGE_NETWORK_LENGTH
        and region.area / max(skeleton_length, 1) < 8.0
    )

    if looks_like_fiber or looks_like_fiber_network:
        clean_mask[component] = True


clean_mask_u8 = clean_mask.astype(np.uint8) * 255

# Optional skeleton preview. This will later be used for measurement.
clean_skeleton = skeletonize(clean_mask)
clean_skeleton_u8 = clean_skeleton.astype(np.uint8) * 255


# ------------------------------------------------------------------
# Diagnostic output
# ------------------------------------------------------------------

comparison = np.hstack(
    [
        gray,
        foreground_display,
        clean_mask_u8,
        clean_skeleton_u8,
    ]
)

cv2.imwrite(
    str(output_dir / f"{input_path.stem}_foreground.png"),
    foreground_display,
)

cv2.imwrite(
    str(output_dir / f"{input_path.stem}_fiber_mask.png"),
    clean_mask_u8,
)

cv2.imwrite(
    str(output_dir / f"{input_path.stem}_skeleton.png"),
    clean_skeleton_u8,
)

cv2.imwrite(
    str(output_dir / f"{input_path.stem}_comparison.png"),
    comparison,
)

print(f"Otsu threshold: {otsu_threshold:.1f}")
print(f"Saved results to: {output_dir}")
print("Comparison columns: original | foreground | clean mask | skeleton")
