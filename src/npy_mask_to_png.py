# -*- coding: utf-8 -*-
"""
Batch-convert .npy mask files to PNG images for quick visual preview.

Usage:
    python npy_mask_to_png.py [input_directory]

Default input directory: E:\\temp project code\\PIMO\\edited_epi_masks
Output: .png files with the same base name, saved in the same directory.

Dependencies: pip install numpy pillow
"""

import sys
from pathlib import Path

# Ensure this script can import sibling helpers when run from arbitrary cwd.
sys.path.append(str(Path(__file__).parent))

import argparse
import os

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


DEFAULT_INPUT_DIR = r"E:\temp project code\PIMO\edited_epi_masks"


def npy_mask_to_png(npy_path: str, png_path: str | None = None) -> bool:
    """
    Convert a single .npy mask to a PNG image.

    Args:
        npy_path: path to the .npy file
        png_path: output PNG path; if None, uses the same directory and base name

    Returns:
        True on success, False otherwise
    """
    npy_path = Path(npy_path)
    if not npy_path.exists():
        print(f"  Skipped (not found): {npy_path}")
        return False

    if png_path is None:
        png_path = npy_path.with_suffix(".png")
    else:
        png_path = Path(png_path)

    try:
        arr = np.load(str(npy_path), allow_pickle=False)
    except ValueError:
        arr = np.load(str(npy_path), allow_pickle=True)

    arr = np.asarray(arr, dtype=np.float32)

    # Support (H,W) or (H,W,1) shapes
    while arr.ndim > 2:
        arr = arr.squeeze()
    if arr.ndim != 2:
        print(f"  Skipped (not a 2D array, shape={arr.shape}): {npy_path}")
        return False

    # Normalize to 0-255: binary mask (0/1 or 0.0-1.0) -> grayscale image
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax - vmin < 1e-6:
        # Constant array
        img_uint8 = np.zeros(arr.shape, dtype=np.uint8)
    else:
        normalized = (arr - vmin) / (vmax - vmin)
        img_uint8 = (normalized * 255).astype(np.uint8)

    if Image is not None:
        img = Image.fromarray(img_uint8, mode="L")
        img.save(str(png_path))
    else:
        try:
            import cv2
            cv2.imwrite(str(png_path), img_uint8)
        except ImportError:
            print("  Error: Pillow or OpenCV is required (pip install pillow)")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch-convert .npy masks to PNG preview images"
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing .npy masks (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory; defaults to the input directory",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    if not input_dir.exists():
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)

    npy_files = sorted(input_dir.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(npy_files)} .npy files\n")

    success = 0
    for npy_path in npy_files:
        png_name = npy_path.with_suffix(".png").name
        png_path = output_dir / png_name
        if npy_mask_to_png(str(npy_path), str(png_path)):
            print(f"  [OK] {npy_path.name} -> {png_path.name}")
            success += 1
        else:
            print(f"  [SKIP] {npy_path.name}")

    print(f"\nDone: {success}/{len(npy_files)} files converted to PNG")


if __name__ == "__main__":
    main()
