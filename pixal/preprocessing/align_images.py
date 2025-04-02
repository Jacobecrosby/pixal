import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from pixal.preprocessing.modules import preproc_module as mod

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

def align_images(image_paths, save_dir, knn_ratio=0.55, npts=10, ransac_thresh=7.0, quiet=False):
    save_dir.mkdir(parents=True, exist_ok=True)
    images = [cv2.imread(str(p)) for p in image_paths]
    if any(img is None for img in images):
        raise ValueError("One or more images could not be loaded.")

    prev_image = images[0]
    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = sift.detectAndCompute(prev_gray, None)

    for i in tqdm(range(1, len(images)), desc="Aligning images", disable=quiet):
        curr_image = images[i]
        if curr_image is None:
            if not quiet:
                print(f"Error loading image: {image_paths[i]}")
            continue

        height, width = prev_image.shape[:2]
        src_npts, dst_npts = mod.get_src_pts(bf, sift, knn_ratio, curr_image, prev_des, prev_kp, npts)

        homography_matrix, mask = cv2.findHomography(src_npts, dst_npts, cv2.RANSAC, ransac_thresh)

        if homography_matrix is not None:
            transformed_image = cv2.warpPerspective(curr_image, homography_matrix, (width, height))
            img_name = Path(image_paths[i]).name
            transformed_path = save_dir / f"aligned_{img_name}"
            cv2.imwrite(str(transformed_path), transformed_image)

            if not quiet:
                score, mse = mod.alignment_score(str(image_paths[0]), str(transformed_path))
                print(f"\n✅ {img_name} saved: {transformed_path}")
                print(f"   → Alignment Score: {score:.3f}, MSE: {mse:.3f}")
        else:
            if not quiet:
                print(f"\n❌ Could not compute homography between {image_paths[0]} and {image_paths[i]}")

def run(input_dir, output_dir=None, config=None, quiet=False):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    image_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    if not quiet:
        print(f"🔍 Aligning {len(image_files)} images...")

    # Load parameters from config or set defaults
    knn_ratio = config.alignment.knn_ratio if config and hasattr(config, 'alignment') else 0.55
    npts = config.alignment.number_of_points if config and hasattr(config, 'alignment') else 10
    ransac_thresh = config.alignment.ransac_thresh if config and hasattr(config, 'alignment') else 7.0

    align_images(image_files, output_dir, knn_ratio, npts, ransac_thresh, quiet)
