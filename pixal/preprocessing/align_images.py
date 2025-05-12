import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold('error')

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pixal.modules import preprocessing as mod
import logging
import difflib

logger = logging.getLogger("pixal")

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)



def align_images(image_paths, save_dir, reference_dir, metric_dir, knn_ratio=0.55, npts=10, ransac_thresh=7.0, save_metrics=False, quiet=False, detect=False):
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if detect:
        images = [cv2.imread(str(p)) for p in reference_dir]
        if any(img is None for img in images):
            raise ValueError("One or more images could not be loaded.")
        prev_image = images[0]
    else:
        images = [cv2.imread(str(p)) for p in image_paths]
        if any(img is None for img in images):
            raise ValueError("One or more images could not be loaded.")
        prev_image = images[0]
        
    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = sift.detectAndCompute(prev_gray, None)
    results = []
    for i in tqdm(range(1, len(images)), desc=f"Aligning images in {save_dir.name}", disable=quiet):
        curr_image = images[i]
        if curr_image is None:
            if not quiet:
                logger.warning(f"Error loading image: {image_paths[i]}")
            continue

        height, width = prev_image.shape[:2]
        src_npts, dst_npts = mod.get_src_pts(bf, sift, knn_ratio, curr_image, prev_des, prev_kp, npts)
        homography_matrix, mask = cv2.findHomography(src_npts, dst_npts, cv2.RANSAC, ransac_thresh)

        if homography_matrix is not None:
            transformed_image = cv2.warpPerspective(curr_image, homography_matrix, (width, height))
            img_name = Path(image_paths[i]).name
            transformed_path = save_dir / f"{img_name}"
            cv2.imwrite(str(transformed_path), transformed_image)

            if not quiet:
                inliers = np.sum(mask)
                inlier_ratio = inliers / len(mask)
                score, mse = mod.alignment_score(str(image_paths[0]), str(transformed_path))
                logger.info(f"✅ {img_name} saved: {transformed_path}")
                logger.info(f"   → Alignment Score: {score:.3f}, MSE: {mse:.3f}")
                results.append({
                    "image": transformed_path,
                    "score": score,
                    "mse": mse,
                    "inliers": inliers,
                    "inlier_ratio": inlier_ratio
                })
        else:
            if not quiet:
                logger.warning(f"❌ Could not compute homography between {image_paths[0]} and {image_paths[i]}")
    if save_metrics:
        
        metric_dir = Path(metric_dir) / Path(save_dir.name)
        metric_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting metric plotting")
        mod.save_alignment_metrics_csv(results,metric_dir)
        mod.plot_alignment_metrics(results,metric_dir)
        mod.stack_intensity_heatmap(save_dir,metric_dir)
        
        metric_dir = metric_dir / "overlay_diagnostics"
        metric_dir.mkdir(parents=True, exist_ok=True)
        mod.save_overlay_diagnostics(save_dir,metric_dir)

def run(input_dir, output_dir=None, reference_dir=None, metric_dir=None, config=None, quiet=False, detect=False):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    subdirs = [p for p in input_path.iterdir() if p.is_dir()]
    if not subdirs:
        subdirs = [input_path]

    if reference_dir:
        reference_subdirs = [p for p in reference_dir.iterdir() if p.is_dir()]
        if not reference_subdirs:
            reference_subdirs = [reference_dir]
    
    knn_ratio = config.alignment.knn_ratio if config and hasattr(config, 'alignment') else 0.55
    npts = config.alignment.number_of_points if config and hasattr(config, 'alignment') else 10
    ransac_thresh = config.alignment.ransac_threshold if config and hasattr(config, 'alignment') else 7.0
    save_metrics = config.save_metrics if config and hasattr(config, 'save_metrics') else False
    
    if not reference_dir:
        for folder in subdirs:
            image_files = sorted([f for f in folder.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            if not image_files:
                if not quiet:
                    logger.warning(f"No images found in {folder}")
                continue

            sub_output = output_path / folder.name
            align_images(image_files, sub_output, reference_dir, metric_dir, knn_ratio, npts, ransac_thresh, save_metrics, quiet, detect)
            if not quiet:
                logger.info(f"📁 Completed alignment for folder: {folder.name}")
    
    else:
        for folder1, folder2 in zip(subdirs,reference_subdirs):
            image_files = sorted([f for f in folder1.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            reference_files = sorted([f for f in folder2.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]) # how to pair input to refernce?
            reference_files = mod.rearrange_by_similarity(reference_files, image_files)

            if len(reference_files) != len(image_files):
                if not quiet:
                    logger.warning(f"Number of images in {folder1} and {folder2} do not match.")
                continue
            if not image_files:
                if not quiet:
                    logger.warning(f"No images found in {folder}")
                continue

            sub_output = output_path / folder1.name
            align_images(image_files, sub_output, reference_files, metric_dir, knn_ratio, npts, ransac_thresh, save_metrics, quiet, detect)
            if not quiet:
                logger.info(f"📁 Completed alignment for folder: {folder1.name}")