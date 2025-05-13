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



def align_images(image_paths, save_dir, reference_dir, metric_dir, knn_ratio=0.55, npts=10, ransac_thresh=7.0, save_metrics=False, quiet=False, detect=False,MIN_SCORE_THRESHOLD=0.5, MAX_MSE_THRESHOLD=10.0, MIN_GOOD_MATCHES=20):
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if detect:
    
        images = [cv2.imread(str(p)) for p in image_paths]
        ref_images = [cv2.imread(str(q)) for q in reference_dir]
        if any(img is None for img in images):
            raise ValueError("One or more images could not be loaded.")
        prev_image = ref_images[0]
       
    else:
        images = [cv2.imread(str(p)) for p in image_paths]
        if any(img is None for img in images):
            raise ValueError("One or more images could not be loaded.")
        prev_image = images[0]
        
    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = sift.detectAndCompute(prev_gray, None)
    results = []

    # sets starting image. m = 1 if detect is False, m = 0 if detect is True
    m = 1 if not detect else 0

    if len(images) <= m:
        if not quiet:
            logger.warning(f"Not enough images to align (found {len(images)}). Skipping alignment.")
        return  # or continue, depending on context

    for i in tqdm(range(m, len(images)), desc=f"Aligning images in {save_dir.name}", disable=quiet):
        curr_image = images[i]
        if curr_image is None:
            if not quiet:
                logger.warning(f"Error loading image: {image_paths[i]}")
            continue

        best_alignment = None
        best_metrics = {"score": -1, "mse": float("inf")}
        best_path = None

        for j, ref_image in enumerate(ref_images):
            prev_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
            prev_kp, prev_des = sift.detectAndCompute(prev_gray, None)

            result = mod.get_src_pts(bf, sift, knn_ratio, curr_image, prev_des, prev_kp, npts, logger)
            if result is None:
                continue

            src_npts, dst_npts = result
            homography_matrix, mask = cv2.findHomography(src_npts, dst_npts, cv2.RANSAC, ransac_thresh)
            if homography_matrix is not None:
                        inliers = np.sum(mask)
                        logger.info(f"   → Found {inliers} inliers for reference image")
                        if inliers < MIN_GOOD_MATCHES:
                            continue  # Skip this reference if too few inliers

                        height, width = ref_image.shape[:2]
                        transformed_image = cv2.warpPerspective(curr_image, homography_matrix, (width, height))
                        img_name = Path(image_paths[i]).name.replace("no_bg", "aligned")
                        transformed_path = save_dir / img_name
                        cv2.imwrite(str(transformed_path), transformed_image)

                        score, mse = mod.alignment_score(str(image_paths[0]), str(transformed_path))

                        if not quiet:
                            logger.info(f"   → Inliers: {inliers}, Score: {score:.3f}, MSE: {mse:.3f}")

                        # Early exit if a good-enough match is found
                        if score >= MIN_SCORE_THRESHOLD and mse <= MAX_MSE_THRESHOLD:
                            best_alignment = transformed_image
                            best_metrics = {"score": score, "mse": mse}
                            best_path = transformed_path
                            break  # No need to test more references

                        # Track best fallback candidate
                        if score > best_metrics["score"] and mse < best_metrics["mse"]:
                            best_alignment = transformed_image
                            best_metrics = {"score": score, "mse": mse}
                            best_path = transformed_path

        if best_alignment is not None and best_path is not None:
            cv2.imwrite(str(best_path), best_alignment)
            logger.info(f"✅ Saved: {best_path}")
            results.append({
                "image": best_path,
                "score": best_metrics["score"],
                "mse": best_metrics["mse"],
                "inliers": inliers,
                "inlier_ratio": inliers / len(mask)
            })
        else:
            logger.warning(f"❌ Could not find a suitable alignment for {image_paths[i]}. Exiting.")
            exit(1)
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
    MIN_SCORE_THRESHOLD = config.alignment.MIN_SCORE_THRESHOLD if config and hasattr(config, 'alignment') else 0.5
    MAX_MSE_THRESHOLD = config.alignment.MAX_MSE_THRESHOLD if config and hasattr(config, 'alignment') else 10.0
    MIN_GOOD_MATCHES = config.alignment.MIN_GOOD_MATCHES if config and hasattr(config, 'alignment') else 20
    
    # Aligns images for preprocessing
    if not reference_dir:
        for folder in subdirs:
            image_files = sorted([f for f in folder.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            if not image_files:
                if not quiet:
                    logger.warning(f"No images found in {folder}")
                continue

            sub_output = output_path / folder.name
            align_images(image_files, sub_output, reference_dir, metric_dir, knn_ratio, npts, ransac_thresh, save_metrics, quiet, detect,MIN_SCORE_THRESHOLD, MAX_MSE_THRESHOLD, MIN_GOOD_MATCHES)
            if not quiet:
                logger.info(f"📁 Completed alignment for folder: {folder.name}")
    
    # Aligns images for validation and detection
    else:
        reference_subdirs = mod.rearrange_by_similarity(subdirs, reference_subdirs)
        for folder1, folder2 in zip(subdirs,reference_subdirs):
            image_files = sorted([f for f in folder1.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            reference_files = sorted([d for d in folder2.iterdir() if d.suffix.lower() in ['.png', '.jpg', '.jpeg']]) # how to pair input to refernce?

            if not image_files:
                if not quiet:
                    logger.warning(f"No images found in {folder}")
                continue
           
            sub_output = output_path / folder1.name
            align_images(image_files, sub_output, reference_files, metric_dir, knn_ratio, npts, ransac_thresh, save_metrics, quiet, detect, MIN_SCORE_THRESHOLD, MAX_MSE_THRESHOLD, MIN_GOOD_MATCHES)
            if not quiet:
                logger.info(f"📁 Completed alignment for folder: {folder1.name}")