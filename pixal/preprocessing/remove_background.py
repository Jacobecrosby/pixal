import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold('error')

from pathlib import Path
from rembg import remove
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

logger = logging.getLogger("pixal")

def process_image(img_file, output_path, target_size=None):
    try:
        with Image.open(img_file) as img:
            img = img.convert("RGBA")
            logger.info(f"Removing background for image: {img}")
            output = remove(img)

            if target_size is None:
                target_size = output.size

            output = output.resize(target_size, Image.LANCZOS)

            output_file = output_path / f"{img_file.stem}_no_bg.png"
            logger.info(f"Saving image with background removed as: {output_file}")
            output.save(output_file)

        return img_file.name, target_size
    except Exception as e:
        return f"Error: {img_file.name} ({e})", target_size

def remove_backgrounds(input_folder, output_folder, max_workers=4, quiet=False):
    input_path = Path(input_folder)
    output_path = output_folder

    if not input_path.exists():
        logger.error(f"Input folder does not exist: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    supported_extensions = ('.jpg', '.jpeg', '.png', '.webp')

    # Loop through all subdirectories (or the top folder if there are no subdirs)
    subdirs = [p for p in input_path.iterdir() if p.is_dir()]
    if not subdirs:
        subdirs = [input_path]

    for folder in subdirs:
        sub_output = output_path / folder.name
        sub_output.mkdir(parents=True, exist_ok=True)

        image_files = [f for f in folder.iterdir() if f.suffix.lower() in supported_extensions]
        if not image_files:
            logger.warning(f"No images found in {folder}")
            continue

        target_size = None
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_image, f, sub_output, target_size): f
                for f in image_files
            }

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"Removing backgrounds in '{folder.name}'", disable=quiet):
                result, size = future.result()
                if isinstance(result, str) and result.startswith("Error"):
                    logger.warning(result)
                elif target_size is None:
                    target_size = size  # Set once

def run(input_folder, output_folder, config=None, quiet=False):
    max_workers = config.remove_background.max_workers if config and hasattr(config, 'remove_background') else 4
    remove_backgrounds(input_folder, output_folder, max_workers,quiet=quiet)

