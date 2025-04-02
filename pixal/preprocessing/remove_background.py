import os
import sys
from pathlib import Path
from rembg import remove
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_image(img_file, output_path):
    try:
        with Image.open(img_file) as img:
            img = img.convert("RGBA")
            output = remove(img)
            output_file = output_path / f"{img_file.stem}_no_bg{img_file.suffix}"
            output.save(output_file)
        return img_file.name
    except Exception as e:
        return f"Error: {img_file.name} ({e})"

def remove_backgrounds(input_folder, output_folder, max_workers=4):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        print(f"Input folder does not exist: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    supported_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in supported_extensions]

    target_size = None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for i, f in enumerate(image_files):
            futures[executor.submit(process_image, f, output_path, target_size)] = f

        for future in tqdm(as_completed(futures), total=len(futures), desc="Removing backgrounds"):
            result, size = future.result()
            if isinstance(result, str) and result.startswith("Error"):
                print(result)
            elif target_size is None:
                target_size = size  # Set the standard size from first processed image


def process_image(img_file, output_path, target_size=None):
    try:
        with Image.open(img_file) as img:
            img = img.convert("RGBA")
            output = remove(img)

            # Set target size if it's not defined
            if target_size is None:
                target_size = output.size  # (width, height)

            output = output.resize(target_size, Image.LANCZOS)

            output_file = output_path / f"{img_file.stem}_no_bg.png"
            output.save(output_file)

        return img_file.name, target_size
    except Exception as e:
        return f"Error: {img_file.name} ({e})", target_size


def run(input_folder, output_folder):
    input_path = Path(input_folder)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    remove_backgrounds(input_folder, output_folder)
