import sys
from pathlib import Path
from PIL import Image

def flip_image_180(image_path: str):
    path = Path(image_path)
    if not path.exists():
        print(f"Error: {image_path} does not exist.")
        return
    
    img = Image.open(path)
    flipped = img.rotate(180)
    flipped.save(path)  # overwrite original
    print(f"Flipped and saved over {path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python flip_image.py <path/to/image>")
    else:
        flip_image_180(sys.argv[1])
