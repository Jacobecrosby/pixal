import logging
from pathlib import Path
from pixal.preprocessing import remove_background, align_images, imagePreprocessor

def run_preprocessing(input_dir, output_dir, config=None,quiet=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_path = output_dir / "logs" / "preprocessing.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger("pixal")

    if not quiet:
        logger.info(f"📁 Logging all preprocessing steps to {log_path}")

    bg_removed_dir = output_dir / "no_bg"
    aligned_dir = output_dir / "aligned"
    npz_dir = output_dir

    remove_background.run(input_dir, bg_removed_dir, config=config, quiet=quiet)
    align_images.run(bg_removed_dir, aligned_dir, config=config, quiet=quiet)
    imagePreprocessor.run(aligned_dir, npz_dir, config=config, quiet=quiet)
