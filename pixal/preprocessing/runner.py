import logging
from pathlib import Path
from pixal.preprocessing import remove_background, align_images, imagePreprocessor
from pixal.modules.config_loader import load_config, resolve_path

def run_preprocessing(input_dir, config=None,quiet=False):
    path_config = load_config("configs/paths.yaml")
    
    metric_dir = resolve_path(path_config.aligned_metrics_path)
    metric_dir.mkdir(parents=True, exist_ok=True)
    
    bg_removed_dir = resolve_path(path_config.remove_background_path)
    bg_removed_dir.mkdir(parents=True, exist_ok=True)
    
    aligned_dir = resolve_path(path_config.aligned_images_path)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    
    npz_dir = resolve_path(path_config.component_model_path)

    reference_dir = None
    
    # Set up logging
    log_path = resolve_path(path_config.log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path / "preprocessing.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger("pixal")

    if not quiet:
        logger.info(f"📁 Logging all preprocessing steps to {log_path}")

    remove_background.run(input_dir, bg_removed_dir, config=config, quiet=quiet)
    align_images.run(bg_removed_dir, aligned_dir, reference_dir, metric_dir, config=config, quiet=quiet)
    imagePreprocessor.run(aligned_dir, npz_dir, config=config, quiet=quiet)

def run_remove_background(input_dir, config=None,quiet=False):
    path_config = load_config("configs/paths.yaml")
    output_dir = resolve_path(path_config.remove_background_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_path = resolve_path(path_config.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True) #bad line?
    logging.basicConfig(
        filename=log_path / "preprocessing.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger("pixal")

    if not quiet:
        logger.info(f"📁 Logging background removal to {log_path}")

    
    remove_background.run(input_dir, output_dir, config=config, quiet=quiet)

def run_align_images(input_dir, config=None,quiet=False):
    path_config = load_config("configs/paths.yaml")
    
    output_dir = resolve_path(path_config.aligned_images_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metric_dir = resolve_path(path_config.aligned_metrics_path)
    metric_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_path = resolve_path(path_config.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path / "preprocessing.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger("pixal")

    if not quiet:
        logger.info(f"📁 Logging image alignment steps to {log_path}")
    
    align_images.run(input_dir, output_dir, metric_dir, config=config, quiet=quiet)

def run_imagePreprocessor(config=None,quiet=False):
    path_config = load_config("configs/paths.yaml")
    input_dir = resolve_path(path_config.aligned_images_path)

    output_dir = resolve_path(path_config.component_model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_path = resolve_path(path_config.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path / "preprocessing.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger("pixal")

    if not quiet:
        logger.info(f"📁 Logging all imagePreprocessor steps to {log_path}")
    
    imagePreprocessor.run(input_dir, output_dir, config, quiet=quiet)