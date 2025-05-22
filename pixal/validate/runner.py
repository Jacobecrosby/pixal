import logging
from pathlib import Path
from pixal.preprocessing import remove_background, align_images, imagePreprocessor
from pixal.modules.config_loader import load_config, resolve_path

def run_validation(input_dir, config=None,quiet=False):
    path_config = load_config("configs/paths.yaml")
    config = load_config("configs/parameters.yaml")
    
    metric_dir = resolve_path(path_config.validate_aligned_metrics_path)
    metric_dir.mkdir(parents=True, exist_ok=True)
    
    bg_removed_dir = resolve_path(path_config.validate_remove_background_path)
    bg_removed_dir.mkdir(parents=True, exist_ok=True)
    
    aligned_dir = resolve_path(path_config.validate_aligned_images_path)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    
    reference_dir = resolve_path(path_config.aligned_images_path)
    
    npz_dir = resolve_path(path_config.component_validate_path)
    
    model_dir = resolve_path(path_config.model_path)

    # Set up logging
    log_path = resolve_path(path_config.validate_log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path / "validation.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger("pixal")

    if not quiet:
        logger.info(f"📁 Logging all preprocessing and validation steps to {log_path}")

    remove_background.run(input_dir, bg_removed_dir, config=config, quiet=quiet)
    align_images.run(bg_removed_dir, aligned_dir, reference_dir, metric_dir, config=config, quiet=quiet,detect=True)
    imagePreprocessor.run(aligned_dir, npz_dir, config=config, quiet=quiet)
    
    from pixal.validate import detect
    detect.run(npz_dir, model_dir, metric_dir, config=config, quiet=quiet)

def run_detection(config=None, quiet=False):
    path_config = load_config("configs/paths.yaml")
    config = load_config("configs/parameters.yaml")
    
    metric_dir = resolve_path(path_config.validate_aligned_metrics_path)
    metric_dir.mkdir(parents=True, exist_ok=True)
    
    bg_removed_dir = resolve_path(path_config.validate_remove_background_path)
    bg_removed_dir.mkdir(parents=True, exist_ok=True)
    
    aligned_dir = resolve_path(path_config.validate_aligned_images_path)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    
    #reference_dir = resolve_path(path_config.aligned_images_path)
    
    npz_dir = resolve_path(path_config.component_validate_path)
    
    model_dir = resolve_path(path_config.model_path)

     # Set up logging
    log_path = resolve_path(path_config.validate_log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path / "validation.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger("pixal")
    
    if not quiet:
        logger.info(f"📁 Logging all preprocessing and validation steps to {log_path}")

    from pixal.validate import detect
    detect.run(npz_dir, model_dir, metric_dir, config=config, quiet=quiet)