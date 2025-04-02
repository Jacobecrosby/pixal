from pixal.preprocessing import remove_background, align_images, imagePreprocessor

def run_preprocessing(input_dir, output_dir, config=None,quiet=False):
    bg_removed_dir = output_dir / "no_bg"
    aligned_dir = output_dir / "aligned"
    npz_dir = output_dir / "npz"

    remove_background.run(input_dir, bg_removed_dir, config=config,quiet=quiet)
    align_images.run(bg_removed_dir, aligned_dir, config=config,quiet=quiet)
    imagePreprocessor.run(aligned_dir, npz_dir, config=config,quiet=quiet)
