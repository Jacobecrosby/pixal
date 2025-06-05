import subprocess
import sys
from pathlib import Path
from glob import glob
from pixal.modules.config_loader import load_config, resolve_path

def run(input_file, config, quiet):
    path_config = load_config("configs/paths.yaml")

    if config.one_hot_encoding:
        if not input_file:
            input_file = resolve_path(path_config.component_model_path)
           
        input_folder = Path(input_file)
        npz_file = str(input_folder / config.preprocessor.file_name)

        
        print(f"Launching subprocess for {npz_file}")
        subprocess.run([
                sys.executable,  # Python executable
                "pixal/train_model/train_one_hot.py",  # Relative path to the subprocess script
                "--input", str(npz_file),
                "--config", "configs/parameters.yaml"
        ])
    
    else:
        if not input_file:
            input_file = resolve_path(path_config.component_model_path)
            
        input_folder = Path(input_file)
        npz_files = glob(str(input_folder / "*/*.npz"))

        for npz_path in npz_files:
            print(f"Launching subprocess for {npz_path}")
            subprocess.run([
                sys.executable,  # Python executable
                "pixal/train_model/train_autoencoder.py",  # Relative path to the subprocess script
                "--input", str(npz_path),
                "--config", "configs/parameters.yaml"
            ])