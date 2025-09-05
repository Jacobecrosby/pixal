import subprocess
import sys
from pathlib import Path
from glob import glob
import yaml
from pixal.modules.config_loader import load_config, resolve_path, resolve_parent_inserted_path

from tensorflow.keras import backend as K
import gc

K.clear_session()
gc.collect()


def run(input_file, config, quiet):
    path_config = load_config("configs/paths.yaml")
    # Load the original YAML
    with open("configs/parameters.yaml", "r") as infile:
        full_config = yaml.safe_load(infile)
        # Extract the 'preprocessing' section
    training_section = full_config.get("model_training", {})

    if config.model_training.one_hot_encoding:
        if not input_file:
            input_file = resolve_path(path_config.component_model_path)
           
        input_folder = Path(input_file)
        npz_file = str(input_folder / config.preprocessing.preprocessor.file_name)

        metadata_path = resolve_parent_inserted_path(path_config.metadata_path, input_folder.name, 1)
        # Save to new YAML file
        with open(metadata_path / "model_training.yaml", "w") as outfile:
            yaml.dump({"model_training": training_section}, outfile, default_flow_style=False)
        
        print(f"Launching subprocess for {npz_file}")
        subprocess.run([
                sys.executable,  
                "pixal/train_model/train_one_hot.py",  
                "--input", str(npz_file),
                "--config", "configs/parameters.yaml"
        ])
    
    else:
        if not input_file:
            input_file = resolve_path(path_config.component_model_path)
        
        input_folder = Path(input_file)
        npz_files = glob(str(input_folder / "*/*.npz"))   

        for npz_path in npz_files:
            path_npz_path = Path(npz_path)
            type_folder = path_npz_path.parent  # One level up from .npz

            # Resolve the metadata path relative to the type folder
            metadata_path = type_folder / resolve_path(path_config.metadata_path.metadata)

            # Save to new YAML file
            with open(metadata_path / "model_training.yaml", "w") as outfile:
                yaml.dump({"model_training": training_section}, outfile, default_flow_style=False)
            
            subprocess.run([
                sys.executable,  
                "pixal/train_model/train_autoencoder.py",  
                "--input", str(npz_path),
                "--config", "configs/parameters.yaml"
            ])