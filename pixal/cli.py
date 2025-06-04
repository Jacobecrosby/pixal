# pixal/cli.py
import traceback
import sys
import argparse
from pathlib import Path
#from pixal import train, detect
from pixal.modules.config_loader import load_config
from pixal.preprocessing import runner as preprocessing_runner
from pixal.preprocessing import remove_background
from pixal.preprocessing import align_images
from pixal.preprocessing import imagePreprocessor


def main():
    run_training = None  # placeholder to supress TensorFlow output
    detect = None       # placeholder to supress Tensorflow output
    try:
        parser = argparse.ArgumentParser(prog="pixal", description="Pixel-based Anomaly Detection CLI")
        subparsers = parser.add_subparsers(dest="command", required=True)

        # Preprocess
        prep = subparsers.add_parser("preprocess", help="Run all preprocessing steps on input images")
        prep.add_argument("--input", "-i", required=True, help="Input image folder")
        prep.add_argument("--quiet","-q", help="Quiet output", action="store_true")

        # Remove background
        rm_bg_cmd = subparsers.add_parser("remove_bg", help="Remove background from images")
        rm_bg_cmd.add_argument("--input", "-i", required=True, help="Input image folder")
        rm_bg_cmd.add_argument("--quiet", "-q", help="Quiet output", action="store_true")

        # align Image
        align_cmd= subparsers.add_parser("align", help="Align images")
        align_cmd.add_argument("--input", "-i", required=True, help="Input image folder")
        align_cmd.add_argument("--quiet", "-q", help="Quiet output", action="store_true")

         # run imagePreprocessor
        make_cmd= subparsers.add_parser("make_input", help="Uses ImagePreprocessor to make ML input")
        #make_cmd.add_argument("--input", "-i", required=True, help="Aligned image folder")
        make_cmd.add_argument("--quiet", "-q", help="Quiet output", action="store_true")

        # Train
        train_cmd = subparsers.add_parser("train", help="Train autoencoder model(s)")
        train_cmd.add_argument("--input", "-i", required=False, help="Input data")
        train_cmd.add_argument("--quiet", "-q", help="Quiet output", action="store_true")

        # Validate
        detect_cmd = subparsers.add_parser("validate", help="Run validation (preprocess + detect) on new images")
        detect_cmd.add_argument("--input","-i", required=True, help="Folder with test images")
        detect_cmd.add_argument("--quiet", "-q", help="Quiet output", action="store_true")

        # Detect
        detect_cmd = subparsers.add_parser("detect", help="Run anomaly detection on new images")
        #detect_cmd.add_argument("--input","-i", required=True, help="Folder with test images")
        detect_cmd.add_argument("--quiet", "-q", help="Quiet output", action="store_true")
        
        args = parser.parse_args()
        cfg = load_config("configs/parameters.yaml") 

        if args.command == "preprocess":
            preprocessing_runner.run_preprocessing(args.input, config=cfg)
        elif args.command == "remove_bg":
            preprocessing_runner.run_remove_background(args.input, config=cfg, quiet=args.quiet)
        elif args.command == "align":
            preprocessing_runner.run_align_images(args.input, config=cfg, quiet=args.quiet)
        elif args.command == "make_input":
            preprocessing_runner.run_imagePreprocessor(config=cfg, quiet=args.quiet)  
        elif args.command == "train":
            if run_training is None:
                from pixal.train_model import run_training
            run_training.run(args.input,config=cfg, quiet=args.quiet)
        elif args.command == "validate":
            if detect is None:
                from pixal.validate import runner as validation_runner
            validation_runner.run_validation(args.input, config=cfg,quiet=args.quiet)
        elif args.command == "detect":
            if detect is None:
                from pixal.validate import runner as validation_runner
            validation_runner.run_detection(config=cfg, quiet=args.quiet)
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb = traceback.extract_tb(exc_tb)[-1]  # Get the last traceback entry
        filename = tb.filename
        line_number = tb.lineno
        print(f"❌ PIXAL CLI crashed: {e} (File: {filename}, Line: {line_number})")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb = traceback.extract_tb(exc_tb)[-1]  # Get the last traceback entry
        filename = tb.filename
        line_number = tb.lineno
        print(f"❌ PIXAL CLI crashed: {e} (File: {filename}, Line: {line_number})")
