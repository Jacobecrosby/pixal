# pixal/cli.py

import argparse
from pathlib import Path
#from pixal import train, detect
from pixal.config_loader import load_config
from pixal.preprocessing import runner as preprocessing_runner
from pixal.preprocessing import remove_background



def main():
    try:
        parser = argparse.ArgumentParser(prog="pixal", description="Pixel-based Anomaly Detection CLI")
        subparsers = parser.add_subparsers(dest="command", required=True)

        # Preprocess
        prep = subparsers.add_parser("preprocess", help="Preprocess input images")
        prep.add_argument("--input", "-i", required=True, help="Input image folder")
        prep.add_argument("--output", "-o", required=True, help="Output folder for processed images")
        prep.add_argument("--quiet","-q", help="Quiet output", action="store_true")

        # Remove background
        rm_bg_cmd = subparsers.add_parser("remove_bg", help="Remove background from images")
        rm_bg_cmd.add_argument("--input", "-i", required=True, help="Input image folder")
        rm_bg_cmd.add_argument("--output", "-o", required=True, help="Output folder for images with removed background")
        rm_bg_cmd.add_argument("--quiet", "-q", help="Quiet output", action="store_true")

        # Train
        #train_cmd = subparsers.add_parser("train", help="Train the anomaly detection model")
        #train_cmd.add_argument("--config", required=True, help="Path to YAML config")

        # Detect
        #detect_cmd = subparsers.add_parser("detect", help="Run anomaly detection on new images")
        #detect_cmd.add_argument("--images", required=True, help="Folder with test images")
        #detect_cmd.add_argument("--model", required=True, help="Path to trained model")
        #detect_cmd.add_argument("--output", required=True, help="Output folder")

        args = parser.parse_args()
        cfg = load_config("configs/config.yaml") if args.command in {"preprocess", "remove_bg"} else None

        if args.command == "preprocess":
            preprocessing_runner.run_preprocessing(Path(args.input), Path(args.output), config=cfg)
        elif args.command == "remove_bg":
            remove_background.run(args.input, args.output, config=cfg, quiet=args.quiet)
        #elif args.command == "train":
        #    train.run(args.config)
        #elif args.command == "detect":
        #    detect.run(args.images, args.model, args.output)
    except Exception as e:
        print(f"❌ PIXAL CLI crashed: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ PIXAL crashed: {e}")
