import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_gpu():
    """Check if GPU is available with TensorFlow"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  - {gpu.name}")
            return True
        else:
            print("No GPU found. Will use CPU for training.")
            return False
    except Exception as e:
        print(f"Error checking GPU: {e}")
        return False

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Successfully installed required packages.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)

def setup_directories():
    """Create necessary directories"""
    directories = ['weights', 'logs', 'data', 'data/images', 'data/annotations']
    for dir_name in directories:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    print("Created necessary directories.")

def run_training(args):
    """Run model training"""
    print("Starting training...")
    try:
        from src.model.train import train_model
        train_model(
            images_dir=args.images_dir,
            annotations_file=args.annotations_file,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

def run_app():
    """Run the GUI application"""
    print("Starting U-Net Segmentation App...")
    try:
        import tkinter as tk
        from app import App
        root = tk.Tk()
        app = App(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting app: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='U-Net Segmentation Setup and Runner')
    parser.add_argument('--mode', choices=['train', 'app', 'setup'], default='setup',
                      help='Mode to run: train, app, or setup')
    parser.add_argument('--images_dir', type=str, help='Directory containing training images')
    parser.add_argument('--annotations_file', type=str, help='Path to COCO annotations file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    
    args = parser.parse_args()
    
    if args.mode == 'setup' or args.mode == 'train':
        print("Setting up U-Net Segmentation project...")
        install_requirements()
        setup_directories()
        check_gpu()
    
    if args.mode == 'train':
        if not args.images_dir or not args.annotations_file:
            parser.error("--images_dir and --annotations_file are required for training mode")
        run_training(args)
    elif args.mode == 'app':
        run_app()

if __name__ == "__main__":
    main()
