import numpy as np
import tensorflow as tf
import os
import sys
import json
import cv2
from tqdm import tqdm
from skimage import draw
from pycocotools import mask as coco

# Add parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.model.unet import UNet
from pycocotools.coco import COCO

class ModelTrainer:
    def __init__(self, input_size=(256, 256, 3), batch_size=4):
        # Configure GPU and memory settings
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for gpu in physical_devices:
                    # Enable memory growth
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Set memory limit (optional, adjust based on your GPU)
                    # tf.config.set_logical_device_configuration(
                    #     gpu,
                    #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
                    # )
                print(f"GPU(s) configured for training: {[gpu.name for gpu in physical_devices]}")
                
                # Configure data loading for better GPU utilization
                tf.data.experimental.enable_debug_mode()
                tf.config.optimizer.set_jit(True)  # Enable XLA optimization
                
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPU found. Training will proceed on CPU.")
            
        self.input_size = input_size
        self.batch_size = batch_size
        self.model = None
        
    def load_coco_dataset(self, images_dir, annotation_file):
        """Load and process data from COCO format"""
        # Load annotation file
        print(f"Loading annotations from {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
            
        images = []
        masks = []
        image_paths = []  # Store original image paths
        
        # Get all images
        print("Loading and processing data...")
        
        for img_info in tqdm(coco_data['images'], desc="Loading images"):
            img_path = os.path.join(images_dir, img_info['file_name'])
            
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found, skipping...")
                continue
                
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}, skipping...")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create mask from annotations
            mask = np.zeros((img_info['height'], img_info['width']))
            
            # Find annotations for this image
            img_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_info['id']]
            
            for ann in img_anns:
                # Handle different segmentation formats
                if 'segmentation' in ann:
                    if isinstance(ann['segmentation'], dict):  # RLE format
                        if 'counts' in ann['segmentation']:
                            mask_part = coco.maskUtils.decode(ann['segmentation'])
                            mask = np.maximum(mask, mask_part)
                    elif isinstance(ann['segmentation'], list):  # Polygon format
                        # Convert polygon to mask
                        poly = np.array(ann['segmentation'][0]).reshape((-1, 2))
                        rr, cc = draw.polygon(poly[:, 1], poly[:, 0], mask.shape)
                        mask[rr, cc] = 1
                
            # Resize image and mask
            img = cv2.resize(img, self.input_size[:2])
            mask = cv2.resize(mask, self.input_size[:2])
            mask = (mask > 0).astype(np.float32)
            
            images.append(img)
            masks.append(mask)
            image_paths.append(img_path)
            
        if not images:
            raise Exception("No valid images found")
            
        print(f"Successfully loaded {len(images)} images and masks")
        return np.array(images), np.array(masks), image_paths
        
    def prepare_dataset(self, images, masks):
        """Prepare dataset with GPU optimization"""
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, masks))
        
        # Optimize dataset for GPU training
        dataset = dataset.cache()  # Cache the dataset in memory
        dataset = dataset.shuffle(buffer_size=1000)  # Shuffle with larger buffer
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch next batch
        
        return dataset
        
    def create_model(self):
        """Create and compile model"""
        self.model = UNet(input_size=self.input_size).build_model()
        
        # Use binary crossentropy for segmentation
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]
        )
        return self.model
        
    def train(self, train_dataset, validation_split=0.2, epochs=50, callbacks=None):
        """Train the model with GPU optimizations"""
        if self.model is None:
            self.create_model()
            
        # Calculate steps per epoch
        steps_per_epoch = len(train_dataset) // self.batch_size
        validation_steps = int(steps_per_epoch * validation_split)
        
        # Default callbacks for GPU training
        if callbacks is None:
            callbacks = []
            
        # Add GPU-specific callbacks
        callbacks.extend([
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # TensorBoard callback for monitoring GPU usage
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                profile_batch='500,520'  # Profile GPU performance
            )
        ])
        
        # Train with mixed precision if GPU is available
        if len(tf.config.list_physical_devices('GPU')) > 0:
            print("Training with mixed precision on GPU")
        else:
            print("Training on CPU")
            
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_split=validation_split,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
        
    def load_weights(self, weights_path):
        """Load trained weights"""
        if self.model is None:
            self.create_model()
        self.model.load_weights(weights_path)
        
class TqdmCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epochs_pbar = None
        self.train_pbar = None
        
    def on_train_begin(self, logs=None):
        print("Starting training...")
        
    def on_epoch_begin(self, epoch, logs=None):
        if self.epochs_pbar is None:
            self.epochs_pbar = tqdm(total=self.params['epochs'], desc='Epochs', position=0)
        if self.train_pbar is None:
            self.train_pbar = tqdm(total=self.params['steps'], desc='Training', position=1, leave=False)
        self.train_pbar.reset()
        
    def on_train_batch_end(self, batch, logs=None):
        self.train_pbar.update(1)
        metrics = {
            'loss': f"{logs['loss']:.4f}",
            'acc': f"{logs['accuracy']:.4f}"
        }
        if 'mean_io_u' in logs:
            metrics['IoU'] = f"{logs['mean_io_u']:.4f}"
        self.train_pbar.set_postfix(metrics)
        
    def on_epoch_end(self, epoch, logs=None):
        self.epochs_pbar.update(1)
        metrics = {
            'val_loss': f"{logs['val_loss']:.4f}",
            'val_acc': f"{logs['val_accuracy']:.4f}"
        }
        # Add validation IoU if available
        if 'val_mean_io_u' in logs:
            metrics['val_IoU'] = f"{logs['val_mean_io_u']:.4f}"
        self.epochs_pbar.set_postfix(metrics)
        
        # Print detailed metrics for this epoch
        print(f"\nEpoch {epoch + 1}/{self.params['epochs']}")
        print(f"loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f}", end='')
        if 'mean_io_u' in logs:
            print(f" - IoU: {logs['mean_io_u']:.4f}", end='')
        print(f"\nval_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}", end='')
        if 'val_mean_io_u' in logs:
            print(f" - val_IoU: {logs['val_mean_io_u']:.4f}", end='')
        print("\n")
        
    def on_train_end(self, logs=None):
        self.epochs_pbar.close()
        self.train_pbar.close()
        print("\nTraining completed!")
        
def create_sample_model():
    """Create sample model with random weights"""
    trainer = ModelTrainer()
    model = trainer.create_model()
    
    # Create weights directory if not exists
    os.makedirs('weights', exist_ok=True)
    
    # Save model
    model.save_weights('weights/unet.keras')
    print("Created and saved sample model")

if __name__ == '__main__':
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Set absolute paths for data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Load COCO data
    images_dir = os.path.join(project_root, 'src', 'data', 'images')
    annotations_file = os.path.join(project_root, 'src', 'data', 'annotations.json')
    
    try:
        print("Loading COCO data...")
        print(f"Images directory: {images_dir}")
        print(f"Annotations file: {annotations_file}")
        
        images, masks, image_paths = trainer.load_coco_dataset(images_dir, annotations_file)
        
        print(f"Loaded {len(images)} images and masks")
        print("Preparing dataset...")
        
        dataset = trainer.prepare_dataset(images, masks)
        
        print("Starting training...")
        
        # Train model
        history = trainer.train(dataset, epochs=50)
        
        print("Training completed!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
