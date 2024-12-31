# %% [markdown]
# # U-Net Semantic Segmentation (Kaggle Version)
# 
# This notebook demonstrates the implementation and usage of U-Net for semantic segmentation with GPU support.
# You can run this notebook directly in Kaggle!

# %% [markdown]
# ## 1. Setup and Requirements
# First, let's install any additional required packages and check GPU availability.

# %%
# Kaggle already has most packages installed, but we'll add any missing ones
!pip install -q pycocotools

# %%
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm.notebook import tqdm
from pycocotools.coco import COCO

print("TensorFlow version:", tf.__version__)

# Check GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) available and configured: {[gpu.name for gpu in physical_devices]}")
        # Enable mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision training enabled")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found. Running on CPU.")

# %% [markdown]
# ## 2. U-Net Model Implementation

# %%
class UNet:
    def __init__(self, input_size=None, n_classes=1):
        self.input_size = input_size
        self.n_classes = n_classes
        
    def conv_block(self, inputs, filters):
        """Convolutional block with two conv layers"""
        conv = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')(inputs)
        conv = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')(conv)
        return conv
    
    def upconv_block(self, inputs, skip_connection, filters):
        """Upsampling block with skip connection"""
        up = tf.keras.layers.UpSampling2D(size=(2, 2))(inputs)
        up = tf.keras.layers.Conv2D(filters, 2, activation='relu', padding='same')(up)
        concat = tf.keras.layers.Concatenate()([up, skip_connection])
        conv = self.conv_block(concat, filters)
        return conv
        
    def build_model(self):
        # Set compute dtype for better GPU performance
        compute_dtype = tf.float16 if len(tf.config.list_physical_devices('GPU')) > 0 else tf.float32
        
        # Input layer
        if self.input_size is None:
            inputs = tf.keras.layers.Input(shape=(None, None, 3))
        else:
            inputs = tf.keras.layers.Input(shape=self.input_size)
            
        # Cast input to float16 for GPU optimization
        if compute_dtype == tf.float16:
            inputs = tf.keras.layers.Cast(dtype=tf.float16)(inputs)
        
        # Encoder
        conv1 = self.conv_block(inputs, 64)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = self.conv_block(pool1, 128)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = self.conv_block(pool2, 256)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = self.conv_block(pool3, 512)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
        
        # Bridge
        conv5 = self.conv_block(pool4, 1024)
        
        # Decoder
        up6 = self.upconv_block(conv5, conv4, 512)
        up7 = self.upconv_block(up6, conv3, 256)
        up8 = self.upconv_block(up7, conv2, 128)
        up9 = self.upconv_block(up8, conv1, 64)
        
        # Cast back to float32 for output
        if compute_dtype == tf.float16:
            up9 = tf.keras.layers.Cast(dtype=tf.float32)(up9)
        
        # Output
        outputs = tf.keras.layers.Conv2D(self.n_classes, 1, activation='sigmoid')(up9)
        
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def compile_model(self, model, learning_rate=1e-4):
        # Use AMP optimizer if GPU is available
        if len(tf.config.list_physical_devices('GPU')) > 0:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]
        )
        return model

# %% [markdown]
# ## 3. Data Loading and Preprocessing
# 
# ### Option 1: Using Kaggle Dataset
# If you're using this notebook in Kaggle, you can load a dataset directly:

# %%
# Example: Loading data from Kaggle dataset
# Uncomment and modify these lines based on your dataset
"""
# Assuming you've added a dataset to your Kaggle notebook
DATASET_PATH = '../input/your-dataset-name'
images_dir = f'{DATASET_PATH}/images'
annotations_file = f'{DATASET_PATH}/annotations/instances.json'
"""

# %% [markdown]
# ### Option 2: Using Custom Dataset
# You can also use your own dataset by uploading it to Kaggle:

# %%
def load_coco_dataset(images_dir, annotations_file, limit=None):
    """Load and preprocess COCO dataset"""
    coco = COCO(annotations_file)
    
    # Get all image IDs
    image_ids = coco.getImgIds()
    if limit:
        image_ids = image_ids[:limit]
    
    images = []
    masks = []
    image_paths = []
    
    for img_id in tqdm(image_ids, desc="Loading dataset"):
        # Load image info
        img_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join(images_dir, img_info['file_name'])
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Create mask
        mask = np.zeros((img_info['height'], img_info['width']))
        for ann in anns:
            mask = np.maximum(mask, coco.annToMask(ann))
            
        images.append(img)
        masks.append(mask)
        image_paths.append(image_path)
        
    return np.array(images), np.array(masks), image_paths

def prepare_dataset(images, masks, batch_size=4):
    """Prepare dataset with GPU optimization"""
    # Normalize images
    images = images.astype(np.float32) / 255.0
    
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    
    # Optimize dataset for GPU training
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# %% [markdown]
# ## 4. Model Training

# %%
# Load and prepare dataset
# Uncomment and modify these lines based on your dataset setup
"""
images, masks, image_paths = load_coco_dataset(images_dir, annotations_file, limit=100)
dataset = prepare_dataset(images, masks, batch_size=4)

# Create and compile model
unet = UNet()
model = unet.build_model()
model = unet.compile_model(model)

# Training callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'unet_best.keras',
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        profile_batch='500,520'
    )
]

# Train model
history = model.fit(
    dataset,
    epochs=50,
    validation_split=0.2,
    callbacks=callbacks
)
"""

# %% [markdown]
# ## 5. Inference and Visualization

# %%
def load_and_preprocess_image(image_path):
    """Load and preprocess image for inference"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def predict_mask(model, image):
    """Predict segmentation mask"""
    # Add batch dimension
    image_batch = np.expand_dims(image, axis=0)
    
    # Predict
    prediction = model.predict(image_batch)
    mask = prediction[0] > 0.5
    return mask.astype(np.uint8)

def visualize_results(image, mask):
    """Visualize original image, mask and overlay"""
    # Create colored mask
    mask_colored = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(132)
    plt.title('Segmentation Mask')
    plt.imshow(mask, cmap='jet')
    plt.axis('off')
    
    plt.subplot(133)
    plt.title('Overlay')
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.show()

# %% [markdown]
# ### Example: Perform inference on a test image
# Uncomment and modify these lines to test on your own images:

# %%
"""
# Load and test an image
test_image_path = 'path/to/test/image.jpg'
image = load_and_preprocess_image(test_image_path)
mask = predict_mask(model, image)
visualize_results(image, mask)
"""
