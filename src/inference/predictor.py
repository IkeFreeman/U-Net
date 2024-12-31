import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

class Predictor:
    def __init__(self, weights_path='weights/unet_best.keras'):
        """
        Initialize predictor with model weights
        Args:
            weights_path: Path to model weights file
        """
        from src.model.unet import UNet
        self.model = UNet(input_size=None).build_model()
        try:
            self.model.load_weights(weights_path)
        except Exception as e:
            raise Exception(f"Failed to load model weights from {weights_path}: {str(e)}")
        
    def predict(self, image):
        """
        Predict segmentation mask for a single image
        Args:
            image: Input image (numpy array)
        Returns:
            Predicted mask at original image size
        """
        if len(image.shape) == 3:
            # Add batch dimension if not present
            image = np.expand_dims(image, axis=0)
            
        if image.dtype != np.float32:
            # Normalize image to [0, 1] range
            image = image.astype(np.float32) / 255.0
            
        # Get original size
        original_size = image.shape[1:3]
        
        # Make prediction
        prediction = self.model.predict(image)
        mask = prediction[0]  # Remove batch dimension
        
        # Threshold mask
        mask = (mask > 0.5).astype(np.uint8)
        
        return mask
        
    def overlay_mask(self, image, mask, alpha=0.5):
        """
        Overlay the predicted mask on the original image
        Args:
            image: Original image (H, W, 3)
            mask: Predicted mask (H, W)
            alpha: Transparency of the overlay (0-1)
        Returns:
            Image with overlaid mask
        """
        # Create colored mask
        mask_colored = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = cv2.addWeighted(image, 1-alpha, mask_colored, alpha, 0)
        return overlay
        
    def visualize_prediction(self, image, save_path=None):
        """
        Visualize original image, prediction mask and overlay
        Args:
            image: Input image
            save_path: Optional path to save visualization
        """
        # Make prediction
        mask = self.predict(image)
        overlay = self.overlay_mask(image, mask)
        
        # Create visualization
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
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
