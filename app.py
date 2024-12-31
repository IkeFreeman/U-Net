import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from src.model.unet import UNet
from src.model.train import ModelTrainer
from src.inference.predictor import Predictor
from src.data.data_loader import DataLoader
import tensorflow as tf

class App:
    def __init__(self, root):
        self.root = root
        self.root.title('U-Net Semantic Segmentation')
        self.root.geometry('1200x800')
        
        # Style
        style = ttk.Style()
        style.configure('TButton', padding=5, font=('Arial', 10))
        style.configure('TLabel', font=('Arial', 10))
        
        # Main container with notebook
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Inference tab
        self.inference_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.inference_frame, text='Inference')
        self.setup_inference_tab()
        
        # Training tab
        self.training_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.training_frame, text='Training')
        self.setup_training_tab()
        
        self.current_image = None
        self.predictor = None
        
    def setup_inference_tab(self):
        # Top frame for buttons and info
        top_frame = ttk.Frame(self.inference_frame)
        top_frame.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(top_frame)
        button_frame.pack(side=tk.LEFT, fill='x')
        
        ttk.Button(button_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Segment Image", command=self.segment_image).pack(side=tk.LEFT, padx=5)
        
        # Image info label
        self.info_label = ttk.Label(top_frame, text="")
        self.info_label.pack(side=tk.LEFT, padx=20)
        
        # Create canvas frame
        canvas_frame = ttk.Frame(self.inference_frame)
        canvas_frame.grid(row=1, column=0, columnspan=3, sticky='nsew')
        
        # Configure grid weights
        self.inference_frame.grid_rowconfigure(1, weight=1)
        self.inference_frame.grid_columnconfigure((0,1,2), weight=1)
        
        # Create canvases with scrollbars
        self.setup_scrollable_canvas(canvas_frame, 0, "Original Image")
        self.setup_scrollable_canvas(canvas_frame, 1, "Segmentation Mask")
        self.setup_scrollable_canvas(canvas_frame, 2, "Overlay")
        
    def setup_scrollable_canvas(self, parent, col, title):
        # Create a frame for the canvas
        frame = ttk.LabelFrame(parent, text=title, padding="5")
        frame.grid(row=0, column=col, padx=5, sticky='nsew')
        
        # Create canvas with scrollbars
        canvas = tk.Canvas(frame, highlightthickness=0)
        xscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        
        # Configure canvas
        canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        
        # Grid layout
        canvas.grid(row=0, column=0, sticky='nsew')
        xscroll.grid(row=1, column=0, sticky='ew')
        yscroll.grid(row=0, column=1, sticky='ns')
        
        # Configure frame grid
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        # Store canvas reference
        if col == 0:
            self.original_canvas = canvas
        elif col == 1:
            self.mask_canvas = canvas
        else:
            self.overlay_canvas = canvas
            
    def update_canvas_image(self, canvas, image, title):
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
            
        # Store as PhotoImage
        photo = ImageTk.PhotoImage(image)
        
        # Update canvas
        canvas.delete("all")
        canvas.create_image(0, 0, image=photo, anchor='nw')
        canvas.image = photo  # Keep reference
        
        # Update scrollregion
        canvas.configure(scrollregion=(0, 0, image.width, image.height))
        
    def load_image(self):
        # Open file chooser dialog
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        try:
            # Load image
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Error", "Could not load the selected image.")
                return
                
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Store original image
            self.current_image = img
            
            # Update info label
            h, w = img.shape[:2]
            self.info_label.configure(text=f"Image size: {w}x{h} pixels")
            
            # Display original image
            self.update_canvas_image(self.original_canvas, img, "Original Image")
            
            # Clear other canvases
            self.mask_canvas.delete("all")
            self.overlay_canvas.delete("all")
            
            # Update window title
            self.root.title(f'U-Net Semantic Segmentation - {os.path.basename(file_path)}')
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
            
    def segment_image(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
            
        try:
            # Initialize predictor if not already done
            if self.predictor is None:
                self.predictor = Predictor()
                
            # Get prediction
            mask = self.predictor.predict(self.current_image)
            
            # Create colored mask
            mask_colored = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
            mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
            
            # Create overlay
            overlay = self.predictor.overlay_mask(self.current_image, mask)
            
            # Update displays
            self.update_canvas_image(self.mask_canvas, mask_colored, "Segmentation Mask")
            self.update_canvas_image(self.overlay_canvas, overlay, "Overlay")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during segmentation: {str(e)}")
            
    def setup_training_tab(self):
        # Training parameters
        param_frame = ttk.LabelFrame(self.training_frame, text="Training Parameters", padding="10")
        param_frame.grid(row=0, column=0, sticky='ew', pady=10)
        
        # Epochs
        ttk.Label(param_frame, text="Epochs:").grid(row=0, column=0, padx=5, pady=5)
        self.epochs_var = tk.StringVar(value="50")
        ttk.Entry(param_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Batch size
        ttk.Label(param_frame, text="Batch Size:").grid(row=0, column=2, padx=5, pady=5)
        self.batch_size_var = tk.StringVar(value="4")
        ttk.Entry(param_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Data paths
        data_frame = ttk.LabelFrame(self.training_frame, text="Training Data", padding="10")
        data_frame.grid(row=1, column=0, sticky='ew', pady=10)
        
        # Images folder
        ttk.Label(data_frame, text="Images Folder:").grid(row=0, column=0, padx=5, pady=5)
        self.train_images_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.train_images_var, width=40).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(data_frame, text="Browse", command=lambda: self.browse_folder(self.train_images_var)).grid(row=0, column=2, padx=5, pady=5)
        
        # Annotations file
        ttk.Label(data_frame, text="Annotations File:").grid(row=1, column=0, padx=5, pady=5)
        self.annotations_var = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.annotations_var, width=40).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(data_frame, text="Browse", command=lambda: self.browse_file(self.annotations_var)).grid(row=1, column=2, padx=5, pady=5)
        
    def browse_folder(self, var):
        folder = filedialog.askdirectory()
        if folder:
            var.set(folder)
            
    def browse_file(self, var):
        file_path = filedialog.askopenfilename(
            title="Select File",
            filetypes=[
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            var.set(file_path)
            
    def start_training(self):
        try:
            # Get parameters
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            images_dir = self.train_images_var.get()
            annotations_file = self.annotations_var.get()
            
            if not images_dir or not annotations_file:
                messagebox.showerror("Error", "Please select both images folder and annotations file")
                return
                
            # Get image paths
            image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            if len(image_paths) == 0:
                messagebox.showerror("Error", "No images found in the selected folder")
                return
                
            # Prepare dataset
            self.status_var.set("Preparing dataset...")
            self.root.update()
            
            # Create trainer with specified batch size
            self.trainer = ModelTrainer(batch_size=batch_size)
            train_images, train_masks = self.trainer.prepare_dataset(image_paths, annotations_file)
            
            # Create progress callback
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, progress_var, status_var, root):
                    self.progress_var = progress_var
                    self.status_var = status_var
                    self.root = root
                    
                def on_epoch_end(self, epoch, logs=None):
                    progress = ((epoch + 1) / self.params['epochs']) * 100
                    self.progress_var.set(progress)
                    self.status_var.set(f"Epoch {epoch + 1}/{self.params['epochs']}, Loss: {logs['loss']:.4f}")
                    self.root.update()
            
            # Start training
            self.status_var.set("Training started...")
            self.root.update()
            
            history = self.trainer.train(
                train_images, train_masks,
                epochs=epochs,
                callbacks=[ProgressCallback(self.progress_var, self.status_var, self.root)]
            )
            
            self.status_var.set("Training completed! Model saved.")
            messagebox.showinfo("Success", "Training completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set(f"Error: {str(e)}")
            
if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
