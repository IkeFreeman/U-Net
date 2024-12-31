import tensorflow as tf
from tensorflow.keras import layers, models

class UNet:
    def __init__(self, input_size=None, n_classes=1):
        """
        Initialize U-Net model
        Args:
            input_size: Optional tuple (height, width, channels). If None, accepts any size
            n_classes: Number of output classes
        """
        # Configure GPU memory growth
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU(s) available and configured: {[gpu.name for gpu in physical_devices]}")
                # Set mixed precision policy for better GPU performance
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("Mixed precision training enabled")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPU found. Running on CPU.")
            
        self.input_size = input_size
        self.n_classes = n_classes
        
    def build_model(self):
        # Set compute dtype for better GPU performance
        compute_dtype = tf.float16 if len(tf.config.list_physical_devices('GPU')) > 0 else tf.float32
        
        # Use dynamic input size if not specified
        if self.input_size is None:
            inputs = layers.Input(shape=(None, None, 3))
        else:
            inputs = layers.Input(shape=self.input_size)
            
        # Cast input to float16 for GPU optimization if available
        if compute_dtype == tf.float16:
            inputs = layers.Cast(dtype=tf.float16)(inputs)
        
        # Encoder
        conv1 = self.conv_block(inputs, 64)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = self.conv_block(pool1, 128)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = self.conv_block(pool2, 256)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = self.conv_block(pool3, 512)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
        
        # Bridge
        conv5 = self.conv_block(pool4, 1024)
        
        # Decoder
        up6 = self.upconv_block(conv5, conv4, 512)
        up7 = self.upconv_block(up6, conv3, 256)
        up8 = self.upconv_block(up7, conv2, 128)
        up9 = self.upconv_block(up8, conv1, 64)
        
        # Cast back to float32 for output
        if compute_dtype == tf.float16:
            up9 = layers.Cast(dtype=tf.float32)(up9)
        
        # Output
        outputs = layers.Conv2D(self.n_classes, 1, activation='sigmoid')(up9)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def conv_block(self, inputs, filters):
        """Convolutional block with two conv layers"""
        conv = layers.Conv2D(filters, 3, activation='relu', padding='same')(inputs)
        conv = layers.Conv2D(filters, 3, activation='relu', padding='same')(conv)
        return conv
    
    def upconv_block(self, inputs, skip_connection, filters):
        """Upsampling block with skip connection"""
        up = layers.UpSampling2D(size=(2, 2))(inputs)
        up = layers.Conv2D(filters, 2, activation='relu', padding='same')(up)
        concat = layers.Concatenate()([up, skip_connection])
        conv = self.conv_block(concat, filters)
        return conv
        
    def compile_model(self, model, learning_rate=1e-4):
        """Compile model with optimizer and metrics"""
        # Use AMP (Automatic Mixed Precision) optimizer if GPU is available
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
