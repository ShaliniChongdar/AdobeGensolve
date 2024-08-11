# AdobeGensolve
CURVETOPIA: A Journey into the World of Curves

TASK 1 : REGULARISE & IDENTIFY CURVES:

# model
                  import tensorflow as tf
                  from tensorflow.keras import datasets, layers, models
                  import matplotlib.pyplot as plt
                  import numpy as np

This code snippet imports TensorFlow and Keras for deep learning tasks, including building and training neural networks. It also imports Matplotlib for plotting and visualizing data, and NumPy for handling arrays and performing numerical operations. The combination of these libraries is commonly used in machine learning and data science.


                  from google.colab import drive
                  drive.mount('/content/drive')


This code snippet mounts Google Drive to the file system in a Google Colab environment, allowing you to access files stored in your Google Drive from within the Colab notebook. The mounted drive will appear at the `/content/drive` directory, making it easy to read and write files directly to your Drive.


                  import os
                  from PIL import Image
                  import numpy as np
                  
                  def load_and_resize_doodles(input_folders, size=(28, 28)):
                      X = []
                  
                      for input_folder in input_folders:
                          for filename in os.listdir(input_folder):
                              if filename.endswith(('.png', '.jpg', '.jpeg')):
                                  with Image.open(os.path.join(input_folder, filename)) as img:
                                      # Convert to grayscale
                                      img = img.convert('L')
                  
                                      # Resize the image
                                      img_resized = img.resize(size, Image.LANCZOS)
                  
                                      # Convert image to numpy array
                                      img_array = np.array(img_resized)
                  
                                      # Invert the image (assuming doodles are dark on light background)
                                      img_array = 255 - img_array
                  
                                      # Enhance contrast
                                      img_array = np.clip(img_array * 1.5, 0, 255).astype(np.uint8)
                  
                                      # Normalize to [0, 1]
                                      img_array = img_array / 255.0
                  
                                      # Append to X
                                      X.append(img_array)
                                      print(f"Processed: {os.path.join(input_folder, filename)}")
                      # Convert list to numpy array
                      X = np.array(X)
                      return X
                  # Usage
                  input_folders = [
                      '/content/drive/MyDrive/Circles & Ellipses',
                      '/content/drive/MyDrive/Rectangles & Rounded Rectangles',
                      '/content/drive/MyDrive/Straight Lines',
                      '/content/drive/MyDrive/Star',
                      '/content/drive/MyDrive/Regular Polygons'
                  ]
                  
                  X = load_and_resize_doodles(input_folders)
                  print(f"X shape: {X.shape}")
                  print(f"Shape of each image: {X[0].shape}")

This code loads and processes images from specific folders, converting them into a consistent format suitable for machine learning tasks. It resizes the images to 28x28 pixels, converts them to grayscale, inverts them (making dark doodles on a light background), enhances their contrast, and normalizes their pixel values to the range [0,1]. The processed images are stored in a NumPy array `X`, which is then used for visualization and further analysis.


                import numpy as np
                
                y_train = np.concatenate([
                    np.ones(441),
                    np.full(205, 2),
                    np.full(150, 3),
                    np.full(116, 4),
                    np.full(191, 5)
                ])
                Y_train.shape

This code creates a NumPy array `y_train` containing labels for a dataset. The array is formed by concatenating different arrays, where:

- `441` elements are labeled `1`,
- `205` elements are labeled `2`,
- `150` elements are labeled `3`,
- `116` elements are labeled `4`, and
- `191` elements are labeled `5`.

Finally, `y_train.shape` is used to display the shape of the resulting `y_train` array.

                          import numpy as np
                          from sklearn.model_selection import train_test_split
                          indices = np.arange(len(y_train))
                          np.random.shuffle(indices)
                          X = X[indices]
                          y_train = y_train[indices]
                          
                          X_train, X_test, y_train, y_test = train_test_split(X, y_train, test_size=0.2, random_state=42)
                          X_train.shape
                          y_train.shape
                          X_train[0]
                          X_train_flattened = X_train.reshape(len(X_train), 28*28)
                          X_test_flattened = X_test.reshape(len(X_test), 28*28)
                          X_train_flattened.shape
                          X_train_flattened[0]

This code shuffles and splits a dataset into training and testing sets, then flattens the images for machine learning.

- It shuffles the dataset by generating random indices and rearranging `X` and `y_train` accordingly.
- The dataset is split into training (80%) and testing (20%) sets using `train_test_split`.
- The training and testing images are reshaped from 28x28 pixels into 1D arrays of 784 elements (`28*28`), making them suitable for certain machine learning models.
- Finally, the shapes and first elements of these flattened arrays are checked.


                          from tensorflow import keras
                          
                          import tensorflow as tf
                          from tensorflow import keras
                          from tensorflow.keras import layers
                          
                          model = tf.keras.Sequential([
                              # Reshape the input from (784,) to (28, 28, 1)
                              layers.Reshape((28, 28, 1), input_shape=(784,)),
                          
                              layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
                              layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
                              layers.MaxPooling2D(pool_size=(2, 2)),
                          
                              layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
                              layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
                              layers.MaxPooling2D(pool_size=(2, 2)),
                          
                              layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                              layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                              layers.MaxPooling2D(pool_size=(2, 2)),
                          
                              layers.Dropout(0.3),
                              layers.Flatten(),
                              layers.Dense(256, activation='relu'),
                              layers.BatchNormalization(),
                              layers.Dropout(0.3),
                              layers.Dense(128, activation='relu'),
                              layers.BatchNormalization(),
                              layers.Dense(10, activation='softmax')  # 10 output classes
                          ])
                          
                          # Compile the model
                          model.compile(
                              loss='sparse_categorical_crossentropy',
                              optimizer=tf.keras.optimizers.Adam(),
                              metrics=['accuracy']
                          )
                          
                          # Train the model
                          model.fit(X_train_flattened, y_train, epochs=50, validation_split=0.2)
                          model.evaluate(X_test_flattened,y_test)

This code defines, compiles, and trains a Convolutional Neural Network (CNN) model using TensorFlow and Keras.

- Model Structure:
  - Reshape: Converts the input from a flat 784-element array into a 28x28x1 image format.
  - Conv2D & MaxPooling: Three sets of convolutional layers followed by max-pooling layers extract features and reduce spatial dimensions.
  - Dropout: Randomly drops out 30% of neurons to prevent overfitting.
  - Flatten & Dense Layers: Flattens the output and connects it to dense layers for classification.
  - BatchNormalization: Normalizes layer inputs to improve training stability.
  - Output Layer: Uses softmax activation to output probabilities for 10 classes.
- Compilation: The model uses sparse categorical cross entropy loss and the Adam optimizer.
- Training: The model is trained for 50 epochs with 20% of the training data used for validation.
- Evaluation: The model's performance is evaluated on the test data.



TASK 2 :


