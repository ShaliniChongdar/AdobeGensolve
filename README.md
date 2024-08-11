# AdobeGensolve
CURVETOPIA: A Journey into the World of Curves

# **TASK 1 : REGULARISE & IDENTIFY CURVES:**


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



# **TASK 2 : UNDERSTANDING AND IDENTIFYING SYMMETRIES** 

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    def preprocess_image(img):
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        # Apply adaptive thresholding to handle uneven lighting
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        return thresh
    
    def check_symmetry(img, angle, threshold=0.8):
        height, width = img.shape
        center = (width // 2, height // 2)

    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate image
    rotated = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_NEAREST)

    # Split the rotated image
    left = rotated[:, :width//2]
    right = rotated[:, width//2:]
    right_flipped = cv2.flip(right, 1)

    # Calculate similarity
    diff = cv2.absdiff(left, right_flipped)
    similarity = 1 - (np.sum(diff) / (width * height * 255))

    return similarity > threshold

    def find_symmetries(image_path, angle_step=2, threshold=0.95):
        # Read and preprocess the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = preprocess_image(img)

    symmetries = []
    for angle in range(0, 180, angle_step):
        if check_symmetry(img, angle, threshold):
            symmetries.append(angle)

    return symmetries

    def plot_symmetries(image_path, symmetries):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        center = (width // 2, height // 2)

    plt.figure(figsize=(12, 10))
    plt.imshow(img, cmap='gray')

    for angle in symmetries:
        # Calculate end points of the line
        rho = max(width, height)
        a = np.cos(np.radians(angle))
        b = np.sin(np.radians(angle))
        x0 = a * rho + center[0]
        y0 = b * rho + center[1]
        x1 = -a * rho + center[0]
        y1 = -b * rho + center[1]

        plt.plot([x0, x1], [y0, y1], color='r', linestyle='--', label=f'Symmetry at {angle}°')

    plt.title(f'Hand-drawn Image with Detected Symmetry Axes (Total: {len(symmetries)})')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    image_path = '/content/Hexagon (6).png'
    symmetries = find_symmetries(image_path)
    print(f"Number of symmetries found: {len(symmetries)}")
    for angle in symmetries:
      print(f"Symmetry found at angle: {angle}°")

    plot_symmetries(image_path, symmetries)


Here’s a brief description of the code:

1. preprocess_image(img):
   - Applies Gaussian blur and adaptive thresholding to preprocess an image for better symmetry detection.

2. check_symmetry(img, angle, threshold=0.8):
   - Rotates the image by a specified angle, checks for symmetry by comparing the left and right halves, and returns whether the symmetry is above a certain threshold.

3. find_symmetries(image_path, angle_step=2, threshold=0.95):
   - Reads and preprocesses an image, then checks for symmetry at various angles. Returns a list of angles where symmetry is detected.

4. plot_symmetries(image_path, symmetries):
   - Displays the image with lines indicating detected symmetry axes.

5. Usage:
   - Reads an image, detects symmetries, prints the results, and plots the symmetries on the image.

This script is used for detecting and visualizing symmetrical axes in images.


# **TASK 3: COMPLETE THE IMAGE**


    import cv2
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras import layers, models
    import matplotlib.pyplot as plt


The code imports essential libraries for image processing, machine learning, and visualization. It utilizes OpenCV (`cv2`) for handling image data, NumPy for numerical operations, and TensorFlow/Keras for loading and building deep learning models. Matplotlib is used for plotting and visualizing data or model results. This setup is typically used for tasks like image classification or other computer vision applications.


    
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    def build_model(input_shape=(64, 64, 1)):
        model = models.Sequential()
    
    # Define input layer
    model.add(layers.Input(shape=input_shape))

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    
    # Flatten and Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))

    # Reshape output to 64x64 dimensions
    model.add(layers.Dense(64*64, activation='relu'))
    model.add(layers.Reshape((64, 64, 1)))

    # Output layer (Sigmoid for binary mask)
    model.add(layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

    # Build the model
    model = build_model()
    model.summary()
    
    import numpy as np
    import cv2
    import random
    from sklearn.model_selection import train_test_split
    
    def draw_incomplete_shape(img_size=64):
        img = np.ones((img_size, img_size, 1), dtype=np.uint8) * 255  # White background
        shape_type = random.choice(['line', 'circle', 'ellipse', 'rectangle', 'polygon', 'star'])

    if shape_type == 'line':
        start = (random.randint(0, img_size//2), random.randint(0, img_size))
        end = (random.randint(img_size//2, img_size), random.randint(0, img_size))
        cv2.line(img, start, end, 0, 2)  # Black line
    
    elif shape_type == 'circle':
        center = (random.randint(img_size//4, 3*img_size//4), random.randint(img_size//4, 3*img_size//4))
        radius = random.randint(img_size//8, img_size//4)
        cv2.circle(img, center, radius, 0, 2)  # Black circle
    
    elif shape_type == 'ellipse':
        center = (random.randint(img_size//4, 3*img_size//4), random.randint(img_size//4, 3*img_size//4))
        axes = (random.randint(img_size//8, img_size//4), random.randint(img_size//8, img_size//4))
        angle = random.randint(0, 180)
        cv2.ellipse(img, center, axes, angle, 0, 360, 0, 2)  # Black ellipse
    
    elif shape_type == 'rectangle':
        top_left = (random.randint(0, img_size//2), random.randint(0, img_size//2))
        bottom_right = (random.randint(img_size//2, img_size), random.randint(img_size//2, img_size))
        cv2.rectangle(img, top_left, bottom_right, 0, 2)  # Black rectangle
    
    elif shape_type == 'polygon':
        num_sides = random.randint(3, 6)
        radius = random.randint(img_size//8, img_size//4)
        center = (random.randint(img_size//4, 3*img_size//4), random.randint(img_size//4, 3*img_size//4))
    points = []
            for i in range(num_sides):
                angle = 2 * np.pi * i / num_sides
                x = int(center[0] + radius * np.cos(angle))
                y = int(center[1] + radius * np.sin(angle))
                points.append([x, y])
            points = np.array(points)
            cv2.polylines(img, [points], isClosed=True, color=0, thickness=2)  # Black polygon
    
    elif shape_type == 'star':
        points = []
        center = (random.randint(img_size//4, 3*img_size//4), random.randint(img_size//4, 3*img_size//4))
        inner_radius = random.randint(img_size//8, img_size//4)
        outer_radius = random.randint(inner_radius, img_size//3)
        for i in range(10):
            angle = i * np.pi / 5
            r = outer_radius if i % 2 == 0 else inner_radius
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            points.append([x, y])
        points = np.array(points)
        cv2.polylines(img, [points], isClosed=True, color=0, thickness=2)  # Black star

    return img
    
    def generate_synthetic_data(num_samples=1000, img_size=64):
        X = []
        y = []
        for _ in range(num_samples):
            # Generate an incomplete shape
            img = draw_incomplete_shape(img_size)

        # Create an occluded version of the shape to simulate incompleteness
        mask = np.ones((img_size, img_size, 1), dtype=np.uint8) * 255  # White background
        occlusion = np.random.randint(0, 2)  # Randomly decide to occlude or not
        
    if occlusion:
                occlusion_size = random.randint(10, img_size // 2)
                x1 = random.randint(0, img_size - occlusion_size)
                y1 = random.randint(0, img_size - occlusion_size)
                x2 = x1 + occlusion_size
                y2 = y1 + occlusion_size
                mask[y1:y2, x1:x2] = 0  # Create an occlusion area
        
        incomplete_img = cv2.bitwise_and(img, mask)

        X.append(img)
        y.append(incomplete_img)

    X = np.array(X) / 255.0  # Normalize
    y = np.array(y) / 255.0  # Normalize

    return X, y
    
    # Generate data
    X, y = generate_synthetic_data(1000)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


The code imports essential libraries for image processing, machine learning, and visualization. It utilizes OpenCV (`cv2`) for handling image data, NumPy for numerical operations, and TensorFlow/Keras for loading and building deep learning models. Matplotlib is used for plotting and visualizing data or model results. This setup is typically used for tasks like image classification or other computer vision applications.


    # Build the model
    model = build_model()
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))
    
    # Save the model in the native Keras format
    model.save('shape_completion_model.keras')
    

The code builds, compiles, and trains a deep learning model using TensorFlow/Keras. It uses the Adam optimizer and binary crossentropy loss, training for 20 epochs with a batch size of 32. The model's performance is validated with a separate validation set. Finally, the trained model is saved in Keras's native format for future use.


    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")
    
    # Test the model on some validation examples
    import matplotlib.pyplot as plt
    
    # Choose a random sample from the validation set
    sample_idx = random.randint(0, len(X_val)-1)
    example_input = X_val[sample_idx].reshape(1, 64, 64, 1)
    predicted_output = model.predict(example_input)
    
    # Display the input and output
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(X_val[sample_idx].reshape(64, 64), cmap='gray')
    plt.title("Input Image")
    
    plt.subplot(1, 3, 2)
    plt.imshow(y_val[sample_idx].reshape(64, 64), cmap='gray')
    plt.title("Ground Truth")
    
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_output.reshape(64, 64), cmap='gray')
    plt.title("Predicted Completion")
    plt.show()

    

The code evaluates the trained model on a validation set, printing the loss and accuracy. It then randomly selects a sample from the validation set to visualize the model's performance. The input image, ground truth, and the model's predicted output are displayed side by side using Matplotlib, showcasing the model's ability to complete or predict the missing parts of the image.



    # Load your pre-trained model (replace with your model's path)
    model = load_model('shape_completion_model.keras')
    
    # Function to process image and detect open shapes
    def preprocess_image(image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, edges
    # Function to identify open shapes
    def identify_open_shapes(contours, edges):
        open_shapes = []
        for contour in contours:
            # Approximate the contour
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, False)  # False means open shape
            # Check if the shape is open by checking endpoints
            if len(approx) > 2:
                open_shapes.append(approx)
        return open_shapes
    
    # Function to predict and complete open shapes
    def complete_open_shapes(image, open_shapes):
        completions = []
        for shape in open_shapes:
            # Extract bounding box of the shape
            x, y, w, h = cv2.boundingRect(shape)
            roi = image[y:y+h, x:x+w]
            
        # Check if roi has 3 channels
        if roi.ndim == 3 and roi.shape[2] == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gray, (64, 64))  # Assuming model input is 64x64
            roi_resized = roi_resized[:, :, np.newaxis]  # Add channel dimension
        else:
            roi_resized = cv2.resize(roi, (64, 64))  # If already grayscale
        roi_resized = roi_resized / 255.0  # Normalize
        roi_resized = np.expand_dims(roi_resized, axis=0)
        
        # Predict the completion
        completion = model.predict(roi_resized)
        completion = completion.reshape(64, 64)
        completion = (completion * 255).astype(np.uint8)
     completion = cv2.resize(completion, (w, h))  # Resize back to original size
        
        # Create output by overlaying completion on original image
        if roi.ndim == 3 and roi.shape[2] == 3:
            completion_colored = cv2.cvtColor(completion, cv2.COLOR_GRAY2BGR)
            overlay = image.copy()
            overlay[y:y+h, x:x+w] = completion_colored
        else:
            overlay = image.copy()
            overlay[y:y+h, x:x+w] = np.stack([completion]*3, axis=-1)  # Convert grayscale to BGR

        completions.append(overlay)
    
    return completions


    
    # Function to display suggestions
    def display_suggestions(original_image, completions):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, len(completions) + 1, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
    
    for i, completion in enumerate(completions):
        plt.subplot(1, len(completions) + 1, i + 2)
        plt.imshow(cv2.cvtColor(completion, cv2.COLOR_BGR2RGB))
        plt.title(f"Suggestion {i + 1}")
    
    plt.show()

    

The code loads a pre-trained model to detect and complete open shapes in an image. It processes the image to find contours and identify open shapes, then uses the model to predict and fill in the missing parts. The completed shapes are overlaid on the original image, and multiple completion suggestions are displayed using Matplotlib.


    
    # Main Function to execute the logic
    def main(image_path):
        # Load image
        image = cv2.imread(image_path)
        
    # Preprocess and identify open shapes
    contours, edges = preprocess_image(image)
    open_shapes = identify_open_shapes(contours, edges)
    
    # Complete open shapes using the model
    completions = complete_open_shapes(image, open_shapes)
    
    # Display suggestions
    display_suggestions(image, completions)
    
    # Example usage
    if __name__ == "__main__":
        main('incomplete_star.png')

The `main` function loads an image, identifies any open shapes, completes them using a pre-trained model, and displays the original image alongside the completed versions. It’s designed to process an image file (e.g., `incomplete_star.png`) and visualize possible shape completions.



