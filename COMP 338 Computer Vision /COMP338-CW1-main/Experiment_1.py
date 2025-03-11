import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolve2D_color(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the padding size
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Create a zero-padded image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Create an empty result image
    result_image = np.zeros_like(image)

    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest from the padded image
            roi = padded_image[i:i + kernel_height, j:j + kernel_width]

            # Perform element-wise multiplication and sum
            result_image[i, j] = np.sum(roi * kernel)

    return result_image

# Experiment 1: Single Channel Convolution
# Load a single-channel grayscale image
image_gray = cv2.imread('output.png', cv2.IMREAD_GRAYSCALE)

# Define a simple blur kernel
kernel = np.ones((50, 50), np.float32) / 2500.0

# Convert image to float32 for better precision during convolution
image_float32 = image_gray.astype(np.float32)

# Apply convolve2D_color function
result_convolve2D_color = convolve2D_color(image_float32, kernel)

# Apply cv2.filter2D function
result_filter2D = cv2.filter2D(image_float32, -1, kernel)

# Display the original, convolved by convolve2D_color, and convolved by cv2.filter2D images
plt.figure(figsize=(10, 3))

plt.subplot(1, 3, 1)
plt.imshow(image_gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(result_convolve2D_color, cmap='gray')
plt.title('Convolved by convolve2D_color')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(result_filter2D, cmap='gray')
plt.title('Convolved by cv2.filter2D')
plt.axis('off')

plt.show()
