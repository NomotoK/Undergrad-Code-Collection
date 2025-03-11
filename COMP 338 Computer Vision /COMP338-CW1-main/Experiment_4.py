import numpy as np
import cv2
import time

def convolve2D_color(image, kernel):
    image_height, image_width, num_channels = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the padding size
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Create zero-padded images for each channel
    padded_images = [np.pad(image[:, :, channel], ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
                     for channel in range(num_channels)]

    # Create an empty result image
    result_image = np.zeros_like(image)

    # Perform convolution for each channel
    for channel in range(num_channels):
        for i in range(image_height):
            for j in range(image_width):
                # Extract the region of interest from the padded image
                roi = padded_images[channel][i:i + kernel_height, j:j + kernel_width]

                # Perform element-wise multiplication and sum
                result_image[i, j, channel] = np.sum(roi * kernel)

    return result_image

# Experiment 4: Performance Benchmarking
# Load a color image for testing
image = cv2.imread('output.png')

# Define a simple kernel
kernel = np.ones((50, 50), np.float32) / 2500.0

# Convert image to float32 for better precision during convolution
image_float32 = image.astype(np.float32)

# Experiment 3: Performance Benchmarking
# Measure the execution time of the implemented convolve2D_color function
start_time = time.time()
result_convolve2D_color = convolve2D_color(image_float32, kernel)
elapsed_time_convolve2D_color = time.time() - start_time

# Measure the execution time of the cv2.filter2D function
start_time = time.time()
result_cv2_filter2D = cv2.filter2D(image_float32, -1, kernel)
elapsed_time_cv2_filter2D = time.time() - start_time

# Compare execution times
print(f"Convolve2D_color Execution Time: {elapsed_time_convolve2D_color:.4f} seconds")
print(f"cv2.filter2D Execution Time: {elapsed_time_cv2_filter2D:.4f} seconds")

