import numpy as np
import cv2
import matplotlib.pyplot as plt

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

# Experiment 2: Color Image Convolution
# Load a color image for testing
image = cv2.imread('output.png')

# Convert image to float32 for better precision during convolution
image_float32 = image.astype(np.float32)

# Define a convolution kernel (edge detection)
kernel = np.array([[1, 1, 1],
                   [1, -7, 1],
                   [1, 1, 1]])

# Normalize the kernel
kernel = kernel / np.sum(np.abs(kernel))

# Apply the implemented convolve2D_color function
result_convolve2D = convolve2D_color(image_float32, kernel)

# Apply the cv2.filter2D function
result_filter2D = cv2.filter2D(image_float32, -1, kernel)

# Convert the results to uint8 for display
result_convolve2D_display = result_convolve2D.astype(np.uint8)
result_filter2D_display = result_filter2D.astype(np.uint8)

# Display the original, convolved (implemented), and convolved (cv2.filter2D) images
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(result_convolve2D_display, cv2.COLOR_BGR2RGB))
plt.title('Convolved (Implemented)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result_filter2D_display, cv2.COLOR_BGR2RGB))
plt.title('Convolved (cv2.filter2D)')
plt.axis('off')

plt.show()

