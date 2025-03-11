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


# Example usage:
# Load a color image
image = cv2.imread('output.png')

# Convert image to float32 for better precision during convolution
image_float32 = image.astype(np.float32)

# Define a kernel (e.g., a simple blur kernel)
kernel = np.ones((50, 50), np.float32) / 2500.0


# Perform 2D convolution on color image
result = convolve2D_color(image_float32, kernel)

# Display the original and convolved images using Matplotlib
plt.figure(figsize=(8, 4))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')


# Convolved image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title('Convolved Image')

plt.show()

