import numpy as np
import cv2

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

# Experiment 5: Edge Cases
# Setup: Use a small image (2x2 pixels) and an identity kernel

# Create a small color image
small_image = np.array([[[10, 20, 30], [40, 50, 60]],
                        [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8)

# Define an identity kernel
identity_kernel = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

#sharpen_kernel = np.array([[0, -1, 0],
#                           [-1, 5, -1],
#                           [0, -1, 0
#                           ]])

# Apply the implemented convolve2D_color function
result_impl = convolve2D_color(small_image.astype(np.float32), identity_kernel)

# Apply the cv2.filter2D function
result_cv2 = cv2.filter2D(small_image.astype(np.float32), -1, identity_kernel)

# Display the results using print()
print("Original Image:")
print(small_image)

print("\nResult using convolve2D_color:")
print(result_impl.astype(np.uint8))

print("\nResult using cv2.filter2D:")
print(result_cv2.astype(np.uint8))
