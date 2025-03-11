import cv2
import matplotlib.pyplot as plt

# Load the color images
image1 = cv2.imread('victoria1.jpg')
image2 = cv2.imread('victoria2.jpg')

# Convert color images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = orb.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray_image2, None)

# Reduce the number of keypoints for visualization
num_keypoints_to_draw = 50
keypoints1 = sorted(keypoints1, key=lambda x: x.response, reverse=True)[:num_keypoints_to_draw]
keypoints2 = sorted(keypoints2, key=lambda x: x.response, reverse=True)[:num_keypoints_to_draw]

# Create images with reduced keypoints drawn
image1_with_keypoints = cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image2_with_keypoints = cv2.drawKeypoints(image2, keypoints2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Visualize the images with reduced keypoints
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image1_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('Image 1 with ORB Keypoints')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image2_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('Image 2 with ORB Keypoints')
plt.axis('off')

plt.show()

