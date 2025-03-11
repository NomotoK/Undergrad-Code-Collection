import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


# Load the color images
image1 = cv2.imread('victoria1.jpg')
image2 = cv2.imread('victoria2.jpg')

# Convert color images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT and ORB detectors
sift = cv2.SIFT_create()
orb = cv2.ORB_create()

# Benchmarking SIFT
start_time_sift = time.time()
keypoints_sift1, descriptors_sift1 = sift.detectAndCompute(gray_image1, None)
keypoints_sift2, descriptors_sift2 = sift.detectAndCompute(gray_image2, None)
elapsed_time_sift = time.time() - start_time_sift

# Benchmarking ORB
start_time_orb = time.time()
keypoints_orb1, descriptors_orb1 = orb.detectAndCompute(gray_image1, None)
keypoints_orb2, descriptors_orb2 = orb.detectAndCompute(gray_image2, None)
elapsed_time_orb = time.time() - start_time_orb

# Print benchmark results
print(f"SIFT Keypoint Detection and Descriptor Computation Time: {elapsed_time_sift:.4f} seconds")
print(f"ORB Keypoint Detection and Descriptor Computation Time: {elapsed_time_orb:.4f} seconds")

# Perform keypoint matching using Brute-Force Matcher (similar to the previous code)
bf = cv2.BFMatcher()
matches_sift = bf.knnMatch(descriptors_sift1, descriptors_sift2, k=2)
matches_orb = bf.knnMatch(descriptors_orb1, descriptors_orb2, k=2)

#visualization
good_matches_sift = []
for m, n in matches_sift:
    if m.distance < 0.75 * n.distance:
        good_matches_sift.append(m)


good_matches_orb = []
for m, n in matches_orb:
    if m.distance < 0.75 * n.distance:
        good_matches_orb.append(m)

# Draw keypoint matches with color visualizations
image_matches_sift = cv2.drawMatches(image1, keypoints_sift1, image2, keypoints_sift2, good_matches_sift, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
image_matches_orb = cv2.drawMatches(image1, keypoints_orb1, image2, keypoints_orb2, good_matches_orb, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the images with keypoint matches and benchmark results
plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image_matches_sift, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoint Matches')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(image_matches_orb, cv2.COLOR_BGR2RGB))
plt.title('ORB Keypoint Matches')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.bar(['SIFT', 'ORB'], [elapsed_time_sift, elapsed_time_orb], color=['blue', 'green'])
plt.title('Keypoint Detection and Descriptor Computation Time')
plt.ylabel('Time (seconds)')

plt.tight_layout()
plt.show()
