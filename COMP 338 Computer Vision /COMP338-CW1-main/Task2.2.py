import cv2
import numpy as np
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

# Detect keypoints and compute descriptors using SIFT
keypoints_sift1, descriptors_sift1 = sift.detectAndCompute(gray_image1, None)
keypoints_sift2, descriptors_sift2 = sift.detectAndCompute(gray_image2, None)

# Detect keypoints and compute descriptors using ORB
keypoints_orb1, descriptors_orb1 = orb.detectAndCompute(gray_image1, None)
keypoints_orb2, descriptors_orb2 = orb.detectAndCompute(gray_image2, None)

# Create BFMatcher (Brute-Force Matcher)
bf = cv2.BFMatcher()

# Perform keypoint matching using SIFT descriptors
matches_sift = bf.knnMatch(descriptors_sift1, descriptors_sift2, k=2)

# Perform keypoint matching using ORB descriptors
matches_orb = bf.knnMatch(descriptors_orb1, descriptors_orb2, k=2)

# Apply ratio test to filter good matches
good_matches_sift = []
for m, n in matches_sift:
    # if m.distance < 0.75 * n.distance:
        good_matches_sift.append(m)

good_matches_orb = []
for m, n in matches_orb:
    # if m.distance < 0.75 * n.distance:
        good_matches_orb.append(m)

# Draw keypoint matches with color visualizations
image_matches_sift = cv2.drawMatches(image1, keypoints_sift1, image2, keypoints_sift2, good_matches_sift, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
image_matches_orb = cv2.drawMatches(image1, keypoints_orb1, image2, keypoints_orb2, good_matches_orb, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the images with keypoint matches
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_matches_sift, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoint Matches')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_matches_orb, cv2.COLOR_BGR2RGB))
plt.title('ORB Keypoint Matches')
plt.axis('off')

plt.show()
