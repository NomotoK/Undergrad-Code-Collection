import cv2
from matplotlib import pyplot as plt
import numpy as np

image = cv2.imread('/Users/hailin/Documents/COMP338_Labs/output.png')

image = cv2.resize(image, (200, 200))

cv2.imshow('image', image)
# cv2.waitKey(0)

# convert the image to grayscale
output = image.copy()
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
cv2.imshow('image', gray)
# cv2.waitKey(0)

# threshold the image by setting all pixel values less than 225
# to 255 (white; foreground) and all pixel values >= 225 to 255
# (black; background), thereby segmenting the image
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow('image', thresh)
# cv2.waitKey(0)

ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(gray,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [gray, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
# plt.show()


#apply the convolution operation to the image:
img = image.copy()

kernel = np.ones((5,5),np.float32)/25 # define the kernel/filter
dst = cv2.filter2D(img,-1,kernel)     # apply the filter

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
# plt.show()

img = gray.copy()

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# plt.show()



# Convert the image to YUV color space
yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
# Split the YUV image into Y, U, and V channels
y_channel, u_channel, v_channel = cv2.split(yuv_image)
# Create subplots to display the channels
plt.figure(figsize=(12, 4))

# Display the Y (luminance) channel
plt.subplot(131)
plt.imshow(y_channel, cmap='gray')
plt.title('Y (Luminance)')
plt.axis('off')

# Display the U (chrominance blue) channel
plt.subplot(132)
plt.imshow(u_channel, cmap='gray')
plt.title('U (Chrominance Blue)')
plt.axis('off')

# Display the V (chrominance red) channel
plt.subplot(133)
plt.imshow(v_channel, cmap='gray')
plt.title('V (Chrominance Red)')
plt.axis('off')
# plt.show()




# Define different kernel sizes
kernel_sizes = [3, 5, 7, 9, 11]

# Create subplots to display the filtered results
plt.figure(figsize=(15, 5))

for i, kernel_size in enumerate(kernel_sizes):
    # Apply Gaussian blur with the current kernel size
    filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Display the filtered image
    plt.subplot(1, 5, i + 1)
    plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Kernel Size {kernel_size}x{kernel_size}')
    plt.axis('off')

plt.show()