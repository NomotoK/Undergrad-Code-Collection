import cv2
import numpy as np

img = cv2.imread('colour-200dpi_page-0009.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(3,3),0)
edges = cv2.Canny(blur,50,100)

countours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, countours, -1, (0,255,0), 3)

# contract window
# cv2.namedWindow('img', cv2.WINDOW_NORMAL)
# cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
cv2.imshow('img',img)
cv2.imshow('edges',edges)
cv2.waitKey(0)