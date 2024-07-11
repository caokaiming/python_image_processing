import cv2
import numpy as np

# 讀取影像
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# 轉換為灰階
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 檢測輪廓
edges1 = cv2.Canny(gray1, 50, 150)
edges2 = cv2.Canny(gray2, 50, 150)

# 提取輪廓
contours1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 計算輪廓面積
contour_area1 = sum(cv2.contourArea(cnt) for cnt in contours1)
contour_area2 = sum(cv2.contourArea(cnt) for cnt in contours2)

# 計算圖像總面積
total_area1 = image1.shape[0] * image1.shape[1]
total_area2 = image2.shape[0] * image2.shape[1]

# 計算輪廓占比
contour_percentage1 = (contour_area1 / total_area1) * 100
contour_percentage2 = (contour_area2 / total_area2) * 100

# 比較輪廓占比
difference = contour_percentage2 - contour_percentage1

print(f"第一張圖的輪廓占比: {contour_percentage1:.2f}%")
print(f"第二張圖的輪廓占比: {contour_percentage2:.2f}%")
print(f"第二張圖的輪廓部分比第一張圖多了: {difference:.2f}%")

# 顯示影像和輪廓
cv2.drawContours(image1, contours1, -1, (0, 255, 0), 2)
cv2.drawContours(image2, contours2, -1, (0, 255, 0), 2)

cv2.imshow('Image 1 Contours', image1)
cv2.imshow('Image 2 Contours', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
