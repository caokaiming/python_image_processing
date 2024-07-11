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

# 假設我們只考慮最大的輪廓
cnt1 = max(contours1, key=cv2.contourArea)
cnt2 = max(contours2, key=cv2.contourArea)

# 計算相似度
similarity = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I1, 0.0)
print(f"輪廓相似度: {similarity}")

# 顯示影像和輪廓
cv2.drawContours(image1, [cnt1], -1, (0, 255, 0), 2)
cv2.drawContours(image2, [cnt2], -1, (0, 255, 0), 2)

cv2.imshow('Image 1 Contours', image1)
cv2.imshow('Image 2 Contours', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
