import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('whiteBoard.jpg')

clone = img.copy()
points = []

def select_point(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Click on the 4 corners of the whiteboard (top left → top right → bottom left → bottom right)", clone)

# mode klik
cv2.imshow("Click on the 4 corners of the whiteboard (top left → top right → bottom left → bottom right)", clone)
cv2.setMouseCallback("Click on the 4 corners of the whiteboard (top left → top right → bottom left → bottom right)", select_point)

print("Click the 4 corners of the whiteboard in sequence.")
print("Press any button in the image window when finished.")

cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) != 4:
    raise ValueError(f"You choose {len(points)} dot. Must be 4 dots!")

pts1 = np.float32(points)
print("✅ Selected point:", pts1)

width, height = 700, 450
pts2 = np.float32([
    [0, 0],
    [width, 0],
    [0, height],
    [width, height]
])

M = cv2.getPerspectiveTransform(pts1, pts2)
warped = cv2.warpPerspective(img, M, (width, height))

cv2.imshow("Original image", img)
cv2.imshow("Straighten Whiteboard", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("straightWhiteboard.jpg", warped)
print("💾 Result saved as 'straightWhiteboard.jpg'")