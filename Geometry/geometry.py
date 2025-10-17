import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('whiteBoard.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows, cols = img.shape[:2]
center = (cols // 2, rows // 2)
M_rotate = cv2.getRotationMatrix2D(center, 30, 1.0)
rotated = cv2.warpAffine(img_rgb, M_rotate, (cols, rows))

scaled = cv2.resize(rotated, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)

flipped = cv2.flip(scaled, 1)

clone = flipped.copy()
points = []

def select_point(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(clone, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("Click on the 4 corners of the whiteboard (top left → top right → bottom left → bottom right)", clone)

cv2.namedWindow("Click on the 4 corners of the whiteboard (top left → top right → bottom left → bottom right)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Click on the 4 corners of the whiteboard (top left → top right → bottom left → bottom right)", 800, 600)
cv2.imshow("Click on the 4 corners of the whiteboard (top left → top right → bottom left → bottom right)", clone)
cv2.setMouseCallback("Click on the 4 corners of the whiteboard (top left → top right → bottom left → bottom right)", select_point)

print("🖱️ Click on the 4 corners of the whiteboard (top left → top right → bottom left → bottom right)")
print("Press any button in the image window when finished.")

cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) != 4:
    raise ValueError(f"❌ You choose {len(points)} dot. Must be 4 dots!")

pts1 = np.float32(points)
width, height = 700, 450
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
warped = cv2.warpPerspective(flipped, M_perspective, (width, height))

titles = [
    "Original image",
    "Rotation 30°",
    "Scaling 1.2×",
    "Flipping Horizontal",
    "Perspective Transformation (Straighten)"
]
images = [img_rgb, rotated, scaled, flipped, warped]

plt.figure(figsize=(16, 9))
for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

cv2.imwrite("straight.jpg", cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
print("💾 Result saved as 'straight.jpg'")