import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

image = imageio.imread('images.jpeg')

# Jika citra RGB, ubah ke grayscale
if image.ndim == 3:
    image = np.mean(image, axis=2)

image = image.astype(float)

roberts_x = np.array([[1, 0],
                      [0, -1]])

roberts_y = np.array([[0, 1],
                      [-1, 0]])

def roberts_edge(img):
    rows, cols = img.shape
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)

    for i in range(rows - 1):
        for j in range(cols - 1):
            gx[i, j] = np.sum(img[i:i+2, j:j+2] * roberts_x)
            gy[i, j] = np.sum(img[i:i+2, j:j+2] * roberts_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    return magnitude

roberts_result = roberts_edge(image)

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

def sobel_edge(img):
    rows, cols = img.shape
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            gx[i, j] = np.sum(img[i-1:i+2, j-1:j+2] * sobel_x)
            gy[i, j] = np.sum(img[i-1:i+2, j-1:j+2] * sobel_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    return magnitude

sobel_result = sobel_edge(image)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Citra Asli")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Deteksi Tepi Roberts")
plt.imshow(roberts_result, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Deteksi Tepi Sobel")
plt.imshow(sobel_result, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
