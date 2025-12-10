import imageio as img
import numpy as np
import matplotlib.pyplot as plt

robertX = np.array([
    [1,0],
    [0,-1]
])

robertY = np.array([
    [0,1],
    [-1,0]
])

image = img.imread('file_00000000f574720980c09191e92bd56d.png', mode='F')

imgPad = np.pad(image, pad_width=1, mode='constant', constant_values=0)

Gx = np.zeros_like(imgPad)
Gy = np.zeros_like(imgPad)

for y in range (1, imgPad.shape [0]-1):
    for x in range(1, imgPad.shape[1]-1):
        area = imgPad[y:y+2, x:x+2]
        Gx[y-1,x-1] = np.sum(area * robertX)
        Gy[y-1,x-1] = np.sum(area * robertY)
        
G = np.sqrt(Gx**2 + Gy **2)
G = (G/G.max()) * 255
G = np.clip(G,0,255)
G = G.astype(np.uint8)

plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.imshow(image)

plt.subplot(2,2,2)
plt.imshow(Gx, cmap='gray')

plt.subplot(2,2,2)
plt.imshow(Gy, cmap='gray')

plt.subplot(2,2,4)
plt.imshow(G, cmap= 'gray')

plt.show()