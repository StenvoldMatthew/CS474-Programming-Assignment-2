import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    norm_img = 255 * (image - min_val) / (max_val - min_val)
    return norm_img.astype(np.uint8)

def showImages(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')
    plt.show()

def pad_image(image, pad_height, pad_width):
    return np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

def getAveFilter(size):
    return np.ones((size, size), dtype=np.float32) / (size * size)


def getGaussianFilter(size):
    # Getting this were harder than the coding assignment
    array = None
    if size == 7:
        array =  [[1, 1, 2, 2, 2, 1, 1],
                  [1, 2, 2, 4, 2, 2, 1],
                  [2, 2, 4, 8, 4, 2, 2],
                  [2, 4, 8, 16, 8, 4, 2],
                  [2, 2, 4, 8, 4, 2, 2],
                  [1, 2, 2, 4, 2, 2, 1],
                  [1, 1, 2, 2, 2, 1, 1]]
    if size == 15:
        array =  [[2, 2, 3, 4, 5, 5, 6, 6, 6, 5, 5, 4, 3, 2, 2],
                  [2, 3, 4, 5, 7, 7, 8, 8, 8, 7, 7, 5, 4, 3, 2],
                  [3, 4, 6, 7, 9, 10, 10, 11, 10, 10, 9, 7, 6, 4, 3],
                  [4, 5, 7, 9, 10, 12, 13, 13, 13, 12, 10, 9, 7, 5, 4],
                  [5, 7, 9, 11, 13, 14, 15, 16, 15, 14, 13, 11, 9, 7, 5],
                  [5, 7, 10, 12, 14, 16, 17, 18, 17, 16, 14, 12, 10, 7, 5],
                  [6, 8, 10, 13, 15, 17, 19, 19, 19, 17, 15, 13, 10, 8, 6],
                  [6, 8, 11, 13, 16, 18, 19, 20, 19, 18, 16, 13, 11, 8, 6],
                  [6, 8, 10, 13, 15, 17, 19, 19, 19, 17, 15, 13, 10, 8, 6],
                  [5, 7, 10, 12, 14, 16, 17, 18, 17, 16, 14, 12, 10, 7, 5],
                  [5, 7, 9, 11, 13, 14, 15, 16, 15, 14, 13, 11, 9, 7, 5],
                  [4, 5, 7, 9, 10, 12, 13, 13, 13, 12, 10, 9, 7, 5, 4],
                  [3, 4, 6, 7, 9, 10, 10, 11, 10, 10, 9, 7, 6, 4, 3],
                  [2, 3, 4, 5, 7, 7, 8, 8, 8, 7, 7, 5, 4, 3, 2],
                  [2, 2, 3, 4, 5, 5, 6, 6, 6, 5, 5, 4, 3, 2, 2],]
        
    npArray = np.array(array, dtype=np.float32)
    # Make it so the array adds up to 1
    return npArray / np.sum(npArray)
        
def runFunction(filename):
    images = [np.array(Image.open(filename).convert('L'), dtype=np.float32)]
    titles = ["Original"]
    for i in [7, 15]:
        images.append(doGaussian(filename, i))
        titles.append(f"Gaussian Smoothing {i}x{i}")
        images.append(doAve(filename, i))
        titles.append(f"Average Smoothing {i}x{i}")

    showImages(images, titles)
        
def doAve(filename, maskSize):
    mask = getAveFilter(maskSize)
    image = np.array(Image.open(filename).convert('L'), dtype=np.float32)
    paddedImage = pad_image(image, maskSize // 2, maskSize // 2)
    output_image = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = paddedImage[i:i + maskSize, j:j + maskSize]
            output_image[i, j] = np.sum(region * mask)
    
    return output_image

def doGaussian(filename, maskSize):
    mask = getGaussianFilter(maskSize)
    image = np.array(Image.open(filename).convert('L'), dtype=np.float32)
    paddedImage = pad_image(image, maskSize // 2, maskSize // 2)
    output_image = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = paddedImage[i:i + maskSize, j:j + maskSize]
            output_image[i, j] = np.sum(region * mask)
    
    return output_image
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Converter')
    parser.add_argument('-f','--image_file', type=str, default = "sf.png", help='path to image file')
    parser.add_argument('-s','--filterSize', type=int, default = 7, help='n by n for the mask sizes')
    args = parser.parse_args()
    runFunction(args.image_file)