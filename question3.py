import numpy as np
import argparse
from PIL import Image
import random
import matplotlib.pyplot as plt

def showImages(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')
    plt.show()

def doCorrection(filename, filterSize, noisePercent):
    if not filename.lower().endswith('.png'):
        filename += '.png'
    # Converts the png file to PGM
    image = np.array((Image.open(filename)).convert('L'), dtype=np.float32)
    
    # Add Salt Pepper noise
    corruptImage = addNoise(image, noisePercent)

    medianImage = doMedian(corruptImage, filterSize)
    meanImage = doMean(corruptImage, filterSize)

    imageArray = [image, corruptImage, medianImage, meanImage]
    titles = ["Original Image", "Corrupted Image", "Median Filtered", "Averaging Filtered"]
    showImages(imageArray, titles)

def doMedian(image, filterSize):
    padded_image = np.pad(image, pad_width=filterSize // 2, mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + filterSize, j:j + filterSize]
            filtered_image[i, j] = getMedian(region) 
    
    return filtered_image

def doMean(image, filterSize):
    padded_image = np.pad(image, pad_width=filterSize // 2, mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + filterSize, j:j + filterSize]
            filtered_image[i, j] = getMean(region)

    return filtered_image

def getMedian(array):
    newArray = np.sort(array, None) # I could make my own sorting function but this isn't a class on sorting algorithms, so I used the built in function
    return newArray[newArray.size // 2]
    

def getMean(array):
    total = 0
    for row in array:
        for num in row:
            total += num

    return (total // array.size)

    

def addNoise(image, noisePercent):
    newImage = np.copy(image)
    num = int(noisePercent * image.size / 100)
    for _ in range(num):
        x, y = random.randint(0, image.shape[0] - 1), random.randint(0, image.shape[1] - 1)
        newImage[x, y] = random.choice([0, 255])  # Set pixel to black or white
    
    return newImage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Converter')
    parser.add_argument('-f','--image_file', type=str, default = "lenna.png", help='path to image file')
    parser.add_argument('-s','--filterSize', type=int, default = 7, help='n by n for the mask sizes')
    parser.add_argument('-n','--noisePercent', type=int, default = 30, help='What percent of the image should be corrupted')
    args = parser.parse_args()
    doCorrection(args.image_file, args.filterSize, args.noisePercent)