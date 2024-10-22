import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    norm_img = 255 * (image - min_val) / (max_val - min_val)
    return norm_img.astype(np.uint8)

def pad_image(image, pad_height, pad_width):
    return np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

def showImages(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')
    plt.show()

def doCorrrelation(imageFile, maskFile):
    # Open Files as np arrays
    image = np.array((Image.open(imageFile)).convert('L'), dtype=np.float32)
    mask = np.array((Image.open(maskFile)).convert('L'), dtype=np.float32)

    img_height, img_width = image.shape
    mask_height, mask_width = mask.shape

    # Pad the image to handle borders
    padded_image = pad_image(image, mask_height // 2, mask_width // 2)

    result = np.zeros((img_height, img_width), dtype=np.float32)

    # Perform correlation by sliding the mask over the image
    for i in range(img_height):
        for j in range(img_width):
            # Extract the region of interest from the padded image
            region = padded_image[i:i + mask_height, j:j + mask_width]
            # Perform element-wise multiplication and sum the result
            result[i, j] = np.sum(region * mask)

    # Normalize the correlation result to [0, 255]
    result_norm = normalize(result)


    imageArray = [image, mask, result_norm]
    titles = ["1", "2", "3"]
    showImages(imageArray, titles)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Converter')
    parser.add_argument('-f','--image_file', type=str, default = "Image.pgm", help='path to image file')
    parser.add_argument('-m','--mask_file', type=str, default = "Pattern.pgm", help='path to mask file')
    args = parser.parse_args()
    doCorrrelation(args.image_file, args.mask_file)