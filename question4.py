import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

def pad_image(image, pad_height, pad_width):
    return np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    norm_img = 1+((image - min_val) / (max_val - min_val+1e-8))
    return norm_img.astype(np.float32)

def showImages(images, titles):
    # Determine the number of images
    num_images = len(images)
    
    # Calculate number of rows and columns
    if num_images > 4:
        num_rows = (num_images // 4) + (num_images % 4 > 0)  # Add an extra row if there are leftovers
        num_cols = 4
    else:
        num_rows = 1
        num_cols = num_images

    # Create subplots with the calculated rows and columns
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Plot each image
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')

    # Hide any unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

def doSharpen(filename):
    # Define arrays
    prewitt_gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_gy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

    sobel_gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    laplacian_mask = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)

    image = np.array(Image.open(filename).convert('L'), dtype=np.float32)

    images = [image]
    titles = ["Original"]

    images.append(getImage(image, prewitt_gx))
    titles.append("Prewitt Gx")
    images.append(getImage(image, prewitt_gy))
    titles.append("Prewitt Gy")
    images.append(np.sqrt(images[1]**2 + images[2]**2))
    titles.append("Prewitt Magnitude")

    images.append(getImage(image, sobel_gx))
    titles.append("Sobel Gx")
    images.append(getImage(image, sobel_gy))
    titles.append("Sobel Gy")
    images.append(np.sqrt(images[4]**2 + images[5]**2))
    titles.append("Sobel Magnitude")

    images.append(getImage(image, laplacian_mask))
    titles.append("Laplacian")

    showImages(images, titles)
    images[0] = np.zeros(image.shape)
    for i in range(len(images)):
        images[i] = normalize(images[i])
    showImages(images * image, titles)

def getImage(image, array):
    mask = array
    maskSize = mask.shape[1]
    paddedImage = pad_image(image, maskSize // 2, maskSize // 2)
    outputImage = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = paddedImage[i:i + maskSize, j:j + maskSize]
            outputImage[i, j] = np.sum(region * mask)

    return outputImage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Converter')
    parser.add_argument('-f','--image_file', type=str, default = "lenna.png", help='path to image file')
    args = parser.parse_args()
    doSharpen(args.image_file)