import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt



def doGradient():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Converter')
    parser.add_argument('-f','--image_file', type=str, default = "peppers.png", help='path to image file')
    args = parser.parse_args()
    doCorrrelation(args.image_file)