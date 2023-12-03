import getFeatures as gf
import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import os 
import time

data_location = 'data/parrington'

def getFiles():
    'Get absolute path of all files to be stitched as a tuple'

    abs_path = os.path.abspath(data_location)
    data = [os.path.join(abs_path, f) for f in os.listdir(data_location)]
    return data

def showImg(img, autoclose = False):
    'Simply displays the image as a matplotlib plot'

    plt.figure()
    plt.imshow(img)
    plt.show(block=False)
    if autoclose == True:
        plt.pause(0.5)
        plt.close()
    return 0


def stitcher():
    'Runs a loop to stitch all the images in a sequence'

    data = getFiles()
    currentImg = cv2.imread(data[0])
    for idx, im in enumerate(data):
        if idx == 0:
            continue
        nextImg = cv2.imread(im)
        currentImg = gf.stitchMe(currentImg, nextImg)
    
    showImg(currentImg)
    print("End Program")
    return 0

if __name__ == "__main__":
    stitcher()


