
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import matcher

def transformCylindrical():
    return 0


def extractFeatureDescriptions(image, detector):
    'Generate a tuple of Keypoints'
    keypoints = detector.detect(image, None)
    (key,des) = detector.compute(image, keypoints)
    return (key, des)


def removeOutliars():
    """Remove outliars with RANSAC"""

    return 0

def stitchMe(imgA, imgB):
    """
    Stitches two given images with the right keypoints
    and returns the combined image
    """

    detector = cv2.ORB_create(nfeatures=200, scoreType=cv2.ORB_FAST_SCORE)


    (keypointsA, descriptorsA)  = extractFeatureDescriptions(imgA, detector)
    
    (keypointsB, descriptorsB) = extractFeatureDescriptions(imgB, detector)
    
    matcher.matchFeatures(keypointsA, keypointsB)
    removeOutliars()
    
    kpdraw = cv2.drawKeypoints(imgB, keypointsB, None, color=(0,10,150))
    cv2.imshow('1',kpdraw)
    cv2.waitKey(0)
    
    print("well, its done")


test1 = cv2.imread('data/parrington/prtn00.jpg')
test2 = cv2.imread('data/parrington/prtn01.jpg')

stitchMe(test1, test2)


