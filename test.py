import cv2 
import numpy as np
import math
def getKeypointInfo(keypoints):
    for item in keypoints:
        print(item.pt)
        print(item.size)
        print(item.angle)
        print(item.response)
        print(item.octave)
        print(item.class_id)
        break
        
    
    return 0

def cylindrical_projection(img, focal_length):
    height, width, _ = img.shape
    cylinder_proj = np.zeros(shape=img.shape, dtype=np.uint8)
    
    for y in range(-int(height/2), int(height/2)):
        for x in range(-int(width/2), int(width/2)):
            cylinder_x = focal_length*math.atan(x/focal_length)
            cylinder_y = focal_length*y/math.sqrt(x**2+focal_length**2)
            
            cylinder_x = round(cylinder_x + width/2)
            cylinder_y = round(cylinder_y + height/2)

            if cylinder_x >= 0 and cylinder_x < width and cylinder_y >= 0 and cylinder_y < height:
                cylinder_proj[cylinder_y][cylinder_x] = img[y+int(height/2)][x+int(width/2)]
    
    # Crop black border
    # ref: http://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    _, thresh = cv2.threshold(cv2.cvtColor(cylinder_proj, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
        
    return cylinder_proj[y:y+h, x:x+w]

def matcherx(key1, des1, key2, des2):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(des1, des2, 2)
    matches = []
    ratio = 0.75
    for m in rawMatches:

        if len(m) == 2 and m[0].distance < m[1].distance*ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))


    if len(matches)>4:
        pointsA = np.float32([key1[i].pt for (_,i) in matches])
        pointsB = np.float32([key2[i].pt for (i,_) in matches])


        (H, status) = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, ransacReprojThreshold= 4.0)

        print("Homography is: \n", H)

        return (matches, H, status)





    return (None,None,None)


def func():
    test1 = cv2.imread('data/parrington/prtn00.jpg')
    test2 = cv2.imread('data/parrington/prtn01.jpg')
    test1 = cylindrical_projection(test1, 704)
    test2 = cylindrical_projection(test2, 704)
    detector = cv2.ORB_create()
    
    keypoints = detector.detect(test1)
    (key1, des1) = detector.compute(test1, keypoints)

    keypoints = detector.detect(test2)
    (key2, des2) = detector.compute(test2, keypoints)

    (matchx, H, status) = matcherx(key1, des1, key2, des2)

    if (matchx,H,status) == (None,None,None):
        print("No homography discovered. Exiting ....")
        return 0
    
    result = cv2.warpPerspective(test1,
                                 H, 
                                 (test1.shape[1]+ test2.shape[1], test1.shape[0])
                                 )
    
    result[0:test2.shape[0], 0:test2.shape[1]] = test2

    cv2.imshow("Result", result)
    cv2.waitKey(0)
    


    # for i in range(0,len(key1)):
    #     print((key1[i].pt,des1[i]), flush=True)
    # getKeypointInfo(keypoints=keypoints)
    print(test1.shape)

    # cyl = cylindrical_projection(test1, 704.907)
    # cv2.imshow('1',cyl)
    # cv2.waitKey(0)
func()
