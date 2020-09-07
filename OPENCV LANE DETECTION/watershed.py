from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import glob
import imutils
import cv2

def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))
def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))
def stackAndShow(a, b):
    horiz = np.vstack((a, b))
    cv2.imshow('image', horiz)
    return cv2.waitKey(0)

def perspective_transform(img, src, dst):
    """
     Convert the vehicle’s camera view of the scene into a top-down “bird’s-eye” view.
        Parameters:
            img: Input image.
            src: Source points.
            dst: Destination points.
    """
    image_shape = img.shape
    img_size = (image_shape[1], image_shape[0])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    # Return the resulting image and matrix
    return warped, M, Minv

def transform(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (0,0), fx=2, fy=2)
    val = 20
    shifted = cv2.pyrMeanShiftFiltering(image, val, val)
    cv2.imshow('shifted', shifted)
    shifted = kmeans_color_quantization(shifted, clusters = 5)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresh", thresh)


    return stackAndShow(image, shifted)




if __name__ == "__main__":
    for path in sorted(glob.glob('inputs/*'), reverse=True):
        returnval = transform(path)
        if returnval == ord('q'):
            break