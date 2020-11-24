import cv2  
import numpy as np  

def pixel_film(image_coord,img_shape):
    x=image_coord[0]-int(img_shape[1]/2)
    y=-image_coord[1]+int(img_shape[0]/2)-30
    # print(x,y)
    return((x,y))
    # print(x,y)

def film_world(film_coord,f=1000,Y=-1):
    Z=f*Y/film_coord[1]
    X=film_coord[0]*Y/film_coord[1]
    return(X,Z)

def cam_world(x,y,image):
    img_shape=image.shape
    X,Z=film_world(pixel_film((x,y),img_shape))
    # print(X,Z)
    return(X,Z)
# print(film_world(pixel_film((400,299),img_shape)))