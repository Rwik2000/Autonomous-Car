import cv2  
import numpy as np  

def pixel_film(image_coord,img_shape):
    x=image_coord[0]-int(img_shape[1]/2)
    y=-image_coord[1]+int(img_shape[0]/2)
    # print(x,y)
    return((x,y))
    # print(x,y)

def film_world(film_coord,f=0.0036,Y=-2):
    Z=f*Y/film_coord[1]
    X=film_coord[0]*Y/film_coord[1]
    return(X,Y,Z)

def cam_world(x,y,image):
    img_shape=image.shape
    X,Y,Z=film_world(pixel_film((x,y),img_shape))
    # print(hello)
    # print(X,Y,Z)
    return(X,Z)

def cam_tp(u,v,image):
    h,w=image.shape
    f=0.0036
    s=0.00018
    # sy=0.00018
    Y=-2
    
    if(v-h/2)==0:
        v=v+1
    Z=-(f/s)*Y/(v-h/2)
    X=-Y*(u-w/2)/(v-h/2)
    return(X,Z)

def cam_tp1(u_array,v_array,image):
    h,w=image.shape
    f=0.0036
    s=0.00018 
    Y=-2
    Z=-(f/s)*Y/(v_array-h/2)
    X=-Y*(u_array-w/2)/(v_array-h/2)
    return(X,Z)
# print(film_world(pixel_film((400,299),img_shape)))
