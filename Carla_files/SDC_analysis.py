import cv2
import numpy as np
import random
import time

import SDC_data_store
# from SDC_bezier import bezier_coordinates
from SDC_bezier import vehicle_trajectory
import PIL

def left_lane_kernel(image):
    kernel=[[0,-1,1],
            [0,-2,1],
            [0,-1,1]]
    kernel=np.array(kernel)
    kernel=kernel/2
    kernel=np.array(kernel)
    # print(kernel)
    img = cv2.filter2D(image, 0, kernel)
    cv2.imshow("left", img)
    return img


def right_lane_kernel(image):
    kernel=[[1,-1,0],
            [1,-2,0],
            [1,-1,0]]
    kernel=np.array(kernel)
    kernel=kernel/2
    kernel=np.array(kernel)
    img = cv2.filter2D(image, 0, kernel)
    cv2.imshow("right", img)
    return img
def resize(image):
    # print(image.shape)
    if(len(image.shape)==3):
        height,width,_=image.shape
    else:
        height,width=image.shape
    image=cv2.resize(image,(int(width/2),int(height/2)))
    return image
def predict(coordinates,mask):   
    
    xdata = coordinates[0][:,1]
    ydata = coordinates[0][:,0]


    z = np.polyfit(xdata, ydata, 5)
    f = np.poly1d(z)
    # print(min(coordinates[0][:,1]), max(coordinates[0][:,1]))
    t = np.arange(min(coordinates[0][:,1]), max(coordinates[0][:,1]), 1)
    # print(f(t))
    plt.plot(t, f(t))
    

def lane_separation(mask,img, point_cloud):
    right_lane=right_lane_kernel(mask)

    h,w = mask.shape
    # print(h,w)
    _ ,right_lane = cv2.threshold(right_lane, 100, 255, cv2.THRESH_BINARY) 

    left_lane=left_lane_kernel(mask)
    retval,left_lane = cv2.threshold(left_lane, 100, 255, cv2.THRESH_BINARY) 
    indices_right = list(np.where(right_lane==255))
    indices_left = list(np.where(left_lane==255))

    
    if len(indices_right[0])==0:
        # print("right nai hai")
        indices_right[1] = indices_left[1]
        indices_right[0] = [w-2]*(len(indices_left[1]))
    elif len(indices_left[0])==0:
        # print("left nai hai")
        indices_left[1] = indices_right[1]
        indices_left[0] = [0]*(len(indices_right[1]))


    # if not indices_left:
    #     print("left")
    coordinates_right = np.dstack((indices_right[0],indices_right[1]))
    coordinates_left = np.dstack((indices_left[0],indices_left[1]))

    bez_mask, coord_trajectory, trajectory, world_x_left, world_x_right,world_z_left,world_z_right= vehicle_trajectory(coordinates_left,coordinates_right,left_lane,img, point_cloud)
    # bez_mask, centre_x, centre_y=vehicle_trajectory(coordinates_left,coordinates_right,left_lane,img, point_cloud)
    # return bez_mask,centre_x,centre_y
    return bez_mask, coord_trajectory, trajectory, world_x_left,world_x_right, world_z_left,world_z_right

def floodfill(image, point_cloud):
    
    input_img=image.copy()
    # input_img=cv2.cvtColor(input_img,cv2.COLOR_RGB2GRAY)
    input_img=cv2.GaussianBlur(input_img,(11,7),0)
    og_img=input_img.copy()

    height, width,_ = og_img.shape
    val=1
    loDiff=(val, val, val, val)
    upDiff=(val, val, val, val)

    seed_points=4
    w_offset=30
    floodflags = 4
    floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)
    mask = np.zeros((height+2,width+2),np.uint8)    
    for i in range(seed_points):
        for j in range(2):
            seed=(int(width/2)+(-1)**(j)*int(w_offset)*i, height - 20)
            num,image,mask,rect = cv2.floodFill(og_img, mask, seed, (0,0,0), loDiff=loDiff, upDiff=upDiff, flags=floodflags)
            cv2.circle(image, seed, 2, (0, 255, 0), cv2.FILLED, cv2.LINE_AA) 
    
    image=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if image[height-100][int(width/2)][2]<10:
        SDC_data_store.shadow_check=0
    else:
        SDC_data_store.shadow_check=1

    
    kernel = np.ones((37,37),np.uint8)
    mask = cv2.dilate(mask,kernel,0)


    # try:
    # bez_mask,centre_x,centre_y=lane_separation(mask,image, point_cloud)
    bez_mask, coord_trajectory, trajectory, world_x_left, world_x_right , world_z_left,world_z_right=lane_separation(mask,image, point_cloud)
    
    # cv2.imshow("canny",resize(canny))
    cv2.imshow("mask",resize(mask))
    return bez_mask, coord_trajectory, trajectory, world_x_left,world_x_right, world_z_left,world_z_right
    # return bez_mask, centre_x, centre_y


