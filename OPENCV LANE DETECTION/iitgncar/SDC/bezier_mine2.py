import cv2
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib
from scipy.linalg import pascal
from .camera_transf import cam_world,cam_tp,cam_tp1
import sympy.geometry as gk
import sympy
def left_interval(search_point, array):  
    index=None
    l=len(array)
    for i in range(l-1):
        if search_point[0]<array[l-1-i] and search_point[0]>array[l-2-i]:
            index=l-1-i
    return index

def right_interval(search_point, array):  
    index=None
    for i in range(len(array)-1):
        if search_point[0]>array[i] and search_point[0]<array[i+1]:
            index=i
    return index
def bezier_coordinates(coordinates_left, coordinates_right, mask_image,num_points,og_img):
    coordinates_left=coordinates_left[0]
    n_left=len(coordinates_left)
    left_bez_coord=[]
    for i in range(num_points):
        # print(int(n_left/6))
        left_bez_coord.append(np.flip(coordinates_left[i*int(n_left/num_points)],axis=-1))

    coordinates_right=coordinates_right[0]
    n_right=len(coordinates_right)
    right_bez_coord=[]
    for i in range(num_points):
        right_bez_coord.append(np.flip(coordinates_right[i*int(n_right/num_points)],axis=-1))
    turn=bezier_plot(left_bez_coord,right_bez_coord,mask_image,40,og_img)
    return turn
    # print(right_bez_coord)
def bezier_plot(coordinate_left,coordinate_right,mask_image,points,og_img):
    # LEFT
    h, w = mask_image.shape
    n=len(coordinate_left)
    pascal_coord=pascal(n,kind='lower')[-1]
    t=np.linspace(0,1,points)
    p_x_left=np.zeros(points)
    p_y_left=np.zeros(points)
    p_x_right=np.zeros(points)
    p_y_right=np.zeros(points)
    # print(coordinate_left)
    for i in range(n):
        k=(t**(n-1-i))
        l=(1-t)**i
        p_x_left+=np.multiply(l,k)*pascal_coord[i]*coordinate_left[n-1-i][0]
        p_y_left+=np.multiply(l,k)*pascal_coord[i]*coordinate_left[n-1-i][1]
        p_x_right+=np.multiply(l,k)*pascal_coord[i]*coordinate_right[n-1-i][0]
        p_y_right+=np.multiply(l,k)*pascal_coord[i]*coordinate_right[n-1-i][1]

    

    bottom_left=[p_x_left[p_y_left==max(p_y_left)],p_y_left[p_y_left==max(p_y_left)]]
    bottom_right=[p_x_right[p_y_right==max(p_y_right)],p_y_right[p_y_right==max(p_y_right)]]
    bottom_left=[bottom_left[0][0],bottom_left[1][0]]
    bottom_right=[bottom_right[0][0],bottom_right[1][0]]
    
    bottom_centre=[int((bottom_left[0]+bottom_right[0])/2),int((bottom_left[1]+bottom_right[1])/2)]
    search_point=bottom_centre.copy()
    search_point[1]=search_point[1]-150

    mask1=np.zeros((h,w))
    for i in range(len(p_x_left)-1):
        mask1=cv2.circle(mask1,(int(p_x_left[i]),int(p_y_left[i])),1,(255,255,255),5)
        mask1=cv2.line(mask1,(int(p_x_left[i]),int(p_y_left[i])),(int(p_x_left[i+1]),int(p_y_left[i+1])),(255,255,255))
    
    for i in range(len(p_x_right)-1):
        mask1=cv2.circle(mask1,(int(p_x_right[i]),int(p_y_right[i])),1,(255,255,255),5)
        mask1=cv2.line(mask1,(int(p_x_right[i]),int(p_y_right[i])),(int(p_x_right[i+1]),int(p_y_right[i+1])),(255,255,255))
    
    left_sort=np.argsort(p_x_left)
    p_x_left=p_x_left[left_sort]
    p_y_left=p_y_left[left_sort]

    right_sort=np.argsort(p_x_right)
    p_x_right=p_x_right[right_sort]
    p_y_right=p_y_right[right_sort]

    mask1=cv2.line(mask1,tuple(bottom_centre),tuple(search_point),(255,255,255))
    steer_val=0
    if search_point[0]<max(p_x_left):
        index=left_interval(search_point,p_x_left)
        if index:
            print(p_y_left[index],p_y_left[index-1])
            if p_y_left[index]>search_point[1] or p_y_left[index-1]>search_point[1]:
                print(p_x_left[index],p_y_left[index])
                print(index)
                steer_val=1
    else:
        mask1=cv2.putText(mask1,"No intersection",(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))

    if search_point[0]>max(p_x_right):
        index=right_interval(search_point,p_x_right)
        if index:
            if p_y_right[index]>search_point[1] or p_y_right[index+1]>search_point[1]:
                print(p_x_right[index],p_y_right[index])
                print(index)
                steer_val=-1        
    else:
        mask1=cv2.putText(mask1,"No intersection",(50,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))

    mask1=cv2.circle(mask1,tuple(bottom_centre),1,(255,255,255),5)
    mask1=cv2.circle(mask1,tuple(search_point),1,(255,255,255),5)
    cv2.imshow("bezier_op",mask1)
    return steer_val
