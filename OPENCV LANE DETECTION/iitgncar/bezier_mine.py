import cv2
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib
from scipy.linalg import pascal
from .camera_transf import cam_world,cam_tp,cam_tp1

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
    turn=bezier_plot(left_bez_coord,right_bez_coord,mask_image,20,og_img)
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
    h1=h+int(h/2)
    mask1=np.zeros((h,w))
    top_coord_left=np.zeros((len(p_y_left),2))
    top_coord_right=np.zeros((len(p_y_right),2))
    # print(p_x_left,p_y_left)
    print("\n")
    x_prev,z_prev=0,0
    top_coord_left[len(p_x_left)-1][0],top_coord_left[len(p_x_left)-1][1]=cam_world(int(p_x_left[len(p_x_left)-1]),int(p_y_left[len(p_x_left)-1]),mask1)
    top_coord_right[len(p_x_right)-1][0],top_coord_right[len(p_x_right)-1][1]=cam_world(int(p_x_right[len(p_x_right)-1]),int(p_y_right[len(p_x_right)-1]),mask1)
    center=np.array([(top_coord_left[-1][0]+top_coord_right[-1][0])/2,(top_coord_left[-1][1]+top_coord_right[-1][1])/2])
    check=0
    turn="Straight"
    left_cut=[0,0]
    right_cut=[0,0]
    for i in range(len(p_x_left)-1):
        # top_coord_left[i][0],top_coord_left[i][1]=cam_world(int(p_x_left[i]),int(p_y_left[i]),mask1)
        top_coord_left[i][0],top_coord_left[i][1]=cam_tp(int(p_x_left[i]),int(p_y_left[i]),mask1)
        if i>0 and (top_coord_left[i][0]-center[0])*(top_coord_left[i-1][0]-center[0])<0:
            if check==0:
                left_cut=(center[0],(top_coord_left[i][1]+top_coord_left[i-1][0])/2)
            check=1
        # if i>0:
        #     mask2=cv2.line(mask2,(int(x_new*20+w/2),int(800-z_new*20)),(int(x_prev*20+w/2),int(800-z_prev*20)),(255,255,255))
        mask1=cv2.line(mask1,(int(p_x_left[i]),int(p_y_left[i])),(int(p_x_left[i+1]),int(p_y_left[i+1])),(255,255,255))
        
    # top_coord_left[i+1][0],top_coord_left[i+1][1]=cam_world(int(p_x_left[i+1]),int(p_y_left[i+1]),mask1)
    check=0
    
    for i in range(len(p_x_right)-1):
        # top_coord_right[i][0],top_coord_right[i][1]=cam_world(int(p_x_right[i]),int(p_y_right[i]),mask1)
        top_coord_right[i][0],top_coord_right[i][1]=cam_tp(int(p_x_right[i]),int(p_y_right[i]),mask1)
        if i>0 and (top_coord_right[i][0]-center[0])*(top_coord_right[i-1][0]-center[0])<0:
            if check==0:
                right_cut=(center[0],(top_coord_right[i][1]+top_coord_right[i-1][0])/2)
            check=1
        # if i>0:
        #     mask2=cv2.line(mask2,(int(x_new*20+w/2),int(800-z_new*20)),(int(x_prev*20+w/2),int(800-z_prev*20)),(255,255,255))
        mask1=cv2.line(mask1,(int(p_x_right[i]),int(p_y_right[i])),(int(p_x_right[i+1]),int(p_y_right[i+1])),(255,255,255))
    abra,kadabra=cam_tp1(p_x_right,p_y_right,mask1)
    print(abra,kadabra)
    print(top_coord_right)
    if right_cut[1]>left_cut[1]:
        turn="Left"
    elif right_cut[1]<left_cut[1]:
        turn="Right"
    
    mask1=cv2.putText(mask1,turn,(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255) )

    # print(left_cut,right_cut)
    # print()
    # print(top_coord_right)
    
    # print(center)
    # print(top_coord_left[0][0],top_coord_left[-1][0])
    # print(top_coord_right[0][0],top_coord_right[-1][0])
    print(turn)
    mask1=cv2.resize(mask1,(int((w+2)/2),int((h+2)/2)))
    cv2.imshow("mask1",mask1)
    if turn=="Right":
        # print("bezier"+str(1))
        return(1)
    elif turn=="Left":
        # print("bezier"+str(-1))
        return(-1)
    else:
        # print("bezier"+str(0))
        return(0)
    