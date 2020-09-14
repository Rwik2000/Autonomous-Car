import cv2
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib
from scipy.linalg import pascal

def bezier_coordinates(coordinates_left, coordinates_right, img,num_points):
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
    bezier_plot(left_bez_coord,right_bez_coord,img,20)
    # print(right_bez_coord)
def bezier_plot(coordinate_left,coordinate_right,img,points):
    # LEFT
    h, w = img.shape
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
    
    mask=np.zeros((h,w))
    # print(p_x_left,p_y_left)
    for i in range(len(p_x_left)-1):
        mask=cv2.line(mask,(int(p_x_left[i]),int(p_y_left[i])),(int(p_x_left[i+1]),int(p_y_left[i+1])),(255,255,255))
    
    for i in range(len(p_x_right)-1):
        mask=cv2.line(mask,(int(p_x_right[i]),int(p_y_right[i])),(int(p_x_right[i+1]),int(p_y_right[i+1])),(255,255,255))
    
    cv2.imshow("mkas",mask)
    # cv2.waitKey(0)
    # coordinate_left=np.array(coordinate_left)
    # coordinate_right=np.array(coordinate_right)
    # plt.plot(coordinate_left[:,0],coordinate_left[:,1])
    
    # plt.plot(p_x_left,p_y_left)
    # plt.plot(p_x_right,p_y_right)
    # plt.plot(coordinate_right[:,0],coordinate_right[:,1])
    # plt.xlim(0, w)
    # plt.ylim(0, h)
    # plt.gca().invert_yaxis()
    # plt.show()
# c_l=[[1,1],[2,1],[3,3],[4,4],[5,5],[6,6]]   
# c_r=0
# img=np.array((100,100,3))
# bezier(c_l,c_r,img,10)

