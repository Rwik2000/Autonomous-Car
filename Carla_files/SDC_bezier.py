import cv2
import numpy as np
from scipy.linalg import pascal
import pickle
import os
import SDC_data_store
import warnings
from scipy.stats import linregress
# from roughPPO import SDC_PPO_input


def find_bezier_lane(coordinates, points):
    n=len(coordinates)

    pascal_coord=pascal(n,kind='lower')[-1]
    t=np.linspace(0,1,points)

    bezier_x=np.zeros(points)
    bezier_y=np.zeros(points)
  
    for i in range(n):
        k=(t**(n-1-i))
        l=(1-t)**i
        bezier_x+=np.multiply(l,k)*pascal_coord[i]*coordinates[n-1-i][0]
        bezier_y+=np.multiply(l,k)*pascal_coord[i]*coordinates[n-1-i][2]

    for i in range(len(bezier_x)):
        bezier_x[i] = round(bezier_x[i],3)
        bezier_y[i] = round(bezier_y[i],3)
    return bezier_x, bezier_y

def find_bezier_trajectory(coordinates, points):
    n=len(coordinates)

    pascal_coord=pascal(n,kind='lower')[-1]
    t=np.linspace(0,1,points)

    bezier_x=np.zeros(points)
    bezier_y=np.zeros(points)

    for i in range(n):
        k=(t**(n-1-i))
        l=(1-t)**i
        bezier_x+=np.multiply(l,k)*pascal_coord[i]*coordinates[n-1-i][0]
        bezier_y+=np.multiply(l,k)*pascal_coord[i]*coordinates[n-1-i][1]
    for i in range(len(bezier_x)):
        bezier_x[i] = round(bezier_x[i],3)
        bezier_y[i] = round(bezier_y[i],3)
    bezier_coordinates = np.transpose([bezier_x, bezier_y])
    # print(bezier_coordinates)
    return bezier_coordinates

def find_directon(world_x_left, world_z_left, world_x_right, world_z_right):
    # warnings.filterwarnings('error')

    if len(world_z_right) >= 2 and len(world_z_left) >= 2:
        slope1_left = (world_z_left[1] - world_z_left[0])/(world_x_left[1] - world_x_left[0])
        slope1_right = (world_z_right[1] - world_z_right[0])/(world_x_right[1] - world_x_right[0])
        slope2_left = (world_z_left[-1] - world_z_left[-2])/(world_x_left[-1] - world_x_left[-2])
        slope2_right = (world_z_right[-1] - world_z_right[-2])/(world_x_right[-1] - world_x_right[-2])

        # warnings.warn(RuntimeWarning())
    if len(world_z_right) < 2 and len(world_z_left) >= 2:
        slope1_left = (world_z_left[1] - world_z_left[0])/(world_x_left[1] - world_x_left[0])
        slope2_left = (world_z_left[-1] - world_z_left[-2])/(world_x_left[-1] - world_x_left[-2])
        slope2_right = slope2_left
        slope1_right = slope1_left
    
    if len(world_z_right) >= 2 and len(world_z_left) < 2:
        slope1_right = (world_z_right[1] - world_z_right[0])/(world_x_right[1] - world_x_right[0])
        slope2_right = (world_z_right[-1] - world_z_right[-2])/(world_x_right[-1] - world_x_right[-2])
        slope2_left = slope2_right
        slope1_left = slope1_right

    # print(slope1_left, slope1_right)
    # print(slope2_left,slope2_right) 


    direc_1 = 0
    if slope1_left*slope1_right < 0:
        direc_1 = 0
    elif slope1_left > 0 and slope1_right >= 0:
        direc_1 = -1
    elif slope1_left <=0 and slope1_right <0:
        direc_1 =1

    
    direc_2 = 0
    if slope2_left*slope2_right < 0:
        direc_2 = 0
    elif slope2_left > 0 and slope2_right >= 0:
        direc_2 = -1
    elif slope2_left <=0 and slope2_right <0:
        direc_2 =1
    
    direction = [0]*4
    if ((direc_1 == -1 or direc_1 ==0) and direc_2 == -1) or (direc_1 == -1 and direc_2 == 0):
        direction[0] =1
    elif direc_1 == -1 and (direc_2 == 0 or direc_2 == 1):
        direction[1] = 1 
    elif ((direc_1 == 1 or direc_1 ==0) and direc_2 == 1)or (direc_1 == 1 and direc_2 == 0):
        direction[2] =1
    elif direc_1 == 1 and (direc_2 == 0 or direc_2 == -1):
        direction[3] = 1 

    # [a,b,c,d] ------> types of directions
    # a start direction towards left/straight and end direction towards left
    # b start direction towards left and end direction towards right/straight
    # c start direction towards right/straight and end direction towards right
    # d start direction towards right and end direction towards left/straight
    # if everything is zero then straight line

    return direction
def find_traj_intersecton(curve_x, curve_z, vehicle_loc):

    ins_index = len(curve_x) - 1
    for i in range(len(curve_x)-1):
        # print(curve_x[i], curve_z[i])
        if (curve_x[i] - vehicle_loc[0])*(curve_x[i+1]-vehicle_loc[0]) <= 0:

            ins_index = i
            break
    # print(i, ins_index)
    intersection_x = 0
    slope = (curve_z[i] - curve_z[i+1])/(curve_x[i] - curve_x[i+1])

    intersection_y = curve_z[i] -  slope* curve_x[i]
    return((int(intersection_x), int(intersection_y)))

def find_trajectory(mask, world_x_left, world_z_left, world_x_right, world_z_right, min_points):
    print("*********")
    vehicle_loc = np.array([0,0])  
    direction = find_directon(world_x_left, world_z_left, world_x_right, world_z_right)
    intersection = None
    trajectory = []
    bez_prop = 3
    bez_num = 15
    # print(direction)
    point_R = None
    point_S =None
    # Left Turn
    if direction == [1,0,0,0]:
        intersection = find_traj_intersecton(world_x_right, world_z_right, vehicle_loc)
        point_Q = [int(((bez_prop - 1)*intersection[0] + vehicle_loc[0])/bez_prop), int(((bez_prop - 1)*intersection[1] + vehicle_loc[1])/bez_prop)]
        # print(point_Q) 
        point_R = [int(((bez_prop - 1)*intersection[0] + world_x_left[-1])/bez_prop), int(((bez_prop - 1)*intersection[1] + world_z_left[-1])/bez_prop)]         

        final_slope = linregress([world_x_left[-3:], world_z_left[-3:]])
        final_slope = round(float(final_slope.slope), 3)
        intersect_line_slope = ((world_z_left[-1] - world_z_right[-1])/(world_x_left[-1] - world_x_right[-1]))

        point_S = [None,None]

        point_S = [world_x_left[-1], world_z_left[-1]]
        bezier_trajectory = find_bezier_trajectory([vehicle_loc,point_Q, intersection, point_R, point_S], bez_num)
        np.append(trajectory,[world_x_left[-1], world_z_left[-1]])
        trajectory = bezier_trajectory
    #  Right Turn
    elif direction == [0,0,1,0]:
        intersection = find_traj_intersecton(world_x_left, world_z_left, vehicle_loc) 
        point_Q = [int(((bez_prop - 1)*intersection[0] + vehicle_loc[0])/bez_prop), int(((bez_prop - 1)*intersection[1] + vehicle_loc[1])/bez_prop)] 
        # print(point_Q)
        point_R = [int(((bez_prop - 1)*intersection[0] + world_x_right[-1])/bez_prop), int(((bez_prop - 1)*intersection[1] + world_z_right[-1])/bez_prop)] 

        final_slope = linregress([world_x_right[-3:], world_z_right[-3:]])
        # print((world_x_right[-1]-world_x_right[-2])/(world_z_right[-1] - world_z_right[-2]))
        final_slope = round(float(final_slope.slope), 3)
        intersect_line_slope = ((world_z_left[-1] - world_z_right[-1])/(world_x_left[-1] - world_x_right[-1]))

        point_S = [None,None]

        point_S = [world_x_right[-1], world_z_right[-1]]
        bezier_trajectory = find_bezier_trajectory([vehicle_loc,point_Q, intersection, point_R, point_S], bez_num)
        np.append(trajectory,[world_x_right[-1], world_z_right[-1]])
        trajectory = bezier_trajectory
    # if direction == [0,0,0,0]:
    else:
        for i in range(min_points):
            trajectory.append([0, int((world_z_left[i] + world_z_right[i])/2)])
        trajectory.append([0,400])
        trajectory.insert(0,[0,0])
        # print(trajectory)
    return trajectory, point_R, point_S

def bezier_split(trajectory_points, num_points):
    total_dist = 0
    
    # print(trajectory_points)
    for i in range(len(trajectory_points)-1):
        total_dist+=np.linalg.norm(np.array(trajectory_points[i+1]) - np.array(trajectory_points[i]))
    # print(total_dist)
    seg_len = int(total_dist/(num_points - 1))
    # print(seg_len)
    new_trajectory = []
    
    # k = 0
    start_point = trajectory_points[0]
    new_trajectory.append(np.array(list(start_point)))
    rem_dist_next_traj_pt = 0
    rem_dist_next_seg_pt = seg_len
    dist_cov = 0
    shadow_check = 0
    for i in range(len(trajectory_points)-1):
        rem_dist_next_traj_pt = np.linalg.norm(np.array(trajectory_points[i+1]) - start_point)
        check = 0
        k=0
        while check==0:
            k=k+1
            # print(k)
            if rem_dist_next_traj_pt < rem_dist_next_seg_pt:
                rem_dist_next_seg_pt = rem_dist_next_seg_pt - rem_dist_next_traj_pt
                start_point = list(trajectory_points[i+1])
                # rem_dist_next_traj_pt = 0
                check =1
            elif rem_dist_next_traj_pt == rem_dist_next_seg_pt:
                rem_dist_next_seg_pt = seg_len
                # rem_dist_next_traj_pt = 0
                trajectory_points[i+1]= [round(x,3) for x in trajectory_points[i+1]]
                new_trajectory.append(np.array(list(trajectory_points[i+1])))
                start_point = list(trajectory_points[i+1])
                check =1
            else: 
                l = rem_dist_next_seg_pt
                m = rem_dist_next_traj_pt - l
                rem_dist_next_traj_pt = m      
                rem_dist_next_seg_pt = seg_len
                req_point = [(l*trajectory_points[i+1][0] + m*start_point[0])/(l+m),
                            (l*trajectory_points[i+1][1] + m*start_point[1])/(l+m)]
                
                if req_point != list(start_point):
                    start_point = req_point
                    req_point = [round(x,3) for x in req_point]
                    new_trajectory.append(np.array(req_point))
                else:
                    new_trajectory = []
                    for j in range(num_points):
                        new_trajectory.append(np.array([0,10*(j+1)]))
                    shadow_check =1
                    break
                # error correction for shadow shit
        if shadow_check == 1:
            break
    # print(new_trajectory)
    return np.array(new_trajectory)
def bezier_plot(left_world_coord,right_world_coord,mask_image,points):
    # print(points)
    h, w = mask_image.shape    
    
    z_limit=200

    ''' 
    Loading previous 2 frames
    '''
    if SDC_data_store.count==0:
        SDC_data_store.coord_left_prev1=left_world_coord
        SDC_data_store.coord_right_prev1=right_world_coord
    else:
        temp_left=SDC_data_store.coord_left_prev1
        temp_right=SDC_data_store.coord_right_prev1
        SDC_data_store.coord_left_prev1=left_world_coord
        SDC_data_store.coord_right_prev1=right_world_coord
        left_world_coord=left_world_coord+temp_left
        right_world_coord=right_world_coord+temp_right

    left_world_coord=sorted(left_world_coord, key=lambda x: x[2])
    right_world_coord=sorted(right_world_coord, key=lambda x: x[2])
    
    # print(left_world_coord)
    
    world_x_left, world_z_left = find_bezier_lane(left_world_coord, points)
    world_x_right, world_z_right = find_bezier_lane(right_world_coord, points)
    centre_x=np.zeros(points)
    centre_y=np.zeros(points)


    mask2=np.zeros((h,w))
    
    world_x_left=world_x_left[world_z_left<z_limit]
    world_z_left=world_z_left[world_z_left<z_limit]
    
    world_x_right=world_x_right[world_z_right<z_limit]
    world_z_right=world_z_right[world_z_right<z_limit]
    x_mag_fac = 2
    z_mag_fac = 2

    min_points=min(points, len(world_z_left), len(world_z_right))
    trajectory, point_R, point_S= find_trajectory(mask2, world_x_left, world_z_left, world_x_right, world_z_right, min_points)
    # print(trajectory)
    trajectory = bezier_split(trajectory, 20)
    # SDC_PPO_input([list(world_x_left), list(world_z_left)],
    #               [list(world_x_right),list(world_z_right)],
    #                list(trajectory))

    # print(len(trajectory))
    # exit()
    mask2 = cv2.line(mask2, (int(w/2 - x_mag_fac*world_x_left[-1]), h+int(-world_z_left[-1]) ),(int(w/2 - x_mag_fac*world_x_right[-1]), h+int(-world_z_right[-1])),(255,255,255))
    
    if point_R!=None:
        mask2=cv2.circle(mask2,(int(w/2 - x_mag_fac*point_R[0]),h+int(-point_R[1])),5,(255,255,255),2)
        mask2=cv2.circle(mask2,(int(w/2 - x_mag_fac*point_S[0]),h+int(-point_S[1])),5,(255,255,255),4)
    # print(trajectory)
    
    i = 0
    for i in range(min_points-1):

        mask2=cv2.circle(mask2,(int(w/2 - x_mag_fac*world_x_left[i]),h+int(-world_z_left[i])),1,(255,255,255),2)
        mask2=cv2.circle(mask2,(int(w/2 - x_mag_fac*world_x_right[i]),h+int(-world_z_right[i])),1,(255,255,255),2)    
        # if i< len(trajectory)-1:
            # print(trajectory[i][0], )
            
        centre_x[i]=int(w/2-x_mag_fac*(world_x_left[i]+world_x_right[i])/2)
        centre_y[i]=h+int(-(world_z_left[i]+world_z_right[i])/2)   
    
    open_cv_trajectory=[]
    for j in range(len(trajectory)-1):
        mask2 = cv2.circle(mask2,(int(w/2 - x_mag_fac*trajectory[j][0]),h+int(-trajectory[j][1])),1,(255,255,255),2)
        mask2 = cv2.line(mask2,(int(w/2 - x_mag_fac*trajectory[j][0]), h+int(-trajectory[j][1])), 
                                    (int(w/2 - x_mag_fac*trajectory[j+1][0]), h+int(-trajectory[j+1][1])), (255,255,255))
        open_cv_trajectory.append([int(w/2 - x_mag_fac*trajectory[j][0]), h+int(-trajectory[j][1])])
        open_cv_trajectory.append([int(w/2 - x_mag_fac*trajectory[j+1][0]), h+int(-trajectory[j+1][1])])

    centre_x[i+1]=int(w/2-x_mag_fac*(world_x_left[i+1]+world_x_right[i+1])/2)
    centre_y[i+1]=h+int(-(world_z_left[i+1]+world_z_right[i+1])/2)
    mask2=cv2.circle(mask2,(int(w/2-x_mag_fac*world_x_left[i+1]),h+int(-world_z_left[i+1])),1,(255,255,255),5)
    mask2=cv2.circle(mask2,(int(w/2-x_mag_fac*world_x_right[i+1]),h+int(-world_z_right[i+1])),1,(255,255,255),5)   
    

    mask2=cv2.line(mask2,(0,h-z_limit),(w,h-z_limit),(255,255,255))
    
    return mask2, open_cv_trajectory, trajectory, world_x_left,world_x_right,world_z_left,world_z_right
    # return mask2,centre_x,centre_y

def vehicle_trajectory(coordinates_left, coordinates_right, mask_image, og_img, point_cloud):
    h,w,_=og_img.shape
    coordinates_left=coordinates_left[0]
    n_left=len(coordinates_left)
    left_world_coord=[]

    i=0
    while(i<n_left):
        if coordinates_left[i][0]<h and coordinates_left[i][1]<w:
            left_world_coord.append(point_cloud[coordinates_left[i][0]*w+coordinates_left[i][1]])
        if i==0:
            i=10
        else:
            i=2*i
        

    coordinates_right=coordinates_right[0]
    n_right=len(coordinates_right)
    right_world_coord=[]
    i=0
    while(i<n_right):
        if coordinates_right[i][0]<h and coordinates_right[i][1]<w:
            right_world_coord.append(point_cloud[coordinates_right[i][0]*w+coordinates_right[i][1]])
        if i==0:
            i=10
        else:
            i=2*i        

    return bezier_plot(left_world_coord,right_world_coord,mask_image,15)