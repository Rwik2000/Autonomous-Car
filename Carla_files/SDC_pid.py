import numpy as np
import SDC_data_store
from SDC_state import state_update_long,state_update_lat

def PID_throttle(trajectory, reverse):

    cte = abs(trajectory[1][1] - trajectory[0][1])
    # state_update_long(trajectory)
    throttle_factor = 0.095
    throttle_factor_i = 0.043
    throttle_factor_d = 0.005

    sum_cte = sum(list(SDC_data_store.cte_queue))
    a = 0.1*(throttle_factor*cte + throttle_factor_i*sum_cte - throttle_factor_d*(cte - SDC_data_store.prev_cte))

    if reverse == 0:        
        a = max(a, 0.7)

    print("cte : "+str(cte))
    print("throttle : " + str(a))
    SDC_data_store.prev_cte = cte
    if a>1:
        a = 1
    
    SDC_data_store.prev_throttle = a
    return a

def PID_steer(trajectory, reverse, x_left, x_right):
    # yaw_error = np.arctan()
    # print(x_left[0],x_right[0])
    centre_x = (x_left[0] + x_right[0])/2
    lateral_error = 0 - centre_x
    road_width = x_left[0] - x_right[0]
    

    # print(trajectory)
    track_point_yaw = trajectory[-5]
    yaw_error = np.arctan((0 - track_point_yaw[0])/(track_point_yaw[1]))
    
    side_factor = 2.5
    side_factor_i = 0.0
    side_factor_d =  -0

    yaw_factor = 0.5
    yaw_factor_i = 0.0
    yaw_factor_d =  1

    # state_update_lat(x_left,x_right, yaw_error)

    steer = (side_factor*np.exp(-(x_left[0] - 0)) + side_factor_i*sum(list(SDC_data_store.left_tolerance_queue)) + side_factor_d*(round(np.exp(-(x_left[0])),5)-SDC_data_store.prev_left_tol)
             - side_factor*np.exp((x_right[0] - 0)) - side_factor_i* sum(list(SDC_data_store.right_tolerance_queue)) - side_factor_d*(round(np.exp((x_right[0])),5)-SDC_data_store.prev_right_tol)
             + yaw_factor*yaw_error + yaw_factor_i*sum(list(SDC_data_store.yaw_error_queue)) + yaw_factor_d*(yaw_error - SDC_data_store.prev_yaw_error))

    SDC_data_store.prev_yaw_error = round(yaw_error,5)
    SDC_data_store.prev_left_tol = np.exp(-(x_left[0]))
    SDC_data_store.prev_right_tol = np.exp((x_right[0]))
    SDC_data_store.prev_turn_val = steer

    return steer