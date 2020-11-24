import numpy as np
import SDC_data_store

def state_update_long(trajectory):

    cte = abs(trajectory[1][1] - trajectory[0][1])
    # SDC_data_store.sum_cte+=cte
    if len(SDC_data_store.cte_queue) < 10:
        SDC_data_store.cte_queue.append(cte)
    else:
        SDC_data_store.cte_queue.popleft()
        SDC_data_store.cte_queue.append(cte)
    
def state_update_lat(x_left, x_right, yaw_error):
    if len(SDC_data_store.left_tolerance_queue) < 5:
        SDC_data_store.left_tolerance_queue.append(round(np.exp(-(x_left[0] - 0)),5))
        SDC_data_store.right_tolerance_queue.append(round(np.exp((x_right[0] - 0)),5))
        SDC_data_store.yaw_error_queue.append(round(yaw_error,5))
    else:
        SDC_data_store.left_tolerance_queue.popleft()
        SDC_data_store.left_tolerance_queue.append(round(np.exp(-(x_left[0] - 0)),5))
        SDC_data_store.right_tolerance_queue.popleft()
        SDC_data_store.right_tolerance_queue.append(round(np.exp((x_right[0] - 0)),5))
        SDC_data_store.yaw_error_queue.popleft()
        SDC_data_store.yaw_error_queue.append(round(yaw_error,5))