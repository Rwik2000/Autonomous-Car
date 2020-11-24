from collections import deque
def prev_data_init():
    global coord_left_prev1,coord_right_prev1
    global count, turn_value
    global shadow_check
    global prev_throttle, prev_turn_val
    global prev_left_tol,prev_right_tol, prev_yaw_error
    global prev_cte, cte_queue
    global left_tolerance_queue, right_tolerance_queue, yaw_error_queue
    global frame_score, score_queue
    global steer_queue, throttle_queue, rewards_queue, critic_score
    frame_score = 0
    prev_throttle = 0
    prev_turn_val = 0
    score_queue = deque()
    shadow_check=1
    turn_value=0
    count=0
    prev_cte = 0
    prev_left_tol,prev_right_tol,prev_yaw_error =0,0,0
    # sum_cte = 0
    cte_queue = deque()
    left_tolerance_queue =deque()
    right_tolerance_queue=deque()
    yaw_error_queue = deque()
    steer_queue = deque()
    throttle_queue = deque()
    rewards_queue = deque()
    critic_score = deque()
