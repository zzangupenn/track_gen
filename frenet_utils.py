import numpy as np

def load_map(MAP_DIR, map_info, map_ind, conf, scale=1, reverse=False):
    """
    loads waypoints
    """
    map_info = map_info[map_ind][1:]
    conf.wpt_path = map_info[0]
    conf.wpt_delim = map_info[1]
    conf.wpt_rowskip = int(map_info[2])
    conf.wpt_xind = int(map_info[3])
    conf.wpt_yind = int(map_info[4])
    conf.wpt_thind = int(map_info[5])
    conf.wpt_vind = int(map_info[6])
    
    waypoints = np.loadtxt(MAP_DIR + conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
    if reverse: # NOTE: reverse map
        waypoints = waypoints[::-1]
        # if map_ind == 41: waypoints[:, conf.wpt_thind] = waypoints[:, conf.wpt_thind] + 3.14
    # if map_ind == 41: waypoints[:, conf.wpt_thind] = waypoints[:, conf.wpt_thind] + np.pi / 2
    waypoints[:, conf.wpt_yind] = waypoints[:, conf.wpt_yind] * scale
    waypoints[:, conf.wpt_xind] = waypoints[:, conf.wpt_xind] * scale # NOTE: map scales
    
    # NOTE: initialized states for forward
    if conf.wpt_thind == -1:
        print('Convert to raceline format.')
        # init_theta = np.arctan2(waypoints[1, conf.wpt_yind] - waypoints[0, conf.wpt_yind], 
        #                         waypoints[1, conf.wpt_xind] - waypoints[0, conf.wpt_xind])
        waypoints = centerline_to_frenet(waypoints, velocity=5.0)
        conf.wpt_xind = 1
        conf.wpt_yind = 2
        conf.wpt_thind = 3
        conf.wpt_vind = 5
    # else:
    init_theta = waypoints[0, conf.wpt_thind]
    
    return waypoints, conf, init_theta

# def load_map_random_gen(MAP_DIR, map_name, scale=1):
#     import yaml, cv2
#     with open(MAP_DIR + map_name + '.yaml') as stream:
#         info = yaml.load(stream, Loader=yaml.Loader)

#     cv_img = cv2.imread(MAP_DIR + info['image'], -1)
#     obs_list = np.loadtxt(MAP_DIR + map_name + '_obs.csv', delimiter=',', skiprows=0)
#     waypoints = np.loadtxt(MAP_DIR + map_name + '.csv', delimiter=',', skiprows=0)
#     map_origin = info['origin']
#     scale = info['resolution']
    
#     return cv_img, waypoints, obs_list, map_origin, scale
    

def centerline_to_frenet(trajectory, velocity=5.0):   
    '''
    Converts a trajectory in the form [x_m, y_m, w_tr_right_m, w_tr_left_m] to [s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2, w_tr_right_m, w_tr_left_m]
    Assumes constant velocity

    Parameters
    ----------
    trajectory : np.array
        Trajectory in the form [x_m, y_m, w_tr_right_m, w_tr_left_m]
    velocity : float, optional
        Velocity of the vehicle, by default 5.0
    '''
    # Initialize variables
    # eps = 1e-5 * (np.random.randint(0, 1) - 1)
    eps = 0
    s = 0.0
    x = trajectory[0, 0]
    y = trajectory[0, 1]
    psi = np.arctan2((trajectory[1, 1] - trajectory[0, 1]), (trajectory[1, 0] - trajectory[0, 0])) 
    kappa = eps
    vx = velocity
    ax = 0.0
    width_l = trajectory[0, 2]
    width_r = trajectory[0, 3]

    # Initialize output
    output = np.zeros((trajectory.shape[0], 9))
    output[0, :] = np.array([s, x, y, psi, kappa, vx, ax, width_l, width_r])

    # Iterate over trajectory
    for i in range(1, trajectory.shape[0]):
        # Calculate s
        s += np.sqrt((trajectory[i, 0] - trajectory[i-1, 0])**2 + (trajectory[i, 1] - trajectory[i-1, 1])**2)
        # Calculate psi
        psi = np.arctan2((trajectory[i, 1] - trajectory[i-1, 1]), (trajectory[i, 0] - trajectory[i-1, 0]))
        # Calculate kappa
        # eps = 1e-5 * (np.random.randint(0, 1) - 1)
        eps = 0
        kappa = (trajectory[i, 3] - trajectory[i, 2]) / (2 * np.sqrt((trajectory[i, 0] - trajectory[i-1, 0])**2 + (trajectory[i, 1] - trajectory[i-1, 1])**2)) + eps
        # Calculate ax
        ax = 0.0

        # Save to output
        output[i, :] = np.array([s, trajectory[i, 0], trajectory[i, 1], psi, kappa, vx, ax, trajectory[i, 2], trajectory[i, 3]])

    return output

def get_closest_point_vectorized(point, array):
    """
    Find ID of the closest point from point to array.
    Using euclidian norm.
    Works in nd.
    :param point: np.array([x, y, z, ...])
    :param array: np.array([[x1, y1, z1, ...], [x2, y2, z2, ...], [x3, y3, z3, ...], ...])
    :return: id of the closest point
    """

    min_id = np.argmin(np.sum(np.square(array - point), 1))

    return min_id

def determine_side(a, b, p):
    """ Determines, if car is on right side of trajectory or on left side
    Arguments:
         a - point of trajectory, which is nearest to the car, geometry_msgs.msg/Point
         b - next trajectory point, geometry_msgs.msg/Point
         p - actual position of car, geometry_msgs.msg/Point

    Returns:
         1 if car is on left side of trajectory
         -1 if car is on right side of trajectory
         0 if car is on trajectory
    """
    side = (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
    if side > 0:
        return -1
    elif side < 0:
        return 1
    else:
        return 0
    
def get_rotation_matrix_2d(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])

def find_center_of_arc(point, radius, direction):
    """
    :param point: Point on the arc. np.array([x, y])
    :param radius: Radius of arc. - -> arc is going to the right, + -> arc is going to the left.
    :param direction: direction which way the arc continues from the point (angle 0-2pi)
    :return: center: np.array([x, y])
    """
    R = get_rotation_matrix_2d(direction + np.pi / 2.0 * np.sign(radius))
    C = np.squeeze(point + (R @ np.array([[abs(radius)], [0.0]])).T)
    return C
    
def find_arc_end(start_point, radius, start_angle, arc_angle):
    # print(f"start angle: {start_angle}")
    angle = start_angle + arc_angle * np.sign(radius)
    # print(f"angle: {start_angle + np.pi/2.0 * np.sign(radius)}")
    C = find_center_of_arc(start_point, radius, start_angle + np.pi / 2.0 * np.sign(radius))
    arc_end_point = C + abs(radius) * np.array([np.cos(angle), np.sin(angle)])
    return arc_end_point     

def frenet_to_cartesian(pose, trajectory):
    """
    :param pose: [s, ey, eyaw]
    :return:
    """
    # trajectory ... s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
    diff = trajectory[:, 0] - pose[0]

    # print(diff)

    segment_id = np.argmax(diff[diff <= 0])  # should be always id of the point that has smaller s than the point

    # print(segment_id)
    # print(trajectory[segment_id, :])

    if trajectory[segment_id, 4] == 0:
        # line
        yaw = np.mod(trajectory[segment_id, 3] + pose[2], 2.0 * np.pi)
        s_reminder = pose[0] - trajectory[segment_id, 0]
        R1 = get_rotation_matrix_2d(trajectory[segment_id, 3])
        R2 = get_rotation_matrix_2d(trajectory[segment_id, 3] + np.pi / 2.0 * np.sign(pose[1]))
        position = (trajectory[segment_id, 1:3] + (R1 @ np.array([[abs(s_reminder)], [0.0]])).T + (
                R2 @ np.array([[abs(pose[1])], [0.0]])).T).squeeze()
    else:
        # circle
        center = find_center_of_arc(trajectory[segment_id, 1:3],
                                            1.0 / trajectory[segment_id, 4],
                                            trajectory[segment_id, 3])

        s_reminder = pose[0] - trajectory[segment_id, 0]
        start_angle = np.mod(trajectory[segment_id, 3] - np.pi / 2.0 * np.sign(trajectory[segment_id, 4]), 2 * np.pi)
        arc_angle = s_reminder / abs(1.0 / trajectory[segment_id, 4])
        trajectory_point = find_arc_end(trajectory[segment_id, 1:3],
                                                1.0 / trajectory[segment_id, 4],
                                                start_angle,
                                                arc_angle)
        vector = trajectory_point - center

        position = trajectory_point + vector / np.linalg.norm(vector) * pose[1] * (-1) * np.sign(trajectory[segment_id, 4])
        yaw = np.mod(np.arctan2(vector[1], vector[0]) + np.pi / 2.0 * np.sign(trajectory[segment_id, 4]) + pose[2], 2 * np.pi)
    # print(np.array([position[0], position[1], yaw]))
    return np.array([position[0], position[1], yaw])

def cartesian_to_frenet(pose, trajectory):
    """
    :param pose: np.array([x,y,yaw])
    :param trajectory: np.array([s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2]
    )
    :return: frenet_pose: np.array([s, ey, epsi])
    """
    min_id = get_closest_point_vectorized(point=pose[0:2], array=trajectory[:, 1:3])

    a = np.mod(np.arctan2(pose[1] - trajectory[min_id, 2], pose[0] - trajectory[min_id, 1]) - (trajectory[min_id, 3] - np.pi / 2),
                2.0 * np.pi)

    if a > np.pi:
        min_id = (min_id + trajectory.shape[0] - 2) % (trajectory.shape[0] - 1)

    if trajectory[min_id, 4] == 0:
        eyaw = pose[2] - trajectory[min_id, 3]

        if(min_id == trajectory.shape[0] - 1):
            vector = (trajectory[0, 1:3] - trajectory[min_id - 1, 1:3])[np.newaxis].T # Loop back to the start
        else:
            vector = (trajectory[min_id + 1, 1:3] - trajectory[min_id, 1:3])[np.newaxis].T

        vector = vector / np.linalg.norm(vector)

        projector = vector @ vector.T
        
        projection = projector @ (pose[0:2] - trajectory[min_id, 1:3])

        point = trajectory[min_id, 1:3] + projection

        if(min_id == trajectory.shape[0] - 1):
            ey = np.linalg.norm(point - pose[0:2]) * determine_side(trajectory[min_id, 1:3], trajectory[0, 1:3], pose[0:2])
        else:
            ey = np.linalg.norm(point - pose[0:2]) * determine_side(trajectory[min_id, 1:3], trajectory[min_id + 1, 1:3], pose[0:2])
        
        s = trajectory[min_id, 0] + np.linalg.norm(point - trajectory[min_id, 1:3])

    else:
        center = find_center_of_arc(point=np.array([trajectory[min_id, 1], trajectory[min_id, 2]]),
                                            radius=1.0 / trajectory[min_id, 4],
                                            direction=trajectory[min_id, 3])

        ey = (abs(1.0 / trajectory[min_id, 4]) - np.linalg.norm(center - pose[0:2])) * np.sign(trajectory[min_id, 4])

        start_point_angle = np.arctan2(trajectory[min_id, 2] - center[1], trajectory[min_id, 1] - center[0])
        end_angle = np.arctan2(pose[1] - center[1], pose[0] - center[0])

        if end_angle < 0.0:
            end_angle = end_angle + 2.0 * np.pi

        if start_point_angle < 0.0:
            start_point_angle = start_point_angle + 2.0 * np.pi

        angle = np.sign(trajectory[min_id, 4]) * end_angle - np.sign(trajectory[min_id, 4]) * start_point_angle

        if angle < 0.0:
            angle = angle + 2.0 * np.pi

        s = trajectory[min_id, 0] + angle * abs(1.0 / trajectory[min_id, 4])

        eyaw = pose[2] - (end_angle + np.pi / 2.0 * np.sign(trajectory[min_id, 4]))

    if eyaw > np.pi:
        eyaw -= 2.0 * np.pi
    if eyaw < -np.pi:
        eyaw += 2.0 * np.pi

    return np.array([s, ey, eyaw])


