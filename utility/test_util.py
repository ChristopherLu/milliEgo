import numpy as np
from eulerangles import mat2euler, euler2quat, euler2mat


def iround(x):
    """iround(number) -> integer
    Round a number to the nearest integer."""
    y = round(x) - .5
    return int(y) + (y > 0)


def transform44(l):
    _EPS = np.finfo(float).eps * 4.0
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.

    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.

    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.array((
            (1.0, 0.0, 0.0, t[0])
            (0.0, 1.0, 0.0, t[1])
            (0.0, 0.0, 1.0, t[2])
            (0.0, 0.0, 0.0, 1.0)
        ), dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], t[0]),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], t[1]),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], t[2]),
        (0.0, 0.0, 0.0, 1.0)), dtype=np.float64)


def convert_rel_to_44matrix(rot_x, rot_y, rot_z, pose):
    R_pred = euler2mat(rot_x, rot_y, rot_z)
    rotated_pose = np.dot(R_pred, pose[0:3])
    DEGREE_2_RADIUS = np.pi / 180.0
    pred_quat = euler2quat(z=pose[5] * DEGREE_2_RADIUS, y=pose[4] * DEGREE_2_RADIUS,
                           x=pose[3] * DEGREE_2_RADIUS)
    pred_transform_t = transform44([0, rotated_pose[0], rotated_pose[1], rotated_pose[2],
                                    pred_quat[1], pred_quat[2], pred_quat[3], pred_quat[0]])
    return pred_transform_t

