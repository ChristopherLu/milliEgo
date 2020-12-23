"""
Given relative pose as an input (e.g. from Lidar GMAPPING), generate the full trajectory
"""

import os
import argparse
import numpy as np
import pandas
from os.path import join, dirname
import inspect

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta_path', type=str, required=True, help="Path to delta pose")
    args = parser.parse_args()

    delta_file = args.delta_path
    full_traj = []
    delta_pose = pandas.read_csv(delta_file)
    delta_pose = delta_pose[['timestamp', 'x', 'y', 'z', 'w', 'x_l', 'y_l', 'z_l']]
    delta_pose = np.array(delta_pose)

    # initialize the origin
    pred_transform_t_1 = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

    for i in range(len(delta_pose)):
        # use the first 3 vectors to rotate the global pose if you like, otherwise just put 0,0,0
        pred_transform_t = transform44([delta_pose[i][0], delta_pose[i][1], delta_pose[i][2], delta_pose[i][3],
                                        delta_pose[i][5], delta_pose[i][6], delta_pose[i][7], delta_pose[i][4]])
        abs_pred_transform = np.dot(pred_transform_t_1, pred_transform_t)
        full_traj.append(
            [abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2], abs_pred_transform[0, 3],
             abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2], abs_pred_transform[1, 3],
             abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2],
             abs_pred_transform[2, 3]])
        pred_transform_t_1 = abs_pred_transform

    # parent_dir = dirname(dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
    delta_pose_dir = '/'.join(delta_file.split('/')[:-1])
    saved_filename = str(delta_pose_dir) + '/' + 'full_traj_gmapping.csv'
    # saved_filename = str(delta_pose_dir) + '/' + str(os.path.basename(args.delta_path).split('.')[0]) + '.csv'
    np.savetxt(saved_filename, full_traj, delimiter=",")

    print('Full trajectory is saved in: ', saved_filename)

if __name__ == "__main__":
    os.system('hostname')
    main()
