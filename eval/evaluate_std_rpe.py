# Evaluate RPE without SVD-based data association and scaling

from scipy.misc import imread
import numpy
import argparse
import os
from os.path import join, dirname
import inspect
import yaml
import associate
import csv
import math
from eulerangles import mat2euler, euler2mat
import yaml
from os.path import join, dirname

SCALER = 1.0 # scale label: 1, 100, 10000
RADIUS_2_DEGREE = 180.0 / math.pi

def rotated_to_local(T_w_c):
    # Input is 7 DoF absolute poses (3 trans, 4 quat), output is 6 DoF relative poses
    poses_local = []
    T_w_c = numpy.insert(T_w_c, 0, 1, axis=1) # add dummy timestamp
    # print(T_w_c)
    for i in range(1, len(T_w_c)):
        # print(T_w_c[i])
        # print(T_w_c[i][0,1:4])
        T_w_c_im1 = transform44(T_w_c[i-1])
        T_w_c_i = transform44(T_w_c[i])

        T_c_im1_c_i = numpy.dot(numpy.linalg.pinv(T_w_c_im1), T_w_c_i)

        # 3D: x, y, z, roll, pitch, yaw
        eular_c_im1_c_i = mat2euler(T_c_im1_c_i[0:3, 0:3])
        poses_local.append([SCALER * T_c_im1_c_i[0, 3], SCALER * T_c_im1_c_i[1, 3], SCALER * T_c_im1_c_i[2, 3],
                            SCALER * eular_c_im1_c_i[2] * RADIUS_2_DEGREE, SCALER * eular_c_im1_c_i[1] * RADIUS_2_DEGREE,
                            SCALER * eular_c_im1_c_i[0] * RADIUS_2_DEGREE])
    poses_local = numpy.array(poses_local)
    return poses_local


def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.

    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.

    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    _EPS = numpy.finfo(float).eps * 4.0
    # t = l[0,1:4]
    t = [l[0,1], l[0,2], l[0,3]]
    # q = numpy.array(l[0,4:8], dtype=numpy.float64, copy=True)
    q = [l[0,4], l[0,5], l[0,6], l[0,7]]
    q = numpy.array(q, dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    if nq < _EPS:
        return numpy.array((
            (1.0, 0.0, 0.0, t[0])
            (0.0, 1.0, 0.0, t[1])
            (0.0, 0.0, 1.0, t[2])
            (0.0, 0.0, 0.0, 1.0)
        ), dtype=numpy.float64)
    q *= numpy.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], t[0]),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], t[1]),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], t[2]),
        (0.0, 0.0, 0.0, 1.0)), dtype=numpy.float64)

if __name__ == '__main__':
    DESCRIPTION = """This script computes a dataset mean for particular modality."""

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('first_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('second_file', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',
                        default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)', default=1.0)
    parser.add_argument('--max_difference',
                        help='maximally allowed time difference for matching entries (default: 0.02)',
                        default=0.02)
    parser.add_argument('--err_comp', help='append the error stats in a csv (format: csv)')
    parser.add_argument('--model', help='the first column in the err_comp')
    parser.add_argument('--verbose',
                        help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)',
                        action='store_true')
    args = parser.parse_args()

    # Get and associate data based on time
    first_list = associate.read_file_list(args.first_file)
    second_list = associate.read_file_list(args.second_file)
    first_stamps = first_list.keys()
    first_stamps.sort()
    second_stamps = second_list.keys()
    second_stamps.sort()

    matches = associate.associate(first_list, second_list, float(args.offset), float(args.max_difference))

    first_data = numpy.matrix([[float(value) for value in first_list[a][0:7]] for a, b in matches]).transpose()
    second_data = numpy.matrix(
        [[float(value) * float(args.scale) for value in second_list[b][0:7]] for a, b in matches]).transpose()

    rel_first_data = rotated_to_local(first_data.transpose())
    rel_second_data = rotated_to_local(second_data.transpose())
    print(rel_first_data)
    # # Compute Error
    temp_trans_error = rel_first_data[:,0:3] - rel_second_data[:,0:3]
    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(temp_trans_error, temp_trans_error), 1))

    temp_rot_error = rel_first_data[:, 3:6] - rel_second_data[:, 3:6]
    rot_error = numpy.sqrt(numpy.sum(numpy.multiply(temp_rot_error, temp_rot_error), 1))

    # save trans_error
    root_dir = os.path.dirname(os.path.dirname(args.err_comp))
    save_dir = join(root_dir, '3d_rte_log')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('++++++++++++save dir is {}'.format(save_dir))
    save_path = join(save_dir, args.model + '_rte_trans.csv')
    numpy.savetxt(save_path, trans_error, delimiter=',')
    save_path = join(save_dir, args.model + '_rte_rot.csv')
    numpy.savetxt(save_path, rot_error, delimiter=',')

    if args.verbose:
        print("compared_pose_pairs %d pairs"%(len(trans_error)))
        print("translational_error.rmse %f m"%numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error)))
        print("translational_error.mean %f m"%numpy.mean(trans_error))
        print("translational_error.median %f m"%numpy.median(trans_error))
        print("translational_error.std %f m"%numpy.std(trans_error))
        print("translational_error.min %f m"%numpy.min(trans_error))
        print("translational_error.max %f m"%numpy.max(trans_error))

        print("rotational_error.rmse %f deg" % (numpy.sqrt(numpy.dot(rot_error, rot_error) / len(rot_error))))
        print("rotational_error.mean %f deg" % (numpy.mean(rot_error)))
        print("rotational_error.median %f deg" % (numpy.median(rot_error)))
        print("rotational_error.std %f deg" % (numpy.std(rot_error)))
        print("rotational_error.min %f deg" % (numpy.min(rot_error)))
        print("rotational_error.max %f deg" % (numpy.max(rot_error)))
    else:
        print(numpy.mean(trans_error))

    if args.err_comp:
        row = [args.model, numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error)),
               numpy.mean(trans_error),
               numpy.median(trans_error), numpy.std(trans_error), numpy.min(trans_error), numpy.max(trans_error),
               numpy.sqrt(numpy.dot(rot_error, rot_error) / len(rot_error)),
               numpy.mean(rot_error), numpy.median(rot_error),
               numpy.std(rot_error), numpy.min(rot_error),
               numpy.max(rot_error)]
        with open(args.err_comp, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(row)
            file.close()
