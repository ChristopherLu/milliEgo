#!/usr/bin/python
import argparse
from os.path import join
from eulerangles import mat2euler, euler2quat
import pandas

DESCRIPTION = """This script receives a path to the pose (in matrix) and the timestamp files,
                and convert it to a new pose in quaternion format, concatenated with the timestamp."""

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('--pose', required=True, help='''Specify the path to output pose from the network.''')
parser.add_argument('--timestamp', required=True, help='''Specify the path to the pose timestamps.''')
parser.add_argument('--save_path', help='''Specify where to save the new output file.''')
parser.add_argument('--is_gt', help='''0: not ground truth file, 1: gt file.''')

args = parser.parse_args()

with open(args.pose, 'r') as pose_file:
    poses = [line[:-1] for line in pose_file]

with open(args.timestamp, 'r') as time_file:
    timestamps = [line[:-1] for line in time_file]

filename = args.save_path
with open(filename, 'w') as fw:
    for i in range(len(poses)):
        temp_trans = [poses[i].split(',')[3], poses[i].split(',')[7], poses[i].split(',')[11]]
        temp_rot = [[float(poses[i].split(',')[0]), float(poses[i].split(',')[1]), float(poses[i].split(',')[2])],
                    [float(poses[i].split(',')[4]), float(poses[i].split(',')[5]), float(poses[i].split(',')[6])],
                    [float(poses[i].split(',')[8]), float(poses[i].split(',')[9]), float(poses[i].split(',')[10])]]
        euler_rot = mat2euler(temp_rot) # format: z, y, x
        quat_rot = euler2quat(euler_rot[0], euler_rot[1], euler_rot[2]) # format: w, x, y z
        # required format: timestamp tx ty tz qx qy qz qw
        if int(args.is_gt) == 0:
            output_line = str(timestamps[i]) + ' ' + str(temp_trans[0]) + ' ' + str(temp_trans[1]) + ' ' + str(temp_trans[2]) \
                      + ' ' + str(quat_rot[1]) + ' ' + str(quat_rot[2]) + ' ' + str(quat_rot[3]) + ' ' + str(quat_rot[0]) + '\n'
        if int(args.is_gt) == 1:
            output_line = str(timestamps[i+1].split(',')[0]) + ' ' + str(temp_trans[0]) + ' ' + str(temp_trans[1]) + ' ' + str(temp_trans[2]) \
                      + ' ' + str(quat_rot[1]) + ' ' + str(quat_rot[2]) + ' ' + str(quat_rot[3]) + ' ' + str(quat_rot[0]) + '\n'
        fw.write(output_line)

print('File is successfully converted to: ', args.save_path)