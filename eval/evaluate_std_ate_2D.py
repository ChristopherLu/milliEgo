# Evaluate ATE without SVD-based data association and scaling

from scipy.misc import imread
import numpy
import argparse
import os
from os.path import join, dirname
import inspect
import associate
import csv

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

    first_xyz = numpy.matrix([[float(value) for value in first_list[a][0:3]] for a, b in matches]).transpose()
    second_xyz = numpy.matrix(
        [[float(value) * float(args.scale) for value in second_list[b][0:3]] for a, b in matches]).transpose()

    # Compute Error
    alignment_error = second_xyz[:][0:2] - first_xyz[:][0:2]
    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(alignment_error, alignment_error), 0)).A[0]

    # save trans_error
    root_dir = os.path.dirname(args.err_comp)
    save_dir = join(root_dir, '2d_ate_cdf')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = join(save_dir, args.model + '_errcdf.csv')
    numpy.savetxt(save_path, trans_error, delimiter=',')

    print(alignment_error)

    if args.verbose:
        print("compared_pose_pairs %d pairs" % (len(trans_error)))
        print("absolute_translational_error.rmse %f m" % numpy.sqrt(
            numpy.dot(trans_error, trans_error) / len(trans_error)))
        print("absolute_translational_error.mean %f m" % numpy.mean(trans_error))
        print("absolute_translational_error.median %f m" % numpy.median(trans_error))
        print("absolute_translational_error.std %f m" % numpy.std(trans_error))
        print("absolute_translational_error.min %f m" % numpy.min(trans_error))
        print("absolute_translational_error.max %f m" % numpy.max(trans_error))
    else:
        print("%f" % numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error)))

    if args.err_comp:
        row = [args.model, numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error)), numpy.mean(trans_error),
               numpy.median(trans_error), numpy.std(trans_error), numpy.min(trans_error), numpy.max(trans_error)]
        with open(args.err_comp, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(row)
            file.close()