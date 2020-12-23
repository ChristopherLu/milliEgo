"""
Test the model with DOUBLE sensor modalities using h5 files as the input
"""

import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

from os.path import join
# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, os.path.dirname(currentdir))
import argparse
import numpy as np
import json
import time
import math

from utility import plot_util
from utility.data_loader import load_data_multi_timestamp
from utility.test_util import convert_rel_to_44matrix, iround
from utility.networks import build_model_selective_imu, build_model_plus_imu, \
    build_model_att_fusion, build_model_cross_att, build_model_cross_fusion

# keras
from keras import backend as K

K.set_image_dim_ordering('tf')
K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))  #
K.set_learning_phase(0)  # Run testing mode

SCALER = 1.0  # scale label: 1, 100, 10000
RADIUS_2_DEGREE = 180.0 / math.pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', type=str, required=True, help='h5 file sequences, e.g, 1, 6, 13')
    parser.add_argument('--model', type=str, required=True, help='model architecture')
    parser.add_argument('--epoch', type=str, required=True, help='which trained epoch to load in')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='specify the data dir of test data)')
    parser.add_argument('--out_pred', type=str, required=False,
                        help='specify the output of csv file for the prediction)')
    parser.add_argument('--out_gt', type=str, required=False,
                        help='specify the output of csv file for the ground truth)')
    args = parser.parse_args()

    IMU_LENGTH = (np.int(os.path.dirname(args.data_dir)[-1]) - 1) * 5
    if IMU_LENGTH < 10:
        IMU_LENGTH = 10
    print('IMU LENGTH is {}'.format(IMU_LENGTH))
    # Define and construct model
    print("Building network model ......")
    if 'sel-' in args.model:
        network_model = build_model_selective_imu(join('./models', args.model, args.epoch), istraining=False)
    if 'deepmio' in args.model:
        network_model = build_model_plus_imu(join('./models', args.model, args.epoch), istraining=False)
    if 'att-' in args.model:
        nn_opt_path = join('./models', args.model, 'nn_opt.json')
        with open(nn_opt_path) as handle:
            nn_opt = json.loads(handle.read())

        imu_att_type = nn_opt['imu_att_type']
        network_model = build_model_att_fusion(join('./models', args.model, args.epoch),
                                               imu_length=IMU_LENGTH, istraining=False, imu_att_type=imu_att_type)
    if 'cross-' in args.model:
        nn_opt_path = join('./models', args.model, 'nn_opt.json')
        with open(nn_opt_path) as handle:
            nn_opt = json.loads(handle.read())
        if 'only' in args.model:
            network_model = build_model_cross_fusion(join('./models', args.model, args.epoch),
                                               imu_length=IMU_LENGTH, mask_att=nn_opt['cross_att_type'], istraining=False)
        else:
            print(join('./models', args.model, args.epoch))
            network_model = build_model_cross_att(join('./models', args.model, args.epoch),
                                               imu_length=IMU_LENGTH, mask_att=nn_opt['cross_att_type'], istraining=False)
    if 'dio' in args.model:
        network_model = build_model_plus_imu(join('./models', args.model, args.epoch), input_shape=(1, 480, 640, 3),
                                             istraining=False)
    if 'vio' in args.model:
        network_model = build_model_plus_imu(join('./models', args.model, args.epoch), input_shape=(1, 480, 640, 3),
                                             istraining=False)
    network_model.summary(line_length=120)

    seqs = args.seqs.split(',')

    with K.get_session() as sess:
        for seq in seqs:
            test_file = join(args.data_dir, 'turtle_seq_' + seq + '.h5')
            if 'dio' in args.model:
                n_chunk, x_time, x_mm_t, x_imu_t, y_t = load_data_multi_timestamp(test_file, 'depth')
            elif 'vio' in args.model:
                n_chunk, x_time, x_mm_t, x_imu_t, y_t = load_data_multi_timestamp(test_file, 'rgb')
            else:
                n_chunk, x_time, x_mm_t, x_imu_t, y_t = load_data_multi_timestamp(test_file,
                                                                                  'mmwave_middle')  # y (1, 2142, 6)

            y_t = y_t[0]
            print('Data shape: ', np.shape(x_mm_t), np.shape(x_imu_t), np.shape(y_t))
            len_x_i = x_mm_t[0].shape[0]
            print(len_x_i)

            # Set initial pose for GT and prediction
            gt_transform_t_1 = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]])
            pred_transform_t_1 = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])  # initial pose for prediction

            # Initialize value and counter
            count_img = 0
            ls_time = [0, 0, 0, 0]

            out_gt_array = []  # format (x,y) gt and (x,y) prediction
            out_pred_array = []  # format (x,y) gt and (x,y) prediction

            print('Reading images and imu ....')
            # for i in range(0, iround ((len_thermal_x_i-2)/2)):
            for i in range(0, (len_x_i - 1)):
                # Make prediction
                st_cnn_time = time.time()
                x_mm_1 = x_mm_t[0][i]
                x_mm_2 = x_mm_t[0][i + 1]

                x_mm_1 = np.expand_dims(x_mm_1, axis=0)
                x_mm_2 = np.expand_dims(x_mm_2, axis=0)

                # Repeat channels
                if any(x in args.model for x in ['deepmio', 'dio', 'sel-', 'att', 'cross', 'skip']):
                    x_mm_1 = np.repeat(x_mm_1, 3, axis=-1)
                    x_mm_2 = np.repeat(x_mm_2, 3, axis=-1)

                print('x_mm_2 shape is {}'.format(np.shape(x_mm_2)))

                x_imu = x_imu_t[0]
                x_imu = x_imu[i+1, 0:IMU_LENGTH, :]
                x_imu = np.expand_dims(x_imu, axis=0)
                print('x_imu shape is {}'.format(np.shape(x_imu)))

                predicted = sess.run([network_model.outputs],
                                     feed_dict={network_model.inputs[0]: x_mm_1,
                                                network_model.inputs[1]: x_mm_2,
                                                network_model.inputs[2]: x_imu})

                pred_pose = predicted[0][0][0][0]
                prediction_time = time.time() - st_cnn_time
                ls_time[0] += prediction_time

                print('Running (Hz)', 1.0 / (prediction_time))

                # Display the figure
                st_plot_time = time.time()

                # Composing the relative transformationnetwork_model for the prediction
                pred_transform_t = convert_rel_to_44matrix(0, 0, 0, pred_pose)
                abs_pred_transform = np.dot(pred_transform_t_1, pred_transform_t)
                print(abs_pred_transform)

                # Composing the relative transformation for the ground truth
                gt_transform_t = convert_rel_to_44matrix(0, 0, 0, y_t[i])
                abs_gt_transform = np.dot(gt_transform_t_1, gt_transform_t)
                print(abs_gt_transform)

                # Save the composed prediction and gt in a list
                out_gt_array.append(
                    [abs_gt_transform[0, 0], abs_gt_transform[0, 1], abs_gt_transform[0, 2], abs_gt_transform[0, 3],
                     abs_gt_transform[1, 0], abs_gt_transform[1, 1], abs_gt_transform[1, 2], abs_gt_transform[1, 3],
                     abs_gt_transform[2, 0], abs_gt_transform[2, 1], abs_gt_transform[2, 2], abs_gt_transform[2, 3]])

                out_pred_array.append(
                    [abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2], abs_pred_transform[0, 3],
                     abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2], abs_pred_transform[1, 3],
                     abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2],
                     abs_pred_transform[2, 3]])

                plot_time = time.time() - st_plot_time
                ls_time[1] += plot_time

                gt_transform_t_1 = abs_gt_transform
                pred_transform_t_1 = abs_pred_transform
                count_img += 1

            if not os.path.exists('./results'):
                os.makedirs('./results')
            # csv_location = join('./results', exp_name)
            np.savetxt(join('./results', args.model + '_ep' + args.epoch + '_seq' + seq),
                       out_pred_array, delimiter=",")
            np.savetxt(join('./results', 'gt_seq' + seq),
                       out_gt_array, delimiter=",")
            np.savetxt(join('./results', 'time_seq' + seq),
                       x_time, delimiter="\n")

            fig_dir = join('./figs', args.model, seq)
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            out_pred_array, out_gt_array = np.array(out_pred_array), np.array(out_gt_array)
            plot_util.plot2d(out_pred_array, out_gt_array,
                             join(fig_dir, args.model + '_ep' + args.epoch + '_seq' + seq + '.png'))
            for a in range(2):
                if a == 0:
                    ls_time[a] = ls_time[a] / count_img
                    ls_time[a] = int(round(ls_time[a] * 1000, 0))
                else:
                    ls_time[a] = ls_time[a] / count_img
                    ls_time[a] = int(round(ls_time[a] * 1000, 0))

            print('Model Prediction: {0} ms. Plot: {1} ms.'.format(str(ls_time[0]), str(ls_time[1])))
            print("Seq {} Finished!".format(seq))


if __name__ == "__main__":
    os.system("hostname")
    main()
