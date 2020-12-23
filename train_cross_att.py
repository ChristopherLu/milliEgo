"""
Training deep mmwave+imu odometry from pseudo ground truth
"""

import os
os.environ['KERAS_BACKEND']='tensorflow'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

from os.path import join
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import numpy as np
import h5py
import matplotlib as mpl
import yaml
mpl.use('Agg')
import glob
import json


# keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from keras.callbacks import TensorBoard
import keras
from keras import backend as K

K.set_image_dim_ordering('tf')
K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))) #
import math

# utiliy
from utility.networks import build_model_cross_att
from utility.data_loader import load_data_multi, validation_stack


def main():
    print('For mmwave+imu odom!')

    with open(join(currentdir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f)

    MODEL_NAME = cfg['nn_opt']['cross-mio_params']['nn_name']

    data_dir = cfg['mvo']['multimodal_data_dir']
    IMU_LENGTH = (np.int(data_dir[-1]) - 1) * 5
    if IMU_LENGTH < 10:
        IMU_LENGTH = 10
    print('IMU LENGTH is {}'.format(IMU_LENGTH))
    batch_size = cfg['mvo']['batch_size']

    # data_dir = '/root/workspace/datasets/MilliVO/h5_rgb_mmwave/'

    model_dir = join('./models', MODEL_NAME)

    print("Building network model .....")
    print(cfg['nn_opt']['cross-mio_params'])
    model = build_model_cross_att(cfg['nn_opt']['cross-mio_params'],
                                     mask_att=cfg['nn_opt']['cross-mio_params']['cross_att_type'],
                                            imu_length=IMU_LENGTH)
    model.summary(line_length=120)

    # Training without validation set
    checkpoint_path = join('./models', MODEL_NAME, 'best').format('h5')
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                                   mode='min', save_best_only=True, verbose=1)
    def step_decay(epoch):
        initial_lrate = cfg['nn_opt']['cross-mio_params']['lr_rate'] # 0.001, 0.0001
        drop = 0.75
        epochs_drop = 25.0
        lrate = initial_lrate * math.pow(drop,
                                         math.floor((1 + epoch) / epochs_drop))
        print('Learning rate: ' + str(lrate))
        return lrate

    lrate = LearningRateScheduler(step_decay)
    tensor_board = TensorBoard(log_dir=join(model_dir, 'logs'))
    training_loss = []

    # Load validation data
    validation_files = glob.glob(join(data_dir, 'val', '*.h5'))
    x_mm_val_1, x_mm_val_2, x_imu_val_t, y_val_t = validation_stack(validation_files, sensor='mmwave_middle', imu_length=IMU_LENGTH)
    print(x_mm_val_1)
    len_val_i = y_val_t.shape[0]
    print('Final mmwave validation shape:', np.shape(x_mm_val_1), np.shape(y_val_t))

    # grap training files
    training_files = sorted(glob.glob(join(data_dir, 'train', '*.h5')))
    n_training_files = len(training_files)
    for e in range(cfg['mvo']['epochs']+1):
        print("|-----> epoch %d" % e)
        np.random.shuffle(training_files)
        for i, training_file in enumerate(training_files):

            print('---> Loading training file: {}', training_file.split('/')[-1])
            n_chunk, x_mm_t, x_imu_t, y_t = load_data_multi(training_file, 'mmwave_middle')

            # generate random length sequences
            len_x_i = x_mm_t[0].shape[0] # ex: length of sequence is 300

            range_seq = np.arange(len_x_i-batch_size-1)
            np.random.shuffle(range_seq)
            for j in range(len(range_seq) // (batch_size - 1)):
                x_mm_1, x_mm_2, x_imu, y_label = [], [], [], []
                starting = range_seq[j*(batch_size-1)]
                seq_idx_1 = range(starting, starting + (batch_size - 1))
                seq_idx_2 = range(starting + 1, starting + batch_size)
                x_mm_1.extend(x_mm_t[0][seq_idx_1, :, :, :])
                x_mm_2.extend(x_mm_t[0][seq_idx_2, :, :, :])
                # x_imu.extend(x_imu_t[0][seq_idx_2, :, :])
                x_imu.extend(x_imu_t[0][seq_idx_2, 0:IMU_LENGTH, :]) # for 10 imu data
                y_label.extend(y_t[0][seq_idx_1, :])

                x_mm_1, x_mm_2, x_imu, y_label = np.array(x_mm_1), np.array(x_mm_2), \
                                                 np.array(x_imu), np.array(y_label)

                # for flownet
                x_mm_1 = np.repeat(x_mm_1, 3, axis=-1)
                x_mm_2 = np.repeat(x_mm_2, 3, axis=-1)

                y_label = np.expand_dims(y_label, axis=1)
                # x_imu = np.expand_dims(x_imu, axis=1)
                print(np.shape(x_mm_1))

                print('Training data:', np.shape(x_mm_1), np.shape(x_mm_2), np.shape(x_imu))
                print('Epoch: ', str(e), ', Sequence:', str(i), ', Batch: ', str(j), ', Start at index: ', str(starting))

                if i == n_training_files - 1 and j == (len(range_seq) // (batch_size - 1)) - 1:
                    history = model.fit({'image_1': x_mm_1, 'image_2': x_mm_2, 'imu_data': x_imu},
                                        {'fc_trans': y_label[:,:,0:3], 'fc_rot': y_label[:, :,3:6]},
                                        validation_data=(
                                            [x_mm_val_1[0:len_val_i, :, :, :, :],
                                             x_mm_val_2[0:len_val_i, :, :, :, :],
                                             x_imu_val_t[0:len_val_i, :, :]],
                                            [y_val_t[:, :, 0:3],
                                             y_val_t[:, :, 3:6]]),
                                        batch_size=batch_size-1, shuffle='batch', epochs=1,
                                        # callbacks=[checkpointer, tensor_board], verbose=1)
                                        callbacks=[checkpointer, lrate, tensor_board], verbose=1)

                    training_loss.append(history.history['loss'])
                else:
                    model.fit({'image_1': x_mm_1, 'image_2': x_mm_2, 'imu_data': x_imu},
                              {'fc_trans': y_label[:, :, 0:3], 'fc_rot': y_label[:, :, 3:6]},
                              batch_size=batch_size-1, shuffle='batch', epochs=1, verbose=1)

        if ((e % 5) == 0):
            model.save(join(model_dir, str(e).format('h5')))

        if e == 0:
            print('Saving nn options ....')
            with open(join(model_dir, 'nn_opt.json'), 'w') as fp:
                json.dump(cfg['nn_opt']['cross-mio_params'], fp)

    print("Training for model has finished!")

    print('Saving training loss ....')
    train_loss = np.array(training_loss)
    loss_file_save = join(model_dir, 'training_loss.' + MODEL_NAME +'.h5')
    with h5py.File(loss_file_save, 'w') as hf:
        hf.create_dataset('train_loss', data=train_loss)

    print('Finished training ', str(n_training_files), ' trajectory!')

if __name__ == "__main__":
    os.system("hostname")
    # print(SAVE_FILE_NAME.format('h5'))
    main()
