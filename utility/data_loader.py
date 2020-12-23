import os
from os.path import join
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Model, Sequential  # Sequential, Graph
from keras.models import model_from_json, model_from_yaml, load_model
from keras.layers import (Input, Dense, Convolution2D, Activation, Flatten, Dropout, AveragePooling1D, MaxPooling2D,
                          ZeroPadding2D, Merge, Reshape, merge, Multiply, Lambda, AveragePooling2D, GlobalAveragePooling2D, LSTM, TimeDistributed)
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
import h5py
import keras
from keras import backend as K
import numpy as np

# load multi-modal data with IMU always in
def load_data_multi(training_file, sensor):

    # Load data
    x_sensor, x_imu, y = [], [], []
    hdf5_file = h5py.File(training_file, 'r')
    x_sensor_temp = hdf5_file.get(sensor+'_data')
    x_imu_temp = hdf5_file.get('imu_data')
    y_temp = hdf5_file.get('label_data')

    print('Data shape: ' + str(np.shape(x_sensor_temp)))

    # this is for rgb
    # x_rgb_temp = np.squeeze(x_rgb_temp, axis=0)
    # x_imu_temp = np.squeeze(x_imu_temp, axis=0)
    # y_temp = np.squeeze(y_temp, axis=0)

    # this is for raw data

    if x_sensor_temp.shape[0] == 1:
        x_sensor_temp = x_sensor_temp[0]

    if x_imu_temp.shape[0] == 1:
        x_imu_temp = x_imu_temp[0]

    y_temp = y_temp[0]

    print('Data shape: ' + str(np.shape(x_sensor_temp)) + str(str(np.shape(y_temp))))

    data_size = np.size(x_sensor_temp, axis=0)

    # Determine whether the data should be divided into several chunks
    # to fit in memory
    data_per_chunk = 5000
    is_special_case = False
    if data_size > 10000:
        n_chunk = data_size // data_per_chunk
        n_chunk += 1
        is_special_case = True
    else:
        n_chunk = 1

    if is_special_case == True:
        # Divide into several chunks if the length of the data is too large
        for i in range(n_chunk-1):
            x_sensor.append(x_sensor_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :, :, :])
            x_imu.append(x_imu_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :, :, :])
            y.append(y_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :])

        x_sensor.append(x_sensor_temp[(data_size - data_per_chunk):data_size, :, :, :])
        x_imu.append(x_imu_temp[(data_size - data_per_chunk):data_size, :, :, :])
        y.append(y_temp[(data_size - data_per_chunk):(data_size-1), :])
    else:
        x_sensor.append(x_sensor_temp[0:data_size, :, :, :])
        x_imu.append(x_imu_temp[0:data_size, :, :])
        y.append(y_temp[0:(data_size-1), :])
    return n_chunk, x_sensor, x_imu, y


def load_data_triple_timestamp(training_file, sensor_a, sensor_b):
    # Load data
    x_sensor_a, x_sensor_b, x_imu, y = [], [], [], []
    hdf5_file = h5py.File(training_file, 'r')
    x_time = hdf5_file.get('timestamp')
    x_sensor_a_temp = hdf5_file.get(sensor_a + '_data')
    x_sensor_b_temp = hdf5_file.get(sensor_b + '_data')
    x_imu_temp = hdf5_file.get('imu_data')
    y_temp = hdf5_file.get('label_data')

    print('Data shape: ' + str(np.shape(x_sensor_a_temp)))

    if x_sensor_a_temp.shape[0] == 1:
        x_sensor_a_temp = x_sensor_a_temp[0]

    if x_sensor_b_temp.shape[0] == 1:
        x_sensor_b_temp = x_sensor_b_temp[0]

    if x_imu_temp.shape[0] == 1:
        x_imu_temp = x_imu_temp[0]

    y_temp = y_temp[0]

    print('Data shape: ' + str(np.shape(x_sensor_a_temp)) + str(str(np.shape(y_temp))))

    data_size = np.size(x_sensor_a_temp, axis=0)

    # Determine whether the data should be divided into several chunks
    # to fit in memory
    data_per_chunk = 5000
    is_special_case = False
    if data_size > 10000:
        n_chunk = data_size // data_per_chunk
        n_chunk += 1
        is_special_case = True
    else:
        n_chunk = 1

    if is_special_case == True:
        # Divide into several chunks if the length of the data is too large
        for i in range(n_chunk-1):
            x_sensor_a.append(x_sensor_a_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :, :, :])
            x_sensor_b.append(x_sensor_b_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :, :, :])
            x_imu.append(x_imu_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :, :, :])
            y.append(y_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :])

        x_sensor_a.append(x_sensor_a_temp[(data_size - data_per_chunk):data_size, :, :, :])
        x_sensor_b.append(x_sensor_b_temp[(data_size - data_per_chunk):data_size, :, :, :])
        x_imu.append(x_imu_temp[(data_size - data_per_chunk):data_size, :, :, :])
        y.append(y_temp[(data_size - data_per_chunk):(data_size-1), :])
    else:
        x_sensor_a.append(x_sensor_a_temp[0:data_size, :, :, :])
        x_sensor_b.append(x_sensor_b_temp[0:data_size, :, :, :])
        x_imu.append(x_imu_temp[0:data_size, :, :])
        y.append(y_temp[0:(data_size-1), :])
    return n_chunk, x_time, x_sensor_a, x_sensor_b, x_imu, y


def load_data_multi_timestamp(training_file, sensor):

    # Load data
    x_sensor, x_imu, y = [], [], []
    hdf5_file = h5py.File(training_file, 'r')
    x_time = hdf5_file.get('timestamp')
    x_sensor_temp = hdf5_file.get(sensor+'_data')
    x_imu_temp = hdf5_file.get('imu_data')
    y_temp = hdf5_file.get('label_data')


    print('Data shape: ' + str(np.shape(x_sensor_temp)))

    # this is for raw data
    if x_sensor_temp.shape[0] == 1:
        x_sensor_temp = x_sensor_temp[0]

    if x_imu_temp.shape[0] == 1:
        x_imu_temp = x_imu_temp[0]
    y_temp = y_temp[0]

    print('Data shape: ' + str(np.shape(x_sensor_temp)) + str(str(np.shape(y_temp))))

    data_size = np.size(x_sensor_temp, axis=0)

    # Determine whether the data should be divided into several chunks
    # to fit in memory
    data_per_chunk = 5000
    is_special_case = False
    if data_size > 10000:
        n_chunk = data_size // data_per_chunk
        n_chunk += 1
        is_special_case = True
    else:
        n_chunk = 1

    if is_special_case == True:
        # Divide into several chunks if the length of the data is too large
        for i in range(n_chunk-1):
            x_sensor.append(x_sensor_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :, :, :])
            x_imu.append(x_imu_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :, :, :])
            y.append(y_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :])

        x_sensor.append(x_sensor_temp[(data_size - data_per_chunk):data_size, :, :, :])
        x_imu.append(x_imu_temp[(data_size - data_per_chunk):data_size, :, :, :])
        y.append(y_temp[(data_size - data_per_chunk):(data_size-1), :])
    else:
        x_sensor.append(x_sensor_temp[0:data_size, :, :, :])
        x_imu.append(x_imu_temp[0:data_size, :, :])
        y.append(y_temp[0:(data_size-1), :])
    return n_chunk, x_time, x_sensor, x_imu, y

def load_data_single_sensor(training_file, sensor):
    # Load data
    x, x_imu, y = [], [], []
    hdf5_file = h5py.File(training_file, 'r')
    x_temp = hdf5_file.get(sensor+'_data')
    # x_imu_temp = hdf5_file.get('imu_data')
    y_temp = hdf5_file.get('label_data')

    print('Data shape: ' + str(np.shape(x_temp)))

    # this is for rgb
    # x_rgb_temp = np.squeeze(x_rgb_temp, axis=0)
    # x_imu_temp = np.squeeze(x_imu_temp, axis=0)
    # y_temp = np.squeeze(y_temp, axis=0)

    # this is for raw data
    if x_temp.shape[0] == 1:
        x_temp = x_temp[0]
    # x_imu_temp = x_imu_temp[0]
    y_temp = y_temp[0]

    print('Data shape: ' + str(np.shape(x_temp)) + str(str(np.shape(y_temp))))

    data_size = np.size(x_temp, axis=0)

    # Determine whether the data should be divided into several chunks
    # to fit in memory
    data_per_chunk = 5000
    is_special_case = False
    if data_size > 10000:
        n_chunk = data_size // data_per_chunk
        n_chunk += 1
        is_special_case = True
    else:
        n_chunk = 1

    if is_special_case == True:
        # Divide into several chunks if the length of the data is too large
        for i in range(n_chunk-1):
            x.append(x_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :, :, :])
            # x_imu.append(x_imu_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :, :, :])
            y.append(y_temp[(data_per_chunk * i):data_per_chunk * (i + 1), :])
        # 2300
        x.append(x_temp[(data_size - data_per_chunk):data_size, :, :, :])
        # x_imu.append(x_imu_temp[(data_size - data_per_chunk):data_size, :, :, :])
        y.append(y_temp[(data_size - data_per_chunk):(data_size-1), :])
    else:
        x.append(x_temp[0:data_size, :, :, :])
        # x_imu.append(x_imu_temp[0:data_size, :, :])
        y.append(y_temp[0:(data_size-1), :])
    return n_chunk, x, y


def validation_stack(validation_files, sensor='mmwave_middle', imu_length=0):
    x_sensor_val_1, x_sensor_val_2, x_imu_val_t, y_val_t = [], [], [], []
    for validation_file in validation_files:
        print('---> Loading validation file: {}'.format(validation_file.split('/')[-1]))
        if imu_length:
            n_chunk_val, tmp_x_sensor_val_t, tmp_x_imu_val_t, tmp_y_val_t = load_data_multi(validation_file, sensor) # y (1, 2142, 6)
        else:
            n_chunk_val, tmp_x_sensor_val_t, tmp_y_val_t = load_data_single_sensor(validation_file, sensor)

        tmp_y_val_t = tmp_y_val_t[0]
        tmp_y_val_t = np.expand_dims(tmp_y_val_t, axis=1)

        len_val_i = tmp_y_val_t.shape[0] # the length of gt is always less than the length of data
        # Prepare rgb validation data for t-0 and t-1
        tmp_x_sensor_val_1 = []
        for img_idx in range(0, (len_val_i)):
            temp_x = tmp_x_sensor_val_t[0][img_idx]
            tmp_x_sensor_val_1.append(temp_x)

        tmp_x_sensor_val_1 = np.array(tmp_x_sensor_val_1)
        # x_rgb_val_1 = np.expand_dims(x_rgb_val_1, axis=1)

        tmp_x_sensor_val_2 = []
        for img_idx in range(1, (len_val_i+1)):
            temp_x = tmp_x_sensor_val_t[0][img_idx]
            tmp_x_sensor_val_2.append(temp_x)

        tmp_x_sensor_val_2 = np.array(tmp_x_sensor_val_2)

        # for flownet
        if any(x in sensor for x in ['mmwave', 'depth']):
            tmp_x_sensor_val_1 = np.repeat(tmp_x_sensor_val_1, 3, axis=-1)
            tmp_x_sensor_val_2 = np.repeat(tmp_x_sensor_val_2, 3, axis=-1)

        # progressive stack file by file
        y_val_t = np.vstack((y_val_t, tmp_y_val_t)) if np.array(y_val_t).size else tmp_y_val_t
        x_sensor_val_1 = np.vstack((x_sensor_val_1, tmp_x_sensor_val_1)) if np.array(x_sensor_val_1).size else tmp_x_sensor_val_1
        x_sensor_val_2 = np.vstack((x_sensor_val_2, tmp_x_sensor_val_2)) if np.array(x_sensor_val_2).size else tmp_x_sensor_val_2

        if imu_length:
            # for imu
            tmp_x_imu_val_t = tmp_x_imu_val_t[0]
            tmp_x_imu_val_t = tmp_x_imu_val_t[:, 0:imu_length, :]
            tmp_x_imu_val_t = np.array(tmp_x_imu_val_t)

            # add data
            x_imu_val_t = np.vstack((x_imu_val_t, tmp_x_imu_val_t)) \
                if np.array(x_imu_val_t).size else tmp_x_imu_val_t

    if imu_length:
        return x_sensor_val_1, x_sensor_val_2, x_imu_val_t, y_val_t
    else:
        return x_sensor_val_1, x_sensor_val_2, y_val_t


def validation_stack_triple(validation_files, sensor_a='mmwave_middle', sensor_b='rgb', imu_length=0):
    x_sensor_a_val_1, x_sensor_a_val_2, x_sensor_b_val_1, x_sensor_b_val_2, x_imu_val_t, y_val_t = [], [], [], [], [], []
    for validation_file in validation_files:
        print('---> Loading validation file: {}'.format(validation_file.split('/')[-1]))

        n_chunk_val, tmp_x_t, tmp_x_sensor_a_val_t, tmp_x_sensor_b_val_t, tmp_x_imu_val_t, tmp_y_val_t = \
            load_data_triple_timestamp(validation_file, sensor_a=sensor_a, sensor_b=sensor_b)

        tmp_y_val_t = tmp_y_val_t[0]
        tmp_y_val_t = np.expand_dims(tmp_y_val_t, axis=1)

        len_val_i = tmp_y_val_t.shape[0] # the length of gt is always less than the length of data
        # Prepare rgb validation data for t-0 and t-1
        tmp_x_sensor_a_val_1 = []
        for img_idx in range(0, (len_val_i)):
            temp_x = tmp_x_sensor_a_val_t[0][img_idx]
            tmp_x_sensor_a_val_1.append(temp_x)

        tmp_x_sensor_a_val_1 = np.array(tmp_x_sensor_a_val_1)

        tmp_x_sensor_b_val_1 = []
        for img_idx in range(0, (len_val_i)):
            temp_x = tmp_x_sensor_b_val_t[0][img_idx]
            tmp_x_sensor_b_val_1.append(temp_x)

        tmp_x_sensor_b_val_1 = np.array(tmp_x_sensor_b_val_1)

        tmp_x_sensor_a_val_2 = []
        for img_idx in range(1, (len_val_i+1)):
            temp_x = tmp_x_sensor_a_val_t[0][img_idx]
            tmp_x_sensor_a_val_2.append(temp_x)

        tmp_x_sensor_a_val_2 = np.array(tmp_x_sensor_a_val_2)

        tmp_x_sensor_b_val_2 = []
        for img_idx in range(1, (len_val_i + 1)):
            temp_x = tmp_x_sensor_b_val_t[0][img_idx]
            tmp_x_sensor_b_val_2.append(temp_x)

        tmp_x_sensor_b_val_2 = np.array(tmp_x_sensor_b_val_2)

        # for flownet
        if any(x in sensor_a for x in ['mmwave', 'depth']):
            tmp_x_sensor_a_val_1 = np.repeat(tmp_x_sensor_a_val_1, 3, axis=-1)
            tmp_x_sensor_a_val_2 = np.repeat(tmp_x_sensor_a_val_2, 3, axis=-1)

        if any(x in sensor_b for x in ['mmwave', 'depth']):
            tmp_x_sensor_b_val_1 = np.repeat(tmp_x_sensor_b_val_1, 3, axis=-1)
            tmp_x_sensor_b_val_2 = np.repeat(tmp_x_sensor_b_val_2, 3, axis=-1)

        # progressive stack file by file
        y_val_t = np.vstack((y_val_t, tmp_y_val_t)) if np.array(y_val_t).size else tmp_y_val_t

        x_sensor_a_val_1 = np.vstack((x_sensor_a_val_1, tmp_x_sensor_a_val_1)) \
            if np.array(x_sensor_a_val_1).size else tmp_x_sensor_a_val_1
        x_sensor_a_val_2 = np.vstack((x_sensor_a_val_2, tmp_x_sensor_a_val_2)) \
            if np.array(x_sensor_a_val_2).size else tmp_x_sensor_a_val_2

        x_sensor_b_val_1 = np.vstack((x_sensor_b_val_1, tmp_x_sensor_b_val_1)) \
            if np.array(x_sensor_b_val_1).size else tmp_x_sensor_b_val_1
        x_sensor_b_val_2 = np.vstack((x_sensor_b_val_2, tmp_x_sensor_b_val_2)) \
            if np.array(x_sensor_b_val_2).size else tmp_x_sensor_b_val_2

        if imu_length:
            # for imu
            tmp_x_imu_val_t = tmp_x_imu_val_t[0]
            tmp_x_imu_val_t = tmp_x_imu_val_t[:, 0:imu_length, :]
            tmp_x_imu_val_t = np.array(tmp_x_imu_val_t)

            # add data
            x_imu_val_t = np.vstack((x_imu_val_t, tmp_x_imu_val_t)) \
                if np.array(x_imu_val_t).size else tmp_x_imu_val_t

    if imu_length:
        return x_sensor_a_val_1, x_sensor_a_val_2, x_sensor_b_val_1, x_sensor_b_val_2, x_imu_val_t, y_val_t
    else:
        return x_sensor_a_val_1, x_sensor_a_val_2, x_sensor_b_val_1, x_sensor_b_val_2, y_val_t