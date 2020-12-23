import os
from os.path import join
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Model, Sequential  # Sequential, Graph
from keras.models import model_from_json, model_from_yaml, load_model
from keras.layers import (Input, Dense, Convolution2D, Activation, Flatten, Dropout, AveragePooling1D, MaxPooling2D,
                          GlobalAveragePooling1D, ZeroPadding2D, Merge, Reshape, merge, Multiply, Add, Lambda, AveragePooling2D, GlobalAveragePooling2D, LSTM, TimeDistributed)
from keras.layers.advanced_activations import LeakyReLU, ELU
import keras
from keras import backend as K

K.set_image_dim_ordering('tf')
K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))


def FlowNetModule(input, dup=False):
    # inout must follow the follow the shape format: (1, H, W, C)
    if not dup:
        net = TimeDistributed(Convolution2D(64, 7, 7, subsample=(2, 2), border_mode='same'), name='conv1')(input)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU1')(net)
        net = TimeDistributed(Convolution2D(128, 5, 5, subsample=(2, 2), border_mode='same'), name='conv2')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU2')(net)
        net = TimeDistributed(Convolution2D(256, 5, 5, subsample=(2, 2), border_mode='same'), name='conv3')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU3')(net)
        net = TimeDistributed(Convolution2D(256, 3, 3, subsample=(1, 1), border_mode='same'), name='conv3_1')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU4')(net)
        net = TimeDistributed(Convolution2D(512, 3, 3, subsample=(2, 2), border_mode='same'), name='conv4')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU5')(net)
        net = TimeDistributed(Convolution2D(512, 3, 3, subsample=(1, 1), border_mode='same'), name='conv4_1')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU6')(net)
        net = TimeDistributed(Convolution2D(512, 3, 3, subsample=(2, 2), border_mode='same'), name='conv5')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU7')(net)
        net = TimeDistributed(Convolution2D(512, 3, 3, subsample=(1, 1), border_mode='same'), name='conv5_1')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU8')(net)
        net = TimeDistributed(Convolution2D(1024, 3, 3, subsample=(2, 2), border_mode='same'), name='conv6')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU9')(net)
    else:
        net = TimeDistributed(Convolution2D(64, 7, 7, subsample=(2, 2), border_mode='same'), name='conv1'+'_dup')(input)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU1'+'_dup')(net)
        net = TimeDistributed(Convolution2D(128, 5, 5, subsample=(2, 2), border_mode='same'), name='conv2'+'_dup')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU2'+'_dup')(net)
        net = TimeDistributed(Convolution2D(256, 5, 5, subsample=(2, 2), border_mode='same'), name='conv3'+'_dup')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU3'+'_dup')(net)
        net = TimeDistributed(Convolution2D(256, 3, 3, subsample=(1, 1), border_mode='same'), name='conv3_1'+'_dup')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU4'+'_dup')(net)
        net = TimeDistributed(Convolution2D(512, 3, 3, subsample=(2, 2), border_mode='same'), name='conv4'+'_dup')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU5'+'_dup')(net)
        net = TimeDistributed(Convolution2D(512, 3, 3, subsample=(1, 1), border_mode='same'), name='conv4_1'+'_dup')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU6'+'_dup')(net)
        net = TimeDistributed(Convolution2D(512, 3, 3, subsample=(2, 2), border_mode='same'), name='conv5'+'_dup')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU7'+'_dup')(net)
        net = TimeDistributed(Convolution2D(512, 3, 3, subsample=(1, 1), border_mode='same'), name='conv5_1'+'_dup')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU8'+'_dup')(net)
        net = TimeDistributed(Convolution2D(1024, 3, 3, subsample=(2, 2), border_mode='same'), name='conv6'+'_dup')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU9'+'_dup')(net)
    return net

################################################################
# model building functions for lidar, radar and panoramic inputs.
# cfg is downgraded to model path if training_flag is False
################################################################

# Build CROSS-attentive multi-modal odom with IMU always used
def build_model_cross_att(cfg, imu_length=20, input_shape=(1, 64, 256, 3), mask_att='sigmoid', istraining=True, write_mask=False):
    image_1 = Input(shape=input_shape, name='image_1')
    image_2 = Input(shape=input_shape, name='image_2')

    if K.image_dim_ordering() == 'th':
        image_merged = merge([image_1, image_2], mode='concat', concat_axis=1)
    else:
        image_merged = merge([image_1, image_2], mode='concat', concat_axis=-1)

    # --- panoramic image data
    net = FlowNetModule(image_merged)

    # generate the mask for visual features
    visual_mask = TimeDistributed(GlobalAveragePooling2D())(net) # reshape to (?, 1, 1024), 1 stands for timeDistr.
    visual_mask = TimeDistributed(Dense(int(1024/256), activation='relu', use_bias=False, name='visual_mask_relu'))(visual_mask)
    visual_mask = TimeDistributed(Dense(1024, activation='sigmoid', use_bias=False, name='visual_mask_sigmoid'))(visual_mask)
    visual_mask = Reshape((1, 1, 1, 1024))(visual_mask)

    # activate mask by element-wise multiplication
    visual_att_fea = Multiply()([net, visual_mask])
    visual_att_fea = TimeDistributed(Flatten(), name='flatten')(visual_att_fea)

    # IMU data
    imu_data = Input(shape=(imu_length, 6), name='imu_data')
    imu_lstm_1 = LSTM(128, return_sequences=True, name='imu_lstm_1')(imu_data)  # 128, 256

    # channel-wise IMU attention
    reshape_imu = Reshape((1, imu_length * 128))(imu_lstm_1)  # 2560, 5120, 10240
    imu_mask = Dense(128, activation='relu', use_bias=False, name='imu_mask_relu')(reshape_imu)
    imu_mask = Dense(imu_length * 128, activation='sigmoid', use_bias=False, name='imu_mask_sigmoid')(imu_mask)
    imu_att_fea = Multiply()([reshape_imu, imu_mask])

    # cross-modal attention
    imu4visual_mask = Dense(128, activation='relu', use_bias=False, name='imu4visual_mask_relu')(imu_att_fea)
    imu4visual_mask = Dense(4096, activation=mask_att, use_bias=False, name='imu4visual_mask_sigmoid')(imu4visual_mask)
    cross_visual_fea = Multiply()([visual_att_fea, imu4visual_mask])

    visual4imu_mask = Dense(128, activation='relu', use_bias=False, name='visual4imu_mask_relu')(visual_att_fea)
    visual4imu_mask = Dense(imu_length * 128, activation=mask_att, use_bias=False, name='visual4imu_mask_sigmoid')(visual4imu_mask)
    cross_imu_fea = Multiply()([imu_att_fea, visual4imu_mask])

    # Standard merge feature
    merge_features = merge([cross_visual_fea, cross_imu_fea], mode='concat', concat_axis=-1, name='merge_features')

    # Selective features
    forward_lstm_1 = LSTM(512, dropout_W=0.25, return_sequences=True, name='forward_lstm_1')(
        merge_features)  # dropout_W=0.2, dropout_U=0.2
    forward_lstm_2 = LSTM(512, return_sequences=True, name='forward_lstm_2')(forward_lstm_1)

    fc_position_1 = TimeDistributed(Dense(128, activation='relu'), name='fc_position_1')(forward_lstm_2)  # tanh
    dropout_pos_1 = TimeDistributed(Dropout(0.25), name='dropout_pos_1')(fc_position_1)
    fc_position_2 = TimeDistributed(Dense(64, activation='relu'), name='fc_position_2')(dropout_pos_1)  # tanh
    fc_trans = TimeDistributed(Dense(3), name='fc_trans')(fc_position_2)

    fc_orientation_1 = TimeDistributed(Dense(128, activation='relu'), name='fc_orientation_1')(forward_lstm_2)  # tanh
    dropout_orientation_1 = TimeDistributed(Dropout(0.25), name='dropout_wpqr_1')(fc_orientation_1)
    fc_orientation_2 = TimeDistributed(Dense(64, activation='relu'), name='fc_orientation_2')(
        dropout_orientation_1)  # tanh
    fc_rot = TimeDistributed(Dense(3), name='fc_rot')(fc_orientation_2)

    # define model
    if istraining:
        model = Model(input=[image_1, image_2, imu_data], output=[fc_trans, fc_rot])

        model_path = join('./models', 'cnn.h5')
        model.load_weights(model_path, by_name=True)

        for layer in model.layers[0:22]:  # all -11, 22 (freeze all cnn), 19 (train last cnn)
            layer.trainable = False

        # configure learning process with compile()
        rmsProp = keras.optimizers.RMSprop(lr=cfg['lr_rate'], rho=cfg['rho'],
                                           epsilon=float(cfg['epsilon']),
                                           decay=cfg['decay'])
        model.compile(optimizer=rmsProp, loss={'fc_trans': 'mse', 'fc_rot': 'mse'},
                      loss_weights={'fc_trans': cfg['fc_trans'],
                                    'fc_rot': cfg['fc_rot']})
    else:
        if write_mask:
            mask_merge = merge([imu4visual_mask, visual4imu_mask],
                                    mode='concat', concat_axis=-1, name='mask_merge')
            model = Model(input=[image_1, image_2, imu_data], output=[mask_merge])
        else:
            fc_merge = merge([fc_trans, fc_rot], mode='concat', concat_axis=-1, name='delta_pose')
            model = Model(input=[image_1, image_2, imu_data], output=[fc_merge])

        for layer in model.layers[:]:
            layer.trainable = False

        # load weights
        model.load_weights(cfg, by_name=True)
    return model
