from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, add, multiply
from keras.layers import concatenate, core, Dropout
from keras.models import Model
from keras.layers.merge import concatenate
from tensorflow.keras.optimizers import Adam
from keras.layers.core import Lambda
import keras.backend as K
from tensorflow.keras.losses import binary_crossentropy 
from tensorflow.keras.metrics import mean_squared_error
import tensorflow as tf
import math
import tensorflow.keras.backend as K
import pywt
import pywt.data
import torch

def attention_block_2d(x, g, inter_channel, data_format='channels_last'):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = multiply([x, rate])

    return att_x

def Shape_aware_unit(inputs, inter_channel, data_format):
    
    conv1 = Conv2D(inter_channel, (1, 1), padding='same',data_format=data_format)(inputs)
    conv2 = Conv2D(inter_channel/2, (3, 3), padding='same',data_format=data_format)(conv1)
    conv3 = Conv2D(inter_channel/4, (3, 3), padding='same',data_format=data_format)(conv2)
    conca = tf.keras.layers.Concatenate(axis=3)([inputs, conv3])
    out_shape = Conv2D(16, (1, 1), padding='same',data_format=data_format)(conv2)

    return conca, out_shape

def shape_fusion(out_shape1, out_shape2, out_shape3):
    out_shape1 = UpSampling2D(size=(8, 8), data_format='channels_last', interpolation='bilinear', name='shape11')(
        out_shape1)
    out_shape2 = UpSampling2D(size=(4, 4), data_format='channels_last', interpolation='bilinear', name='shape22')(
        out_shape2)
    out_shape3 = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear', name='shape33')(
        out_shape3)
    shape_fuse = concatenate([out_shape1, out_shape2, out_shape3], axis=3, name='cc5')
    out_shape = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', name='out_shape',
                       kernel_initializer='he_normal')(shape_fuse)
    return out_shape

def wavelet_concat(LH, HL, HH, FM, FU, FD, inter_channel, data_format='channels_last'):

    LH = Conv2D(inter_channel, [1, 1], strides=[1, 1], padding='same', data_format=data_format)(LH)
    # print(LH.shape)
    HL = Conv2D(inter_channel, [1, 1], strides=[1, 1], padding='same', data_format=data_format)(HL)
    # print(HL.shape)    
    HH = Conv2D(inter_channel, [1, 1], strides=[1, 1], padding='same', data_format=data_format)(HH)
    # print(HH.shape)
    W = tf.keras.layers.Concatenate(axis=3)([LH, HL, HH])
    # print(FU.shape)
    # print(FD.shape)
    F = tf.keras.layers.Concatenate(axis=3)([FU, FD])
    # print(F.shape)
    F1 = UpSampling2D(size=(2, 2), data_format=data_format)(FM)
    # print(F1.shape)
    F1 = Conv2D(inter_channel*2, (1, 1), dilation_rate=2, padding='same',data_format=data_format)(F1)
    # print(F1.shape)
    F = add([F, F1])
    F = attention_block_2d(F, W, inter_channel = F.get_shape().as_list()[3], data_format='channels_last')
    conca, out_shape = Shape_aware_unit(F, inter_channel, data_format)
    return conca, out_shape


def dwt(x, data_format='channels_last'):

    """
    DWT (Discrete Wavelet Transform) function implementation according to
    "Multi-level Wavelet Convolutional Neural Networks"
    by Pengju Liu, Hongzhi Zhang, Wei Lian, Wangmeng Zuo
    https://arxiv.org/abs/1907.03128
    """

    if data_format == 'channels_last':
        x1 = x[:, 0::2, 0::2, :] #x(2i−1, 2j−1)
        x2 = x[:, 1::2, 0::2, :] #x(2i, 2j-1)
        x3 = x[:, 0::2, 1::2, :] #x(2i−1, 2j)
        x4 = x[:, 1::2, 1::2, :] #x(2i, 2j)

    elif data_format == 'channels_first':
        x1 = x[:, :, 0::2, 0::2] #x(2i−1, 2j−1)
        x2 = x[:, :, 1::2, 0::2] #x(2i, 2j-1)
        x3 = x[:, :, 0::2, 1::2] #x(2i−1, 2j)
        x4 = x[:, :, 1::2, 1::2] #x(2i, 2j)     

    x_LL = x1 + x2 + x3 + x4
    x_LH = -x1 - x3 + x2 + x4
    x_HL = -x1 + x3 - x2 + x4
    x_HH = x1 - x3 - x2 + x4

    if data_format == 'channels_last':
        return x_LL,x_LH,x_HL,x_HH
    elif data_format == 'channels_first':
        return x_LL,x_LH,x_HL,x_HH


def SAWGUnet(img_w, img_h, n_label, data_format='channels_last', features=8):



    inp = Input(shape=(256, 256, 3))
    
    LL1, HL1, LH1, HH1  = dwt(inp, data_format='channels_last')
    print(LL1.shape)
    LL2, HL2, LH2, HH2  = dwt(LL1, data_format='channels_last')
    print(LL2.shape)
    LL3, HL3, LH3, HH3  = dwt(LL2, data_format='channels_last')
    print(LL3.shape)
    

    # encoder
    C10 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(inp)
    C10 = BatchNormalization()(C10)
    C11 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C10)
    C11 = BatchNormalization()(C11)
    FD1 = Conv2D(features, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(C11)
    # print(FD1.shape)
    C20 = Conv2D(features*2, (3, 3), activation='relu', padding='same', data_format=data_format)(FD1)
    C20 = BatchNormalization()(C20)
    C21 = Conv2D(features*2, (3, 3), activation='relu', padding='same', data_format=data_format)(C20)
    C21 = BatchNormalization()(C21)
    FD2 = Conv2D(features*2, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(C21)
    # print(FD2.shape)
    C30 = Conv2D(features*4, (3, 3), activation='relu', padding='same', data_format=data_format)(FD2)
    C30 = BatchNormalization()(C30)
    C31 = Conv2D(features*4, (3, 3), activation='relu', padding='same', data_format=data_format)(C30)
    C31 = BatchNormalization()(C31)
    FD3 =  Conv2D(features*4, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(C31)
    # print(FD3.shape)
    C41 = Conv2D(features*4, (3, 3), activation='relu', padding='same', data_format=data_format)(FD3)
    C41 = BatchNormalization()(C41)
    C42 = Conv2D(features*4, (3, 3), activation='relu', padding='same', data_format=data_format)(C41)
    C42 = BatchNormalization()(C42)
    U1L = Dropout(0.5)(C42)
    U1 = Conv2D(features*4, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format=data_format)(U1L)
    # print(U1.shape)
    # 

    C50 = Conv2D(features*4, (3, 3), activation='relu', padding='same', data_format=data_format)(U1L)
    C50 = BatchNormalization()(C50)
    C51 = Conv2D(features*4, (3, 3), activation='relu', padding='same', data_format=data_format)(C50)
    C51 = BatchNormalization()(C51)
    U2 = UpSampling2D(size=(2, 2), data_format=data_format)(C51)
    # print(U2.shape)
    C60 = Conv2D(features*2, (3, 3), activation='relu', padding='same', data_format=data_format)(U2)
    C60 = BatchNormalization()(C60)
    C61 = Conv2D(features*2, (3, 3), activation='relu', padding='same', data_format=data_format)(C60)
    C61 = BatchNormalization()(C61)
    U3 = UpSampling2D(size=(2, 2), data_format=data_format)(C61)

    C70 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(U3)
    C70 = BatchNormalization()(C70)
    C71 = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(C70)
    C71 = BatchNormalization()(C71)
    U4 = UpSampling2D(size=(2, 2), data_format=data_format)(C71)

    Ffinal = Conv2D(1, (1, 1), padding='same', data_format=data_format)(U4)
    Ffinal = core.Activation('sigmoid')(Ffinal)

    M1, shape_1 = wavelet_concat(LH3, HL3, HH3, U1, FD3, C51, features*4, data_format=data_format)
    M2, shape_2 = wavelet_concat(LH2, HL2, HH2, M1, FD2, C61, features*2, data_format=data_format)
    M3, shape_3 = wavelet_concat(LH1, HL1, HH1, M2, FD1, C71, features, data_format=data_format)

    shape_fused = shape_fusion(shape_1, shape_2, shape_3)
    shape_fused = Conv2D(1, (1, 1), padding='same', data_format=data_format)(shape_fused)
    shape_fused = core.Activation('sigmoid')(shape_fused)


    M3 = UpSampling2D(size=(2, 2), data_format=data_format)(M3)
    Mfinal = Conv2D(1, (1, 1), padding='same', data_format=data_format)(M3)
    Mfinal = core.Activation('sigmoid')(Mfinal)   
    
    model = Model(inputs=inp, outputs=[Mfinal, Ffinal, shape_fused], name='Unet_DSRN')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=[bce_dice_loss, bce_dice_loss, mean_squared_error], loss_weights= [1., 1., 1.], metrics=dice_coefficient)

    return model
  
 
model1 = SAWGUnet(256, 256, 3, data_format='channels_last', features = 8)
model1.summary()
history1 = model1.fit(X_train, [Y_train.astype('float32'), Y_train.astype('float32'), Y_train.astype('float32')], batch_size = 16, validation_split=0.1,  epochs= 70, shuffle=True, callbacks = callbacks_list)

