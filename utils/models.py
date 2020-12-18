#%%
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam,SGD
from keras import backend as K
from utils.dice import dice_loss, soft_cldice_loss
from keras.utils import plot_model

# Define optimizer
# optmz = Adam(lr=0.0001)
optmz = SGD(lr=0.1, momentum=0.8)

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# combine dice + cldice similiar to the experiments in the paper
def combined_loss(y_true, y_pred):
    alpha = 0.5
    data_format = "channels_last"
    return alpha * dice_loss(data_format=data_format)(y_true, y_pred) + (
        1 - alpha
    ) * soft_cldice_loss(k=5, data_format=data_format)(y_true, y_pred)


# Build U-Net model
# def get_unet(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH):
#     inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#     s = Lambda(lambda x: x / 255)(inputs)
#     kernel = 3

#     # ======DownSample=======
#     conv1 = Conv2D(
#         16,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(s)
#     conv1 = Dropout(0.1)(conv1)
#     conv1 = Conv2D(
#         16,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(conv1)
#     pool1 = MaxPooling2D((2, 2))(conv1)

#     conv2 = Conv2D(
#         32,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(pool1)
#     conv2 = Dropout(0.1)(conv2)
#     conv2 = Conv2D(
#         32,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(conv2)
#     pool2 = MaxPooling2D((2, 2))(conv2)

#     conv3 = Conv2D(
#         64,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(pool2)
#     conv3 = Dropout(0.2)(conv3)
#     conv3 = Conv2D(
#         64,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(conv3)
#     pool3 = MaxPooling2D((2, 2))(conv3)

#     conv4 = Conv2D(
#         128,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(pool3)
#     conv4 = Dropout(0.2)(conv4)
#     conv4 = Conv2D(
#         128,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(conv4)
#     pool4 = MaxPooling2D((2, 2))(conv4)
#     # ======DownSample=======

#     # ========Bridge=========
#     conv5 = Conv2D(
#         256,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(pool4)
#     conv5 = Dropout(0.3)(conv5)
#     conv5 = Conv2D(
#         256,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(conv5)
#     # ========Bridge=========

#     # =======UpSample========
#     up6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(conv5)
#     up6 = concatenate([up6, conv4])
#     conv6 = Conv2D(
#         128,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(up6)
#     conv6 = Dropout(0.2)(conv6)
#     conv6 = Conv2D(
#         128,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(conv6)

#     up7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(conv6)
#     up7 = concatenate([up7, conv3])
#     conv7 = Conv2D(
#         64,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(up7)
#     conv7 = Dropout(0.2)(conv7)
#     conv7 = Conv2D(
#         64,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(conv7)

#     up8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(conv7)
#     up8 = concatenate([up8, conv2])
#     conv8 = Conv2D(
#         32,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(up8)
#     conv8 = Dropout(0.1)(conv8)
#     conv8 = Conv2D(
#         32,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(conv8)

#     up9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(conv8)
#     up9 = concatenate([up9, conv1], axis=3)
#     conv9 = Conv2D(
#         16,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(up9)
#     conv9 = Dropout(0.1)(conv9)
#     conv9 = Conv2D(
#         16,
#         (kernel, kernel),
#         activation="elu",
#         kernel_initializer="he_normal",
#         padding="same",
#     )(conv9)
#     # =======UpSample========
#     outputs = Conv2D(1, (1, 1), activation="sigmoid")(conv9)

#     model = Model(inputs=[inputs], outputs=[outputs])
#     model.compile(optimizer=optmz, loss=[combined_loss], metrics=[mean_iou])
#     model.summary()
#     return model
def conv_block_1(i, Base, acti, bn):
    n = Conv2D(Base, (3,3), padding='same')(i)
    n = BatchNormalization()(n) if bn else n
    n = Activation(acti)(n)
    n = Conv2D(Base, (3,3), padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    o = Activation(acti)(n)
    return o

def conv_block_2(i, Base, acti, bn, drop):
    n = MaxPooling2D(pool_size=(2, 2))(i)
    n = Dropout(drop)(n) if drop else n
    o = conv_block_1(n, Base, acti, bn)
    return o

def conv_block_3(i, conca_i, Base, acti, bn, drop):
    n = Conv2DTranspose(Base, (2, 2), strides=(2, 2), padding='same')(i)
    n = concatenate([n, conca_i], axis=3)
    n = Dropout(drop)(n) if drop else n
    o = conv_block_1(n, Base, acti, bn)
    return o

def get_UNet(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, Base, depth, inc_rate, activation, drop, batchnorm, N):
    i = Input((IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))
    # i = Input(shape=img_shape)
    x_conca = []
    n = conv_block_1(i, Base, activation, batchnorm)
    x_conca.append(n)
    for k in range(depth):
        Base = Base*inc_rate
        n = conv_block_2(n, Base, activation, batchnorm, drop)
        if k < (depth-1):
            x_conca.append(n)
    for k in range(depth):
        Base = Base//inc_rate
        n = conv_block_3(n, x_conca[-1-k], Base, activation, batchnorm, drop)
    
    if N == 1:
        o = Conv2D(1, (1,1), activation='sigmoid')(n)
    else:
        o = Conv2D(N, (1,1), activation='softmax')(n)
    
    model = Model(inputs=i, outputs=o)
    # model.compile(optimizer=optmz, loss='binary_crossentropy',metrics=[mean_iou])
    model.compile(optimizer=optmz, loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model

# Build SegUNet model
from keras.layers import BatchNormalization
from keras.layers.core import Activation, Reshape
from keras.layers.merge import Concatenate
from utils.layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


def get_segunet(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    kernel = 3
    pool_size = (2, 2)
    # ======DownSample=======
    conv_1 = Conv2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Conv2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Conv2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Conv2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Conv2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Conv2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Conv2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Conv2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Conv2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Conv2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Conv2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Conv2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Conv2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    # ======DownSample=======

    # ========Bridge=========
    conv_14 = Conv2D(512, (kernel, kernel), padding="same")(pool_5)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Conv2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Conv2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)
    # ========Bridge=========

    # =======UpSample========
    unpool_1 = MaxUnpooling2D(pool_size)([conv_16, mask_5])
    concat_1 = Concatenate()([unpool_1, conv_13])

    conv_17 = Conv2D(512, (kernel, kernel), padding="same")(concat_1)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Conv2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Conv2D(512, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_19, mask_4])
    concat_2 = Concatenate()([unpool_2, conv_10])

    conv_20 = Conv2D(512, (kernel, kernel), padding="same")(concat_2)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Conv2D(512, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Conv2D(256, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_22, mask_3])
    concat_3 = Concatenate()([unpool_3, conv_7])

    conv_23 = Conv2D(256, (kernel, kernel), padding="same")(concat_3)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Conv2D(256, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)
    conv_25 = Conv2D(128, (kernel, kernel), padding="same")(conv_24)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_25, mask_2])
    concat_4 = Concatenate()([unpool_4, conv_4])

    conv_26 = Conv2D(128, (kernel, kernel), padding="same")(concat_4)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Activation("relu")(conv_26)
    conv_27 = Conv2D(64, (kernel, kernel), padding="same")(conv_26)
    conv_27 = BatchNormalization()(conv_27)
    conv_27 = Activation("relu")(conv_27)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_27, mask_1])
    concat_5 = Concatenate()([unpool_5, conv_2])

    conv_28 = Conv2D(64, (kernel, kernel), padding="same")(concat_5)
    conv_28 = BatchNormalization()(conv_28)
    conv_28 = Activation("relu")(conv_28)
    conv_29 = Conv2D(1, (1, 1), padding="valid")(conv_28)
    conv_29 = BatchNormalization()(conv_29)
    conv_29 = Activation("relu")(conv_29)
    # =======UpSample========

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(conv_29)

    model = Model(inputs=[inputs], outputs=[outputs], name="SegUNet")
    model.compile(optimizer=optmz, loss=[combined_loss], metrics=[mean_iou])
    model.summary()
    return model


from keras.layers import UpSampling2D
from keras.regularizers import l2

# Build SegNet model
def get_segnet(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    kernel = 3
    pool_size = (2, 2)

    # ======DownSample=======
    conv1 = Conv2D(
        64, (kernel, kernel), padding="same", kernel_initializer="orthogonal"
    )(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = MaxPooling2D(pool_size)(conv1)

    conv2 = Conv2D(
        128, (kernel, kernel), padding="same", kernel_initializer="orthogonal"
    )(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = MaxPooling2D(pool_size)(conv2)

    conv3 = Conv2D(
        256, (kernel, kernel), padding="same", kernel_initializer="orthogonal"
    )(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    conv3 = MaxPooling2D(pool_size)(conv3)

    conv4 = Conv2D(
        512, (kernel, kernel), padding="same", kernel_initializer="orthogonal"
    )(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    conv4 = MaxPooling2D(pool_size)(conv4)

    conv5 = Conv2D(
        512, (kernel, kernel), padding="same", kernel_initializer="orthogonal"
    )(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)
    conv5 = MaxPooling2D(pool_size)(conv5)
    # ======DownSample=======

    # =======UpSample========
    up1 = UpSampling2D(size=(2, 2))(conv5)
    up1 = Conv2D(
        512, (kernel, kernel), padding="same", kernel_initializer="orthogonal"
    )(up1)
    up1 = BatchNormalization()(up1)

    up2 = UpSampling2D(size=(2, 2))(up1)
    up2 = Conv2D(
        512, (kernel, kernel), padding="same", kernel_initializer="orthogonal"
    )(up2)
    up2 = BatchNormalization()(up2)

    up3 = UpSampling2D(size=(2, 2))(up2)
    up3 = Conv2D(
        256, (kernel, kernel), padding="same", kernel_initializer="orthogonal"
    )(up3)
    up3 = BatchNormalization()(up3)

    up4 = UpSampling2D(size=(2, 2))(up3)
    up4 = Conv2D(
        128, (kernel, kernel), padding="same", kernel_initializer="orthogonal"
    )(up4)
    up4 = BatchNormalization()(up4)

    up5 = UpSampling2D(size=(2, 2))(up4)
    up5 = Conv2D(64, (kernel, kernel), padding="same", kernel_initializer="orthogonal")(
        up5
    )
    up5 = BatchNormalization()(up5)
    # =======UpSample========

    x = Conv2D(
        1,
        (1, 1),
        padding="valid",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(0.005),
    )(up5)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(x)

    model = Model(inputs=[inputs], outputs=[outputs], name="SegNet")
    model.compile(optimizer=optmz, loss=[combined_loss], metrics=[mean_iou])
    model.summary()
    return model


# Build Unet++ model
from keras.layers import AvgPool2D


def conv_batchnorm_relu_block(input_tensor, nb_filter, kernel_size=3):
    x = Conv2D(nb_filter, (kernel_size, kernel_size), padding="same")(input_tensor)
    x = BatchNormalization(axis=2)(x)
    x = Activation("relu")(x)
    return x


def get_unetpp(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, using_deep_supervision=False):

    nb_filter = [32, 64, 128, 256, 512]
    bn_axis = -1
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    conv1_1 = conv_batchnorm_relu_block(inputs, nb_filter=nb_filter[0])
    pool1 = AvgPool2D((2, 2), strides=(2, 2), name="pool1")(conv1_1)

    conv2_1 = conv_batchnorm_relu_block(pool1, nb_filter=nb_filter[1])
    pool2 = AvgPool2D((2, 2), strides=(2, 2), name="pool2")(conv2_1)

    up1_2 = Conv2DTranspose(
        nb_filter[0], (2, 2), strides=(2, 2), name="up12", padding="same"
    )(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name="merge12", axis=bn_axis)
    conv1_2 = conv_batchnorm_relu_block(conv1_2, nb_filter=nb_filter[0])

    conv3_1 = conv_batchnorm_relu_block(pool2, nb_filter=nb_filter[2])
    pool3 = AvgPool2D((2, 2), strides=(2, 2), name="pool3")(conv3_1)

    up2_2 = Conv2DTranspose(
        nb_filter[1], (2, 2), strides=(2, 2), name="up22", padding="same"
    )(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name="merge22", axis=bn_axis)
    conv2_2 = conv_batchnorm_relu_block(conv2_2, nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(
        nb_filter[0], (2, 2), strides=(2, 2), name="up13", padding="same"
    )(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name="merge13", axis=bn_axis)
    conv1_3 = conv_batchnorm_relu_block(conv1_3, nb_filter=nb_filter[0])

    conv4_1 = conv_batchnorm_relu_block(pool3, nb_filter=nb_filter[3])
    pool4 = AvgPool2D((2, 2), strides=(2, 2), name="pool4")(conv4_1)

    up3_2 = Conv2DTranspose(
        nb_filter[2], (2, 2), strides=(2, 2), name="up32", padding="same"
    )(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name="merge32", axis=bn_axis)
    conv3_2 = conv_batchnorm_relu_block(conv3_2, nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(
        nb_filter[1], (2, 2), strides=(2, 2), name="up23", padding="same"
    )(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name="merge23", axis=bn_axis)
    conv2_3 = conv_batchnorm_relu_block(conv2_3, nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(
        nb_filter[0], (2, 2), strides=(2, 2), name="up14", padding="same"
    )(conv2_3)
    conv1_4 = concatenate(
        [up1_4, conv1_1, conv1_2, conv1_3], name="merge14", axis=bn_axis
    )
    conv1_4 = conv_batchnorm_relu_block(conv1_4, nb_filter=nb_filter[0])

    conv5_1 = conv_batchnorm_relu_block(pool4, nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(
        nb_filter[3], (2, 2), strides=(2, 2), name="up42", padding="same"
    )(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name="merge42", axis=bn_axis)
    conv4_2 = conv_batchnorm_relu_block(conv4_2, nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(
        nb_filter[2], (2, 2), strides=(2, 2), name="up33", padding="same"
    )(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name="merge33", axis=bn_axis)
    conv3_3 = conv_batchnorm_relu_block(conv3_3, nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(
        nb_filter[1], (2, 2), strides=(2, 2), name="up24", padding="same"
    )(conv3_3)
    conv2_4 = concatenate(
        [up2_4, conv2_1, conv2_2, conv2_3], name="merge24", axis=bn_axis
    )
    conv2_4 = conv_batchnorm_relu_block(conv2_4, nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(
        nb_filter[0], (2, 2), strides=(2, 2), name="up15", padding="same"
    )(conv2_4)
    conv1_5 = concatenate(
        [up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name="merge15", axis=bn_axis
    )
    conv1_5 = conv_batchnorm_relu_block(conv1_5, nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(
        1, (1, 1), activation="sigmoid", name="output_1", padding="same"
    )(conv1_2)
    nestnet_output_2 = Conv2D(
        1, (1, 1), activation="sigmoid", name="output_2", padding="same"
    )(conv1_3)
    nestnet_output_3 = Conv2D(
        1, (1, 1), activation="sigmoid", name="output_3", padding="same"
    )(conv1_4)
    nestnet_output_4 = Conv2D(
        1, (1, 1), activation="sigmoid", name="output_4", padding="same"
    )(conv1_5)

    if using_deep_supervision:
        model = Model(
            input=inputs,
            output=[
                nestnet_output_1,
                nestnet_output_2,
                nestnet_output_3,
                nestnet_output_4,
            ],
            name="Unet++",
        )
    else:
        model = Model(inputs=inputs, outputs=nestnet_output_4, name="Unet++")

    model.compile(optimizer=optmz, loss=[combined_loss], metrics=[mean_iou])
    model.summary()
    return model


# Build ResUnet model
from keras.layers import Add


def bn_act(x, act=True):
    x = BatchNormalization()(x)
    if act == True:
        x = Activation("relu")(x)
    return x


def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv


def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(
        conv, filters, kernel_size=kernel_size, padding=padding, strides=strides
    )

    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = Add()([conv, shortcut])
    return output


def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(
        x, filters, kernel_size=kernel_size, padding=padding, strides=strides
    )
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = Add()([shortcut, res])
    return output


def upsample_concat_block(x, xskip):
    u = UpSampling2D((2, 2))(x)
    c = Concatenate()([u, xskip])
    return c


def get_ResUNet(IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH):
    f = [16, 32, 64, 128, 256]
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # ======DownSample=======
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    # ======DownSample=======

    # ========Bridge=========
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    # ========Bridge=========

    # =======UpSample========
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])

    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])

    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])

    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    # =======UpSample========

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)

    model = Model(inputs=[inputs], outputs=[outputs], name="ResUnet")
    model.compile(optimizer=optmz, loss=[combined_loss], metrics=[mean_iou])
    model.summary()
    return model

def train(model, trainX, trainY, plotfile, callbacks_list):
    # Fit model
    results = model.fit(
        trainX,
        trainY,
        epochs=40,
        batch_size=256,
        verbose=1,
        shuffle=True,
        validation_split=0.1,
        callbacks=callbacks_list,
    )
    plot_model(
        model, to_file=plotfile, show_shapes=True, show_layer_names=False, rankdir="TB",
    )
    print("Path to plot:", plotfile)
# get_ResUNet(3, 128, 128)
