#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

from keras.optimizers import SGD
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint, EarlyStopping

from utils.models import *
from utils.preprocessing import *
import tensorflow as tf  
tf.test.gpu_device_name()

# Train datasets paths
train_origi_path = './DRIVE/training/images/'
train_truth_path = './DRIVE/training/1st_manual/'
train_mask_path = './DRIVE/training/mask/'
# Test datasets paths
test_origi_path = './DRIVE/test/images/'
test_truth_path = './DRIVE/test/1st_manual/'
test_mask_path = './DRIVE/test/mask/'
# All datasets paths
all_origi_path = './DRIVE/all/images/'
all_truth_path = './DRIVE/all/1st_manual/'
all_mask_path = './DRIVE/all/mask/'

# Pre-processing parameters 
N = 40
channels = 3
height = 584
width = 565

# Setup Index list for bootstrap sampling
Image_Index = []
for x in range(1 , 41):
    Image_Index.append(x)

NumOfUniqueTraining=[]
NumOfUniqueTesting=[]

# 0.92 hours for 1 bootstrap
for bootstraper in range(30,36):
    print('====Random Sampling Start====')
    ## =====Random sampling part=====
    # Create new directories for random datasets
    all_path = './DRIVE/all'
    dirs_list = [all_path,all_origi_path,all_truth_path,all_mask_path]
    for folder in dirs_list:
        if os.path.exists(folder):
            print ("Path exist")
        else:
            os.makedirs(folder)

    # Move the images to new directories
    if os.path.isfile(all_origi_path+'20_test.tif'):
        print ("Original test files exist")
    else:
        move_images(test_origi_path,test_truth_path,test_mask_path,all_origi_path,all_truth_path,all_mask_path,train=False)

    if os.path.isfile(all_origi_path+'40_training.tif'):
        print ("Original train files exist")
    else:
        move_images(train_origi_path,train_truth_path,train_mask_path,all_origi_path,all_truth_path,all_mask_path,train=True)

    # Conduct bootstrap sampling
    train_indexlist = []
    for i in range(40):
        x = random.sample(Image_Index,k=1)
        train_indexlist.append(x[0])
    test_indexlist = list(set(Image_Index) - set(train_indexlist))
    
    NumOfUniqueTraining.append(len(np.unique(train_indexlist)))
    NumOfUniqueTesting.append(len(np.unique(test_indexlist)))
    # Conduct bootstrap sampling

    # Create new directory for datasets
    path = './prepared_datasets/'
    if os.path.exists(path):
        print ("Path exist")
    else:
        os.makedirs(path)

    # Preparing the training datasets
    imgs_train, truth_train, masks_train = read_datasets(N,height,width,channels,train_indexlist, all_origi_path, all_truth_path, all_mask_path, True)
    write_hdf5(imgs_train, path + 'imgs_train_' +str(bootstraper) + '.hdf5')
    write_hdf5(truth_train, path + 'truth_train_' +str(bootstraper) + '.hdf5')
    write_hdf5(masks_train, path + 'masks_train_' +str(bootstraper) + '.hdf5')
    print('Train data done.')

    # Preparing the testing datasets
    imgs_test, truth_test, masks_test = read_datasets(N,height,width,channels,test_indexlist, all_origi_path, all_truth_path, all_mask_path, False)
    write_hdf5(imgs_test, path + 'imgs_test_' +str(bootstraper) + '.hdf5')
    write_hdf5(truth_test, path + 'truth_test_' +str(bootstraper) + '.hdf5')
    write_hdf5(masks_test, path + 'masks_test_' +str(bootstraper) + '.hdf5')
    print('Test data done.')
    ## =====Random sampling part=====
    print('====Random Sampling Done====')


    print('====Preprocessing Start====')
    ## =====Pre-processing part=====
    # Set the path and parameters of getting patches
    N_patches = 100000
    IMG_HEIGHT = 48 
    IMG_WIDTH = 48
    IMG_CHANNELS = 1
    path_data = "./prepared_datasets/"
    imgs_train = path_data + 'imgs_train_'+ str(bootstraper) + '.hdf5'
    truth_train = path_data + 'truth_train_' +str(bootstraper) + '.hdf5'

    # Get the patches of training images
    X_train, Y_train = prepare_training_data(
        imgs_train, truth_train, IMG_HEIGHT, IMG_WIDTH, N_patches
    )
    # print(np.shape(X_train))
    # print(np.shape(Y_train))

    # Check if training data looks all right
    ix = random.randint(0, np.shape(X_train)[0])
    fig = plt.figure(dpi=200)
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(X_train[ix])
    a.set_title("Original")
    # fig.savefig("./model_plot/OrignalMask.png")
    ## =====Pre-processing part=====
    print('====Preprocessing Done====')


    print('====Training Start====')
    ## =====Training part=====
    name = "Unet"
    plotpath = "./model_plot/" + name + '_' + str(bootstraper)
    plotfile = plotpath + ".png"

    # Setup callbacks
    earlystopper = EarlyStopping(monitor="val_loss", patience=12, verbose=1, mode="min")
    csv_logger = CSVLogger(plotpath + ".csv")
    # reduce_lr = ReduceLROnPlateau(
    #     monitor="val_loss", factor=0.1, patience=5, mode="auto", min_lr=0.00000001
    # )
    checkpointer = ModelCheckpoint(
    "h5/" + name + '_' + str(bootstraper) + ".h5", monitor="val_loss", verbose=1, save_best_only=True
    )
    # callbacks_list = [earlystopper, checkpointer, csv_logger, reduce_lr]
    callbacks_list = [earlystopper, checkpointer, csv_logger]
    # Get model
    model = get_UNet(
        IMG_CHANNELS,
        IMG_HEIGHT,
        IMG_WIDTH,
        Base=32,
        depth=4,
        inc_rate=2,
        activation="relu",
        drop=0.2,
        batchnorm=True,
        N=2, 
    )
    # Start train
    train(model, X_train, Y_train, plotfile, callbacks_list)
    ## =====Training part=====
    print('====Training Done====')
# %%
import pandas as pd

data_frame = {'Num of unique image in training dataset': NumOfUniqueTraining, 'Num of unique image in testing dataset': NumOfUniqueTesting} 
data_frame = pd.DataFrame(data=data_frame)
data_frame.to_csv('model_plot/NOUniqueImages.csv')
