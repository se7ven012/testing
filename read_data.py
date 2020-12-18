#%%
import os
import h5py
import shutil
import numpy as np
from PIL import Image
    
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

def write_hdf5(arr,outfile):
    with h5py.File(outfile,"w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)
        
def read_datasets(N,height,width,channels,randomlist,imgs_dir,truth_dir,mask_dir,train = True):
    imgs = np.empty((N,height,width,channels))
    truth = np.empty((N,height,width))
    masks = np.empty((N,height,width))
    counter = 0
    for root, dirs, files in os.walk(imgs_dir):
        for i in randomlist:
            # Original images
            img = Image.open(imgs_dir + files[i-1])
            imgs[counter] = np.asarray(img)
            # Ground truth
            true = Image.open(truth_dir + files[i-1][0:2] + '_manual1.gif')
            truth[counter] = np.asarray(true)
            # Masks
            if int(i) in range(1,21):
                mask = Image.open(mask_dir + files[i-1][0:2] + '_test_mask.gif')
            else:
                mask = Image.open(mask_dir + files[i-1][0:2] + '_training_mask.gif')
            masks[counter] = np.asarray(mask)
            counter+=1
    # Reshaping the images 
    imgs = np.transpose(imgs,(0,3,1,2))
    truth = np.reshape(truth,(N,1,height,width))
    masks = np.reshape(masks,(N,1,height,width))
    
    return imgs, truth, masks

def move_images(imgs_dir,truth_dir,mask_dir,tar_imgs_dir,tar_truth_dir,tar_mask_dir,train):
    for root, dirs, files in os.walk(imgs_dir):
        for i in range(20):
            img_path = imgs_dir + files[i]
            shutil.move(img_path, tar_imgs_dir)
            # Ground truth
            true_path = truth_dir + files[i][0:2] + '_manual1.gif'
            shutil.move(true_path, tar_truth_dir)
            # Masks
            if train:
                mask_path = mask_dir + files[i][0:2] + '_training_mask.gif'
                shutil.move(mask_path, tar_mask_dir)
            else:
                mask_path = mask_dir + files[i][0:2] + '_test_mask.gif'
                shutil.move(mask_path, tar_mask_dir)
#%%
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
#%%
# Random selection on dataset
import random
L = []
for x in range(1 , 41):
    L.append(x)
train_indexlist = random.sample(L, int((2/3)*len(L)))
test_indexlist = list(set(L) - set(train_indexlist))
print(train_indexlist)
print(test_indexlist)

#%%
# Create new directory for datasets
path = './prepared_datasets/'
if os.path.exists(path):
    print ("Path exist")
else:
    os.makedirs(path)

# Preparing the training datasets
imgs_train, truth_train, masks_train = read_datasets(train_indexlist, all_origi_path, all_truth_path, all_mask_path, True)
write_hdf5(imgs_train, path + 'imgs_train.hdf5')
write_hdf5(truth_train, path + 'truth_train.hdf5')
write_hdf5(masks_train, path + 'masks_train.hdf5')
print('Train data done.')

# Preparing the testing datasets
imgs_test, truth_test, masks_test = read_datasets(test_indexlist, all_origi_path, all_truth_path, all_mask_path, False)
write_hdf5(imgs_test, path + 'imgs_test.hdf5')
write_hdf5(truth_test, path + 'truth_test.hdf5')
write_hdf5(masks_test, path + 'masks_test.hdf5')
print('Test data done.')

#%%
# import random

# Image_Index = []
# for x in range(1 , 41):
#     Image_Index.append(x)
# train_indexlist = []
# for i in range(40):
#     x = random.sample(Image_Index,k=1)
#     train_indexlist.append(x[0])
# all_origi_path = './DRIVE/all/images/'
# for root, dirs, files in os.walk(all_origi_path):
#     for i in train_indexlist:
#         if i in range(1,21):
#             print(i,'_test_mask.gif')
#         else:
#             print(i,'_training_mask.gif')
# %%
