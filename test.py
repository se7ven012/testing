#%%
import numpy as np

from keras.models import model_from_json
from keras.models import Model

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

from utils.dice import *
from utils.models import *
from utils.preprocessing import *
np.set_printoptions(suppress=True)

# Make sure the program will be running by GPU
import tensorflow as tf
tf.test.gpu_device_name()

# Set the path and parameters of getting patches 
patch_h = 48
patch_w = 48
stride_h = 5
stride_w = 5

DICE_List=[]
ACC_List=[]
SPE_List=[]
SEN_List=[]
AUC_List=[]

for bootstraper in range(50):
    path_data = './prepared_datasets/'
    imgs_test = path_data + 'imgs_test_' +str(bootstraper) + '.hdf5'
    masks_test = path_data + 'masks_test_' +str(bootstraper) + '.hdf5'
    truth_test = path_data + 'truth_test_' +str(bootstraper) + '.hdf5'

    # Original images
    test_imgs = load_hdf5(imgs_test)

    # Masks
    test_masks = load_hdf5(masks_test)

    # Number of original images
    N_imgs = test_imgs.shape[0]

    # Height and width of original image
    img_h = test_imgs.shape[2]
    img_w = test_imgs.shape[3]

    # Get the patches of testing images
    # patches_imgs_test, patches_imgs_mask, new_h, new_w, test_truth = prepare_testing_data(imgs_test, truth_test, patch_h, patch_w, stride_h, stride_w)
    patches_imgs_test, new_h, new_w, test_truth = prepare_testing_data(imgs_test, truth_test, patch_h, patch_w, stride_h, stride_w)

    # Load model
    model = load_model("h5/Unet_" + str(bootstraper) + ".h5", custom_objects={'mean_iou': mean_iou})

    # Make prediction
    predictions = model.predict(patches_imgs_test, batch_size=64, verbose=1)
    pred = np.empty((predictions.shape[0],1,patch_h,patch_w))
    # print('Prediction:'+ str(np.shape(predictions)))
    # print('Pred:'+ str(np.shape(pred)))

    for i in range(predictions.shape[0]):
        for j in range(patch_h):
            for k in range(patch_w):
                # Take the probability of being vessel pixel as the predicted result for each pixel 
                pred[i,0,j,k] = predictions[i,j,k,1]

    # Restore the patches to the full-size images
    pred_imgs = restore_patches(pred, N_imgs, new_h, new_w, stride_h, stride_w)
    pred_imgs = pred_imgs[:, :, 0:img_h, 0:img_w]

    # Remove the results outside the FOV and only evaluate by results inside the FOV 
    y_pred, y_true = clean_outside(pred_imgs, test_truth, test_masks)


    #%%
    # Reshape the images for visualization
    trueMask = np.reshape(test_truth ,(N_imgs, img_h, img_w))
    predMask = np.reshape(pred_imgs ,(N_imgs, img_h, img_w))

    trueMask = (trueMask[0] * 255).astype(np.uint8)
    trueMask = Image.fromarray(trueMask)
    predMask = (predMask[0] * 255).astype(np.uint8)
    predMask = Image.fromarray(predMask)

    fig = plt.figure(dpi=200)
    a = fig.add_subplot(1, 3, 2)
    plt.imshow(trueMask)
    a.set_title("Ground Truth Mask")
    a = fig.add_subplot(1, 3, 3)
    plt.imshow(predMask)
    a.set_title("Prediction Mask")
    # %%

    # AUC --- Area under the ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    print("\nArea under the ROC curve: " +str(auc))
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig("ROC_curve.png")

    #%%
    # Confusion matrix and related metrics
    y_true = (y_true).astype(np.int)
    for i in range(y_pred.shape[0]):
        if y_pred[i] < 0.5:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    cm = confusion_matrix(y_true, y_pred)
    accuracy = (cm[0, 0] + cm[1, 1]) / sum(sum(cm))
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) # sensitivity
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = 2 * cm[1,1] / (2 * cm[1,1] + cm[1, 0] + cm[0, 1])
    dice = dice_coef(y_true,y_pred)

    DICE_List.append(dice)
    ACC_List.append(accuracy)
    SPE_List.append(specificity)
    SEN_List.append(sensitivity)
    AUC_List.append(auc)

#%%
import pandas as pd

data_frame = {'Dice': DICE_List, 'Accuracy': ACC_List, 'Sensitivity': SEN_List,'Specificity': SPE_List, 'AUC':ACC_List} 
data_frame = pd.DataFrame(data=data_frame)
data_frame.to_csv('results/Unet.csv')