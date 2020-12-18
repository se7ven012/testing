#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy():
    my_dpi = 200

    names = ["Unet", "SegNet", "SegUnet", "UnetPP", "ResUNet"]

    for name in names:
        tra_loss=pd.read_csv('./model_plot/'+name+'.csv',usecols=['loss'])
        val_loss=pd.read_csv('./model_plot/'+name+'.csv',usecols=['val_loss'])
        # loval_mean_iou=pd.read_csv('./model_plot/'+name+'.csv',usecols=['val_mean_iou'])

        fig = plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        ax = fig.add_subplot(111)
        ax.plot(val_loss,label='val_loss')
        ax.plot()
        ax.plot(tra_loss,label='train_loss')
        plt.title(name) 
        plt.legend()
        plt.savefig('./model_plot/'+name+'_'+'loss.png',figsize=(800/my_dpi, 400/my_dpi), dpi=my_dpi)
        plt.show()

    print('Done!')

plot_accuracy()