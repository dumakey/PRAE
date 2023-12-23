import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from random import randint
from preprocessing import ImageTransformer
import dataset_processing
import tensorflow as tf


def plot_generated_samples(datasets, img_size, storage_dir):

    width, height = img_size
    n_datasets = len(datasets)
    n_rows = 4
    n_cols = 2
    for k,dataset in enumerate(datasets):
        n_samples = dataset.shape[0]
        n_figs = int(np.ceil(n_samples/(n_rows*n_cols)))
        s = 0
        for n in range(n_figs):
            fig, ax = plt.subplots(n_rows,n_cols,sharex=True,figsize=(10,10))
            if n_rows == 1:
                ax = np.reshape(ax,(1,2))
            # hide axis
            for ii in range(n_rows):
                for jj in range(n_cols):
                    ax[ii,jj].axis('off')
            for i in range(n_rows):
                for j in range(n_cols):
                    x = np.reshape(dataset[s],(height,width))*255  # Un-scale image
                    x = x.astype('uint8')
                    _, x = cv.threshold(x,50,255,cv.THRESH_BINARY)
                    # x_dec = cv.bitwise_not(x_dec)
                    ax[i,j].imshow(x,cmap='Greys_r')
                    ax[i,j].set_xticks([])
                    ax[i,j].set_yticks([])
                    s += 1
                    if s == n_samples:
                        break
                if s == n_samples:
                    break

            if n_figs > 1:
                if n_datasets == 1:
                    plt.savefig(os.path.join(storage_dir,'Generated_samples_{}.png'.format(n+1)), dpi=100)
                else:
                    plt.savefig(os.path.join(storage_dir,'Dataset_{}_generated_samples_{}.png'.format(k+1,n+1)), dpi=100)
            else:
                if n_datasets == 1:
                    plt.savefig(os.path.join(storage_dir,'Generated_samples.png'), dpi=100)
                else:
                    plt.savefig(os.path.join(storage_dir,'Dataset_{}_generated_samples.png'.format(k+1)), dpi=100)
            plt.close()

def plot_dataset_samples(dataset, predictor, n_samples, img_size, storage_dir, stage='Train'):

    dataset_prep, _ = dataset_processing.preprocess_data(dataset[0],dataset[1])
    width, height = img_size
    m = dataset[0].shape[0]
    
    ## PLOT GENERATED TRAINING DATA ##
    n_rows = 5
    n_figs = int(np.ceil(n_samples/n_rows))
    s = 0
    for j in range(n_figs):
        fig, ax = plt.subplots(n_rows,2,sharex=True,figsize=(10,10))
        if n_rows == 1:
            ax = np.reshape(ax,(1,2))
        # hide axis
        for ii in range(n_rows):
            for jj in range(2):
                ax[ii,jj].axis('off')
                ax[ii,jj].set_xticks([])
                ax[ii,jj].set_yticks([])
        ax[0,0].title.set_text('Predicted\n')
        ax[0,1].title.set_text(stage+'\n')
        for ii in range(n_rows):
            i = randint(0,m-1)
            # Predict
            x_unrolled = dataset_prep[i,:].reshape((1,height*width))
            x_pred = predictor(x_unrolled)
            x_pred = np.reshape(x_pred,(height,width))*255  # Un-scale image
            x_pred = x_pred.astype('uint8')
            _, x_pred = cv.threshold(x_pred,50,255,cv.THRESH_BINARY)
            # x_pred = cv.bitwise_not(x_pred)

            # Plot
            x = dataset[0][i].reshape((height,width))
            # x = cv.bitwise_not(x)
            ax[ii,0].imshow(x_pred,cmap='Greys_r')
            ax[ii,1].imshow(x,cmap='Greys_r')
            s += 1
            if s == n_samples:
                break

        if n_figs > 1:
            plt.savefig(os.path.join(storage_dir,stage+'_'+'training_samples_{}.png'.format(j+1)), dpi=100)
        else:
            plt.savefig(os.path.join(storage_dir,stage+'_'+'training_samples.png'), dpi=100)
        plt.close()


def monitor_hidden_layers(img, model, case_dir, figs_per_row=5, rows_to_cols_ratio=1, idx=None):

    if idx:
        storage_dir = os.path.join(case_dir,'Results','pretrained_model','Hidden_activations','Sample_' + str(idx))
    else:
        storage_dir = os.path.join(case_dir, 'Results', 'pretrained_model', 'Hidden_activations')

    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)

    # Input preprocessing
    img = img.reshape((1,img.size))
    img = 1/255.*img
    img_tensor = tf.convert_to_tensor(img)

    # Activation model setup
    idx_0 = 2
    idx_f = [i for i,layer in enumerate(model.layers) if 'flatten' in layer.name][0]
    layer_outputs = [layer.output for layer in model.layers[idx_0:idx_f]]
    layer_names = [layer.name for layer in model.layers[idx_0:idx_f]]
    activation_model = tf.keras.Model(inputs=model.input,outputs=layer_outputs)
    activations = activation_model.predict(img_tensor,steps=1)

    # Plotting
    figs_per_row = 5
    layer_idx = 1
    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        channel_idx = 0
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        height = layer_activation.shape[1]
        width = layer_activation.shape[2]
        n_rows = int(np.ceil(n_features/figs_per_row))
        display_grid = np.zeros((height*n_rows,figs_per_row*width))
        for row in range(n_rows):
            for col in range(figs_per_row):
                channel_image = layer_activation[0,:,:,channel_idx]
                channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image,0,255).astype('uint8')
                display_grid[row*height:(row + 1)*height,col*width:(col + 1)*width] = channel_image
                channel_idx += 1

                if (channel_idx + 1) > n_features:
                    break

        if n_rows < int(rows_to_cols_ratio*figs_per_row):
            plt.figure(figsize=(25,10))
            plt.suptitle('Layer: {}'.format(layer_name),fontsize=18)
            plt.axis('off')
            plt.imshow(display_grid,aspect='auto',cmap='viridis')  # cmap: plasma / viridis
            plt.savefig(os.path.join(storage_dir,'Layer_{:d}_{}_activations.png'.format(layer_idx,layer_name)),dpi=150)
            plt.close()
        else:
            n_rows_old = n_rows
            n_rows = rows_to_cols_ratio*figs_per_row
            for j in range(int(np.ceil(n_rows_old/n_rows))):
                plt.figure(figsize=(25,10))
                plt.suptitle('Layer: {}'.format(layer_name),fontsize=18)
                plt.axis('off')
                plt.imshow(display_grid[j*height*n_rows:(j+1)*height*n_rows,:],aspect='auto',cmap='viridis')   # cmap: plasma / viridis
                plt.savefig(os.path.join(storage_dir,'Case_{:d}_layer_{:d}_{}_activations_{}.png'.
                                         format(case_ID,layer_idx,layer_name,(j+1))), dpi=150)
                plt.close()

        layer_idx += 1


