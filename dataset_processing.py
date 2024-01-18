import os
import numpy as np
import cv2 as cv
import tensorflow as tf
from sklearn.model_selection import train_test_split

from preprocessing import ImageTransformer


def preprocess_tf_input(im, t):

    im_tf = tf.cast(im,tf.float32)
    im_tf = im_tf/255.

    t_tf = tf.cast(t,tf.float32)

    return im_tf, t_tf

def preprocess_input(im, t):

    im = im.astype(np.float32)
    im = im/255.

    return im, t

def preprocess_tf_dataset(im_tilde, im):

    im_tilde_tf = tf.cast(im_tilde,tf.float32)
    im_tilde_tf = im_tilde_tf/255.

    im_tf = tf.cast(im,tf.float32)
    im_tf = im_tf/255.

    return im_tilde_tf, im_tf

def preprocess_dataset(im_tilde, im):

    im_tilde = im_tilde.astype(np.float32)
    im_tilde = im_tilde/255.

    im = im.astype(np.float32)
    im = im/255.

    return im_tilde, im

def preprocess_tf_data(im):

    im_tf = tf.cast(im,tf.float32)
    im_tf = im_tf/255

    return im_tf

def preprocess_data(im):

    im = im.astype(np.float32)
    im = im/255.

    return im

def read_dataset(case_folder, dataset_folder='Training', format='png'):

    img_filepaths = []
    for (root, case_dirs, _) in os.walk(os.path.join(case_folder,'Datasets',dataset_folder)):
        for case_dir in case_dirs:
            files = [os.path.join(root,case_dir,file) for file in os.listdir(os.path.join(root,case_dir)) if file.endswith(format)]
            img_filepaths += files

    img_list = []
    for filepath in img_filepaths:
        img = cv.imread(filepath)
        img_list.append(img)

    return img_list

def preprocess_image(imgs, new_dims):

    m = len(imgs)
    imgs_processed = np.zeros((m,new_dims[1],new_dims[0],1),dtype='uint8')
    for i in range(m):
        if imgs[i].shape[0:2] != (new_dims[1],new_dims[0]):
            img_processed = ImageTransformer.resize(imgs[i],new_dims)
        else:
            img_processed = imgs[i]
        imgs_processed[i,:,:,0] = cv.cvtColor(img_processed,cv.COLOR_BGR2GRAY)

    return imgs_processed

def get_datasets(case_folder, training_size, img_dims, pierce_size):

    # Read original datasets
    X = read_dataset(case_folder)
    # Resize images, if necessary
    X = preprocess_image(X,img_dims)

    X_train, X_val = train_test_split(X,train_size=training_size,shuffle=True)
    X_cv, X_test = train_test_split(X_val,train_size=0.75,shuffle=True)

    # Generate training pierced dataset: pierced sample + negative pierced sample
    X_Train = [ImageTransformer.pierce(sample,pierce_size,return_neg=True) for sample in X_train]
    X_train_np = np.array([X[0] for X in X_Train])
    X_train_p = np.array([X[1] for X in X_Train])
    X_train = np.array([X for X in X_train])
    # Generate cross-validation pierced dataset: pierced sample + negative pierced sample
    X_Cv = [ImageTransformer.pierce(sample,pierce_size,return_neg=True) for sample in X_cv]
    X_cv_np = np.array([X[0] for X in X_Cv])
    X_cv_p = np.array([X[1] for X in X_Cv])
    X_cv = np.array([X for X in X_cv])
    # Generate test pierced dataset: pierced sample + negative pierced sample
    X_Test = [ImageTransformer.pierce(sample,pierce_size,return_neg=True) for sample in X_test]
    X_test_np = np.array([X[0] for X in X_Test])
    X_test_p = np.array([X[1] for X in X_Test])
    X_test = np.array([X for X in X_test])

    return X_train, X_train_p, X_train_np, X_cv, X_cv_p, X_cv_np, X_test, X_test_p, X_test_np

def create_dataset_pipeline(dataset, is_train=True, num_threads=8, prefetch_buffer=100, batch_size=32):
    '''
    Function to convert a numpy dataset (X,X) into a tensorflow dataset (X_tf,X_tf)
    '''

    dataset_tensor = tf.data.Dataset.from_tensor_slices(dataset)

    if is_train:
        dataset_tensor = dataset_tensor.shuffle(buffer_size=dataset[0].shape[0]).repeat()
    dataset_tensor = dataset_tensor.map(preprocess_tf_dataset,num_parallel_calls=num_threads)
    dataset_tensor = dataset_tensor.batch(batch_size)
    dataset_tensor = dataset_tensor.prefetch(prefetch_buffer)

    return dataset_tensor
    
def create_data_pipeline(data, is_train=True, num_threads=8, prefetch_buffer=100, batch_size=32):
    '''
    Function to convert a numpy array X into a tensorflow object X_tf
    '''

    data_tensor = tf.data.Dataset.from_tensor_slices(data)

    if is_train:
        data_tensor = data_tensor.shuffle(buffer_size=data.shape[0]).repeat()
    data_tensor = data_tensor.map(preprocess_tf_data,num_parallel_calls=num_threads)
    data_tensor = data_tensor.batch(batch_size)
    data_tensor = data_tensor.prefetch(prefetch_buffer)

    return data_tensor

def create_input_pipeline(input, is_train=True, num_threads=8, prefetch_buffer=100, batch_size=32):
    '''
    Function to convert a numpy input dataset (t,X) into a tensorflow dataset (t_tf,X_tf)
    '''

    input_tensor = tf.data.Dataset.from_tensor_slices(input)

    if is_train:
        input_tensor = input_tensor.shuffle(buffer_size=input[0].shape[0]).repeat()
    input_tensor = input_tensor.map(preprocess_tf_input,num_parallel_calls=num_threads)
    input_tensor = input_tensor.batch(batch_size)
    input_tensor = input_tensor.prefetch(prefetch_buffer)

    return input_tensor

def get_tensorflow_datasets(dataset_train, dataset_cv, dataset_test, batch_size=32):

    dataset_tf_train = create_dataset_pipeline(dataset_train,is_train=True,batch_size=batch_size)
    dataset_tf_cv = create_dataset_pipeline(dataset_cv,is_train=False,batch_size=1)
    dataset_test = preprocess_dataset(dataset_test[0],dataset_test[1])
    
    return dataset_tf_train, dataset_tf_cv, dataset_test

def get_tensorflow_inputs(input_train, input_cv, input_test, batch_size=32):

    input_tf_train = create_input_pipeline(input_train,is_train=True,batch_size=batch_size)
    input_tf_cv = create_input_pipeline(input_cv,is_train=False,batch_size=1)
    input_test = preprocess_input(input_test[0],input_test[1])

    return input_tf_train, input_tf_cv, input_test
    
def get_tensorflow_data(data_train, data_cv, data_test, batch_size=32):

    data_tf_train = create_data_pipeline(data_train,is_train=True,batch_size=batch_size)
    data_tf_cv = create_data_pipeline(data_cv,is_train=False,batch_size=1)
    data_test = preprocess_dataset(data_test)
    
    return data_tf_train, data_tf_cv, data_test
    