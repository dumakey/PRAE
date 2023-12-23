# -*- coding: utf-8 -*-
import os
from shutil import rmtree, copytree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import pickle
import cv2 as cv
from random import randint

import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

import reader
import dataset_processing
import models
import dataset_augmentation
import postprocessing


class CGenTrainer:

    def __init__(self, launch_file):

        class parameter_container:
            pass
        class dataset_container:
            pass
        class model_container:
            pass
        class predictions_container:
            pass

        self.parameters = parameter_container()
        self.datasets = dataset_container()
        self.model = model_container()
        self.predictions = predictions_container()

        # Setup general parameters
        casedata = reader.read_case_setup(launch_file)
        self.parameters.analysis = casedata.analysis
        self.parameters.training_parameters = casedata.training_parameters
        self.parameters.img_processing = casedata.img_processing
        self.parameters.img_size = casedata.img_resize
        self.parameters.samples_generation = casedata.samples_generation
        self.parameters.data_augmentation = casedata.data_augmentation
        self.parameters.activation_plotting = casedata.activation_plotting
        self.case_dir = casedata.case_dir

        # Sensitivity analysis variable identification
        sens_vars = [item for item in self.parameters.training_parameters.items() if
                     item[0] not in ('enc_hidden_layers', 'dec_hidden_layers') if type(item[1]) == list]
        self.parameters.sens_variable = sens_vars[0] if len(sens_vars) != 0 else None

        # Check for model reconstruction
        if self.parameters.analysis['import'] == True:
            self.model.imported = True
            model, history = self.reconstruct_model()
            self.model.Model = [model]
            self.model.History = [history]
        else:
            self.model.imported = False

    def __str__(self):
        class_name = type(self).__name__

        return '{}, a class to generate contours based on Bayesian Deep learning algorithms'.format(class_name)

    def launch_analysis(self):

        analysis_ID = self.parameters.analysis['type']
        analysis_list = {
                        'singletraining': self.singletraining,
                        'sensanalysis': self.sensitivity_analysis_on_training,
                        'traingenerate': self.traingenerate,
                        'generate': self.contour_generation,
                        'datagen': self.data_generation,
                        'plotactivations': self.plot_activations,
                        }

        analysis_list[analysis_ID]()

    def sensitivity_analysis_on_training(self):

        # Retrieve sensitivity variable
        sens_variable = self.parameters.sens_variable

        case_dir = self.case_dir
        training_size = self.parameters.training_parameters['train_size']
        batch_size = self.parameters.training_parameters['batch_size']
        img_size = self.parameters.img_size
        pierce_size = self.parameters.img_processing['piercesize']

        self.datasets.data_train, self.datasets.data_cv, self.datasets.data_test = \
        dataset_processing.get_datasets(case_dir,training_size,img_size,pierce_size)
        self.datasets.dataset_train, self.datasets.dataset_cv, self.datasets.dataset_test = \
        dataset_processing.get_tensorflow_datasets(self.datasets.data_train,self.datasets.data_cv,self.datasets.data_test,batch_size)
        if self.model.imported == False:
            self.train_model(sens_variable)
        self.export_model_performance(sens_variable)
        self.export_model(sens_variable)
        self.export_log()

    def singletraining(self):

        case_dir = self.case_dir
        training_size = self.parameters.training_parameters['train_size']
        batch_size = self.parameters.training_parameters['batch_size']
        img_size = self.parameters.img_size
        pierce_size = self.parameters.img_processing['piercesize']

        self.datasets.X_train, self.datasets.X_train_p, self.datasets.X_train_pneg,\
        self.datasets.X_cv, self.datasets.X_cv_p, self.datasets.X_cv_pneg,\
        self.datasets.X_test, self.datasets.X_test_p, self.datasets.X_test_pneg= \
        dataset_processing.get_datasets(case_dir,training_size,img_size,pierce_size)

        # Training over the complete picture
        data_train_c = (self.datasets.X_train,self.datasets.X_train)
        data_cv_c = (self.datasets.X_cv,self.datasets.X_cv)
        data_test_c = (self.datasets.X_test,self.datasets.X_test)

        self.datasets.dataset_train_c, self.datasets.dataset_cv_c, self.datasets.dataset_test_c = \
        dataset_processing.get_tensorflow_datasets(data_train_c,data_cv_c,data_test_c,batch_size)

        '''
        # Training over the negative-pierce picture
        data_train_np = (self.datasets.X_train_np, self.datasets.X_train_np)
        data_cv_np = (self.datasets.X_cv_np, self.datasets.X_cv_np)
        data_test_np = (self.datasets.X_test_np, self.datasets.X_test_np)

        self.datasets.dataset_train_np, self.datasets.dataset_cv_np, self.datasets.dataset_test_np = \
        dataset_processing.get_tensorflow_datasets(data_train_np,data_cv_np,data_test_np,batch_size)
        '''
        if self.model.imported == False:
            self.train_model()
        self.export_model_performance()
        self.export_model()
        self.export_log()
    
    def traingenerate(self):


        # Training
        case_dir = self.case_dir
        training_size = self.parameters.training_parameters['train_size']
        batch_size = self.parameters.training_parameters['batch_size']
        img_size = self.parameters.img_size
        pierce_size = self.parameters.img_processing['piercesize']

        self.datasets.data_train, self.datasets.data_cv, self.datasets.data_test = \
        dataset_processing.get_datasets(case_dir,training_size,img_size,pierce_size)
        self.datasets.dataset_train, self.datasets.dataset_cv, self.datasets.dataset_test = \
        dataset_processing.get_tensorflow_datasets(self.datasets.data_train,self.datasets.data_cv,self.datasets.data_test,batch_size)
        if self.model.imported == False:
            self.train_model()
        self.export_model_performance()
        self.export_model()
        self.export_log()
        
        # Generation
        model_dir = os.path.join(case_dir,'Results',str(self.parameters.analysis['case_ID']),'Model')
        generation_dir = os.path.join(case_dir,'Results','pretrained_model')
        if os.path.exists(generation_dir):
            rmtree(generation_dir)
        copytree(model_dir,generation_dir)
        self.model.imported = True
        self.contour_generation()
        

    def data_generation(self):

        transformations = [{k:v[1:] for (k,v) in self.parameters.img_processing.items() if v[0] == 1}][0]
        augdata_size = self.parameters.data_augmentation[1]
        self.generate_augmented_data(transformations,augdata_size)

    def plot_activations(self):

        # Parameters
        case_dir = self.case_dir
        img_dims = self.parameters.img_size
        latent_dim = self.parameters.training_parameters['latent_dim']
        pierce_size = self.parameters.img_processing['piercesize']
        batch_size = self.parameters.training_parameters['batch_size']
        training_size = self.parameters.training_parameters['train_size']
        n = self.parameters.activation_plotting['n_samples']
        case_ID = self.parameters.analysis['case_ID']
        figs_per_row = self.parameters.activation_plotting['n_cols']
        rows_to_cols_ratio = self.parameters.activation_plotting['rows2cols_ratio']

        # Generate datasets
        self.datasets.data_train, self.datasets.data_cv, self.datasets.data_test = \
        dataset_processing.get_datasets(case_dir,training_size,img_dims,pierce_size)
        self.datasets.dataset_train, self.datasets.dataset_cv, self.datasets.dataset_test = \
        dataset_processing.get_tensorflow_datasets(self.datasets.data_train,self.datasets.data_cv,self.datasets.data_test,batch_size)

        m_tr = self.datasets.data_train[0].shape[0]
        m_cv = self.datasets.data_cv[0].shape[0]
        m_ts = self.datasets.data_test[0].shape[0]
        m = m_tr + m_cv + m_ts

        # Read datasets
        dataset = np.zeros((m,np.prod(img_dims)),dtype='uint8')
        dataset[:m_tr,:] = self.datasets.data_train[0]
        dataset[m_tr:m_tr+m_cv,:] = self.datasets.data_cv[0]
        dataset[m_tr+m_cv:m,:] = self.datasets.data_test[0]

        # Index image sampling
        idx = [randint(1,m) for i in range(n)]
        idx_set = set(idx)
        while len(idx) != len(idx_set):
            extra_item = randint(1,m)
            idx_set.add(extra_item)

        # Reconstruct encoder model
        encoder = self.reconstruct_encoder_CNN()

        # Plot
        for idx in idx_set:
            img = dataset[idx,:]
            postprocessing.monitor_hidden_layers(img,encoder,case_dir,figs_per_row,rows_to_cols_ratio,idx)

    def generate_augmented_data(self, transformations, augmented_dataset_size=1):

        # Set storage folder for augmented dataset
        augmented_dataset_dir = os.path.join(self.case_dir,'Datasets','Augmented')

        # Unpack data
        X = dataset_processing.read_dataset(case_folder=self.case_dir,dataset_folder='To_augment')
        # Generate new dataset
        data_augmenter = dataset_augmentation.datasetAugmentationClass(X,transformations,augmented_dataset_size,augmented_dataset_dir)
        data_augmenter.transform_images()
        data_augmenter.export_augmented_dataset()

    def contour_generation(self):

        if self.model.imported == True:
            storage_dir = os.path.join(self.case_dir,'Results','pretrained_model','Image_generation')
        else:
            storage_dir = os.path.join(self.case_dir,'Results','Image_generation')
        if os.path.exists(storage_dir):
            rmtree(storage_dir)
        os.makedirs(storage_dir)

        # Read parameters
        case_dir = self.case_dir
        casedata = reader.read_case_logfile(os.path.join(case_dir,'Results','pretrained_model','CGVAE.log'))
        n_samples = self.parameters.samples_generation['n_samples']
        training_size = casedata.training_parameters['train_size']
        pierce_size = self.parameters.img_processing['piercesize']
        img_size = casedata.img_size

        if self.model.imported == False:
            self.singletraining()

        if not hasattr(self, 'data_train'):
            data_train, data_cv, data_test = dataset_processing.get_datasets(case_dir,training_size,img_size,pierce_size)
            for model in self.model.Model:
                postprocessing.plot_dataset_samples(data_train,model.predict,n_samples,img_size,storage_dir,stage='Train')
                postprocessing.plot_dataset_samples(data_cv,model.predict,n_samples,img_size,storage_dir,stage='Cross-validation')
                postprocessing.plot_dataset_samples(data_test,model.predict,n_samples,img_size,storage_dir,stage='Test')

        ## GENERATE NEW DATA - SAMPLING ##
        X_samples = self.generate_samples(casedata)
        postprocessing.plot_generated_samples(X_samples,img_size,storage_dir)

    def train_model(self, sens_var=None):

        # Parameters
        input_dim = (*self.parameters.img_size,1)
        latent_dim = self.parameters.training_parameters['latent_dim']
        enc_hidden_layers = self.parameters.training_parameters['enc_hidden_layers']
        dec_hidden_layers = self.parameters.training_parameters['dec_hidden_layers']
        alpha = self.parameters.training_parameters['learning_rate']
        nepoch = self.parameters.training_parameters['epochs']
        batch_size = self.parameters.training_parameters['batch_size']
        l2_reg = self.parameters.training_parameters['l2_reg']
        l1_reg = self.parameters.training_parameters['l1_reg']
        dropout = self.parameters.training_parameters['dropout']
        activation = self.parameters.training_parameters['activation']

        self.model.Model = []
        self.model.History = []
        Model = models.VAE
        if sens_var == None:  # If it is a one-time training
            self.model.Model.append(Model(input_dim,latent_dim,enc_hidden_layers,dec_hidden_layers,alpha,l2_reg,
                                               l1_reg,dropout,activation,mode='train'))
            self.model.History.append(self.model.Model[-1].fit(self.datasets.dataset_train_c,epochs=nepoch,steps_per_epoch=200,
                                                               validation_data=self.datasets.dataset_cv_c,validation_steps=None,
                                                               verbose=1))
        else: # If it is a sensitivity analysis
            if type(alpha) == list:
                for learning_rate in alpha:
                    if self.model.imported == False:
                        model = Model(input_dim,latent_dim,enc_hidden_layers,dec_hidden_layers,learning_rate,
                                           l2_reg,l1_reg,dropout,activation,mode='train')
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,steps_per_epoch=200,
                                                        validation_data=self.datasets.dataset_cv,validation_steps=None,
                                                        verbose=1))
            elif type(l2_reg) == list:
                for regularizer in l2_reg:
                    if self.model.imported == False:
                        model = Model(input_dim,latent_dim,enc_hidden_layers,dec_hidden_layers,alpha,regularizer,
                                           l1_reg,dropout,activation,mode='train')
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,steps_per_epoch=200,
                                                        validation_data=self.datasets.dataset_cv,validation_steps=None,
                                                        verbose=1))
            elif type(l1_reg) == list:
                for regularizer in l1_reg:
                    if self.model.imported == False:
                        model = Model(input_dim,latent_dim,enc_hidden_layers,dec_hidden_layers,alpha,l2_reg,
                                           regularizer,dropout,activation,mode='train')
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,steps_per_epoch=200,
                                                        validation_data=self.datasets.dataset_cv,validation_steps=None,
                                                        verbose=1))
            elif type(dropout) == list:
                for rate in dropout:
                    if self.model.imported == False:
                        model = Model(input_dim,latent_dim,enc_hidden_layers,dec_hidden_layers,alpha,l2_reg,
                                           l1_reg,rate,activation,mode='train')
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,steps_per_epoch=200,
                                                        validation_data=self.datasets.dataset_cv,validation_steps=None,
                                                        verbose=1))
            elif type(activation) == list:
                for act in activation:
                    if self.model.imported == False:
                        model = Model(input_dim,latent_dim,enc_hidden_layers,dec_hidden_layers,alpha,l2_reg,
                                           l1_reg,dropout,act,mode='train')
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,steps_per_epoch=200,
                                                        validation_data=self.datasets.dataset_cv,validation_steps=None,
                                                        verbose=1))
            elif type(latent_dim) == list:
                for dim in latent_dim:
                    if self.model.imported == False:
                        model = Model(input_dim,dim,enc_hidden_layers,dec_hidden_layers,alpha,l2_reg,l1_reg,dropout,
                                           activation,mode='train')
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,steps_per_epoch=200,
                                                        validation_data=self.datasets.dataset_cv,validation_steps=None,
                                                        verbose=1))

    def generate_samples(self, parameters):

        ## BUILD DECODER ##
        output_dim = parameters.img_size
        latent_dim = parameters.training_parameters['latent_dim']
        alpha = parameters.training_parameters['learning_rate']
        dec_hidden_layers = parameters.training_parameters['dec_hidden_layers']
        activation = parameters.training_parameters['activation']
        training_size = parameters.training_parameters['train_size']
        batch_size = parameters.training_parameters['batch_size']
        n_samples = self.parameters.samples_generation['n_samples']
        
        decoder = models.VAE(output_dim,latent_dim,[],dec_hidden_layers,alpha,0.0,0.0,0.0,activation,'sample')  # No regularization
        
        X_samples = []
        for model in self.model.Model:
            # Retrieve decoder weights
            j = 0
            for layer in model.layers:
                if layer.name.startswith('decoder') == False:
                    j += len(layer.weights)
                else:
                    break
            decoder_input_layer_idx = j

            decoder_weights = model.get_weights()[decoder_input_layer_idx:]
            decoder.set_weights(decoder_weights)

            ## SAMPLE IMAGES ##
            samples = np.zeros([n_samples,np.prod(output_dim)])
            for i in range(n_samples):
                t = tf.random.normal(shape=(1,latent_dim))
                samples[i,:] = decoder.predict(t,steps=1)
            X_samples.append(samples)

        return X_samples

    def export_model_performance(self, sens_var=None):

        try:
            History = self.model.History
        except:
            raise Exception('There is no evolution data for this model. Train model first.')
        else:
            if type(History) == list:
                N = len(History)
            else:
                N = 1
                History = [History]

            # Loss evolution plots #
            Nepochs = self.parameters.training_parameters['epochs']
            epochs = np.arange(1,Nepochs+1,1)

            case_ID = self.parameters.analysis['case_ID']
            for i,h in enumerate(History):
                loss_train = h.history['loss']
                loss_cv = h.history['val_loss']

                fig, ax = plt.subplots(1)
                ax.plot(epochs,loss_train,label='Training',color='r')
                ax.plot(epochs,loss_cv,label='Cross-validation',color='b')
                ax.grid()
                ax.set_xlabel('Epochs',size=12)
                ax.set_ylabel('Loss',size=12)
                ax.tick_params('both',labelsize=10)
                ax.legend()
                plt.suptitle('Loss evolution case = {}'.format(str(case_ID)))

                if sens_var:
                    if type(sens_var[1][i]) == str:
                        storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model_performance',
                                                   '{}={}'.format(sens_var[0],sens_var[1][i]))
                    else:
                        storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model_performance',
                                                   '{}={:.3f}'.format(sens_var[0],sens_var[1][i]))
                    loss_plot_filename = 'Loss_evolution_{}_{}={}.png'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
                    loss_filename = 'Model_loss_{}_{}={}.csv'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
                    metrics_filename = 'Model_metrics_{}_{}={}.csv'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
                else:
                    storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model_performance')
                    loss_plot_filename = 'Loss_evolution_{}.png'.format(str(case_ID))
                    loss_filename = 'Model_loss_{}.csv'.format(str(case_ID))
                    metrics_filename = 'Model_metrics_{}.csv'.format(str(case_ID))
                    
                if os.path.exists(storage_dir):
                    rmtree(storage_dir)
                os.makedirs(storage_dir)
                fig.savefig(os.path.join(storage_dir,loss_plot_filename),dpi=200)
                plt.close()

                # Metrics #
                metrics_name = [item for item in h.history if item not in ('loss','val_loss')]
                metrics_val = [(metric,h.history[metric][0]) for metric in metrics_name if metric.startswith('val')]
                metrics_train = [(metric,h.history[metric][0]) for metric in metrics_name if not metric.startswith('val')]

                rows = [metric[0] for metric in metrics_train]
                metric_fun = lambda L: np.array([item[1] for item in L])
                metrics_data = np.vstack((metric_fun(metrics_train),metric_fun(metrics_val))).T
                metrics = pd.DataFrame(index=rows,columns=['Training','CV'],data=metrics_data)
                metrics.to_csv(os.path.join(storage_dir,metrics_filename),sep=';',decimal='.')

                # Loss
                loss_data = np.vstack((list(epochs), loss_train, loss_cv)).T
                loss = pd.DataFrame(columns=['Epoch', 'Training', 'CV'], data=loss_data)
                loss.to_csv(os.path.join(storage_dir,loss_filename), index=False, sep=';', decimal='.')

    def export_model(self, sens_var=None):

        N = len(self.model.Model)
        case_ID = self.parameters.analysis['case_ID']
        for i in range(N):
            if sens_var:
                if type(sens_var[1][i]) == str:
                    storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={}'
                                               .format(sens_var[0],sens_var[1][i]))
                else:
                    storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={:.3f}'
                                               .format(sens_var[0],sens_var[1][i]))
                model_json_name = 'CGVAE_model_{}_{}={}_arquitecture.json'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
                model_weights_name = 'CGVAE_model_{}_{}={}_weights.h5'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
                model_folder_name = 'CGVAE_model_{}_{}={}'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
            else:
                storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model')
                model_json_name = 'CGVAE_model_{}_arquitecture.json'.format(str(case_ID))
                model_weights_name = 'CGVAE_model_{}_weights.h5'.format(str(case_ID))
                model_folder_name = 'CGVAE_model_{}'.format(str(case_ID))

            if os.path.exists(storage_dir):
                rmtree(storage_dir)
            os.makedirs(storage_dir)

            # Export history training
            with open(os.path.join(storage_dir,'History'),'wb') as f:
                pickle.dump(self.model.History[i].history,f)

            # Save model
            # Export model arquitecture to JSON file
            model_json = self.model.Model[i].to_json()
            with open(os.path.join(storage_dir,model_json_name),'w') as json_file:
                json_file.write(model_json)
            self.model.Model[i].save(os.path.join(storage_dir,model_folder_name.format(str(case_ID))))

            # Export model weights to HDF5 file
            self.model.Model[i].save_weights(os.path.join(storage_dir,model_weights_name))

    def reconstruct_model(self, mode='train'):

        storage_dir = os.path.join(self.case_dir,'Results','pretrained_model')
        try:
            casedata = reader.read_case_logfile(os.path.join(storage_dir,'CGVAE.log'))
            img_dim = casedata.img_size
            latent_dim = casedata.training_parameters['latent_dim']
            enc_hidden_layers = casedata.training_parameters['enc_hidden_layers']
            dec_hidden_layers = casedata.training_parameters['dec_hidden_layers']
            activation = casedata.training_parameters['activation']

            # Load weights into new model
            Model = models.VAE(img_dim,latent_dim,enc_hidden_layers,dec_hidden_layers,0.001,0.0,0.0,0.0,activation,
                               mode)
            weights_filename = [file for file in os.listdir(storage_dir) if file.endswith('.h5')][0]
            Model.load_weights(os.path.join(storage_dir,weights_filename))
            class history_container:
                pass
            History = history_container()
            with open(os.path.join(storage_dir,'History'),'rb') as f:
                History.history = pickle.load(f)
            History.epoch = None
            History.model = Model
        except:
            tf.config.run_functions_eagerly(True) # Enable eager execution
            try:
                model_folder = next(os.walk(storage_dir))[1][0]
            except:
                print('There is no model stored in the folder')

            alpha = self.parameters.training_parameters['learning_rate']
            loss = models.loss_function

            Model = tf.keras.models.load_model(os.path.join(storage_dir,model_folder),custom_objects={'loss':loss},compile=False)
            Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),loss=lambda x, y: loss,
                          metrics=[tf.keras.metrics.MeanSquaredError()])

            tf.config.run_functions_eagerly(False) # Disable eager execution

            # Reconstruct history
            class history_container:
                pass
            History = history_container()
            try:
                with open(os.path.join(storage_dir,'History'),'rb') as f:
                    History.history = pickle.load(f)
                History.epoch = np.arange(1,len(History.history['loss'])+1)
                History.model = Model
            except:
                History.epoch = None
                History.model = None

        return Model, History

    def reconstruct_encoder_CNN(self):

        img_dim = self.parameters.img_size
        latent_dim = self.parameters.training_parameters['latent_dim']
        enc_hidden_layers = self.parameters.training_parameters['enc_hidden_layers']
        dec_hidden_layers = self.parameters.training_parameters['dec_hidden_layers']
        activation = self.parameters.training_parameters['activation']

        storage_dir = os.path.join(self.case_dir,'Results','pretrained_model')

        #Encoder = models.encoder_lenet(img_dim,latent_dim,enc_hidden_layers,0.0,0.0,0.0,activation)
        Encoder = models.encoder(np.prod(img_dim),enc_hidden_layers,latent_dim,activation)
        Encoder.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError())

        # Load weights into new model
        Model = models.VAE(img_dim,latent_dim,enc_hidden_layers,dec_hidden_layers,0.001,0.0,0.0,0.0,'relu','train')
        Model.load_weights(os.path.join(storage_dir,'CGVAE_model_weights.h5'))
        enc_CNN_last_layer_idx = [idx for (idx,weight) in enumerate(Model.weights) if weight.shape[0] == latent_dim][0]
        encoder_weights = Model.get_weights()[:enc_CNN_last_layer_idx]
        Encoder.set_weights(encoder_weights)

        return Encoder

    def export_log(self):
        def update_log(parameters, model):
            training = OrderedDict()
            training['TRAINING SIZE'] = parameters.training_parameters['train_size']
            training['LEARNING RATE'] = parameters.training_parameters['learning_rate']
            training['L2 REGULARIZER'] = parameters.training_parameters['l2_reg']
            training['L1 REGULARIZER'] = parameters.training_parameters['l1_reg']
            training['DROPOUT'] = parameters.training_parameters['dropout']
            training['ACTIVATION'] = parameters.training_parameters['activation']
            training['NUMBER OF EPOCHS'] = parameters.training_parameters['epochs']
            training['BATCH SIZE'] = parameters.training_parameters['batch_size']
            training['LATENT DIMENSION'] = parameters.training_parameters['latent_dim']
            training['ENCODER HIDDEN LAYERS'] = parameters.training_parameters['enc_hidden_layers']
            training['DECODER HIDDEN LAYERS'] = parameters.training_parameters['dec_hidden_layers']
            training['OPTIMIZER'] = [model.optimizer._name for model in model.Model]
            training['METRICS'] = [model.metrics_names[-1] if model.metrics_names != None else None for model in model.Model]

            analysis = OrderedDict()
            analysis['CASE ID'] = parameters.analysis['case_ID']
            analysis['ANALYSIS'] = parameters.analysis['type']
            analysis['IMPORTED MODEL'] = parameters.analysis['import']
            analysis['LAST TRAINING LOSS'] = ['{:.3f}'.format(history.history['loss'][-1]) for history in model.History]
            analysis['LAST CV LOSS'] = ['{:.3f}'.format(history.history['val_loss'][-1]) for history in model.History]

            architecture = OrderedDict()
            architecture['INPUT SHAPE'] = parameters.img_size

            return training, analysis, architecture


        parameters = self.parameters
        if parameters.analysis['type'] == 'sensanalysis':
            varname, varvalues = parameters.sens_variable
            for value in varvalues:
                parameters.training_parameters[varname] = value
                training, analysis, architecture = update_log(parameters,self.model)

                case_ID = parameters.analysis['case_ID']
                if type(value) == str:
                    storage_folder = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={}'.format(varname,value))
                else:
                    storage_folder = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={:.3f}'.format(varname,value))
                with open(os.path.join(storage_folder,'CGVAE.log'),'w') as f:
                    f.write('CGVAE log file\n')
                    f.write('==================================================================================================\n')
                    f.write('->ANALYSIS\n')
                    for item in analysis.items():
                        f.write(item[0] + '=' + str(item[1]) + '\n')
                    f.write('--------------------------------------------------------------------------------------------------\n')
                    f.write('->TRAINING\n')
                    for item in training.items():
                        f.write(item[0] + '=' + str(item[1]) + '\n')
                    f.write('--------------------------------------------------------------------------------------------------\n')
                    f.write('->ARCHITECTURE\n')
                    for item in architecture.items():
                        f.write(item[0] + '=' + str(item[1]) + '\n')
                    f.write('--------------------------------------------------------------------------------------------------\n')
                    f.write('->MODEL\n')
                    for model in self.model.Model:
                        model.summary(print_fn=lambda x: f.write(x + '\n'))
                    f.write('==================================================================================================\n')

        else:
            training, analysis, architecture = update_log(self.parameters,self.model)
            case_ID = parameters.analysis['case_ID']
            storage_folder = os.path.join(self.case_dir,'Results',str(case_ID))
            with open(os.path.join(storage_folder,'Model','CGVAE.log'),'w') as f:
                f.write('CGVAE log file\n')
                f.write(
                    '==================================================================================================\n')
                f.write('->ANALYSIS\n')
                for item in analysis.items():
                    f.write(item[0] + '=' + str(item[1]) + '\n')
                f.write(
                    '--------------------------------------------------------------------------------------------------\n')
                f.write('->TRAINING\n')
                for item in training.items():
                    f.write(item[0] + '=' + str(item[1]) + '\n')
                f.write(
                    '--------------------------------------------------------------------------------------------------\n')
                f.write('->ARCHITECTURE\n')
                for item in architecture.items():
                    f.write(item[0] + '=' + str(item[1]) + '\n')
                f.write(
                    '--------------------------------------------------------------------------------------------------\n')
                f.write('->MODEL\n')
                for model in self.model.Model:
                    model.summary(print_fn=lambda x: f.write(x + '\n'))
                f.write(
                    '==================================================================================================\n')
if __name__ == '__main__':
    launcher = r'C:\Users\juan.ramos\PRAE\Scripts\launcher.dat'
    trainer = CGenTrainer(launcher)
    trainer.launch_analysis()