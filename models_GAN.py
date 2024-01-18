import os
import numpy as np
import tensorflow as tf

def cost_function():

    return tf.keras.losses.BinaryCrossentropy()

def loss_function(model, Discriminator, Encoder, fake_logit, fake_label, real_logit, real_label, fake, real, reg):

    J = cost_function()
    if model == 'disc':
        loss = 0.5 * (J(fake_logit,fake_label) + J(real_logit,real_label))
        #loss = -tf.math.reduce_mean(real_logit-fake_logit)
        # Mode collapse regularization penalty
        discriminator_gradient = get_gradient(Discriminator,Encoder,real,fake)
        mode_collapse_reg = gradient_penalty(discriminator_gradient)
    elif model == 'gen':
        loss = J(fake_logit,fake_label)
        #loss = -tf.math.reduce_mean(fake_logit)
        mode_collapse_reg = tf.constant(0.0)

    return loss + reg * mode_collapse_reg

def optimizer(alpha):

    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha,beta_1=0.9,beta_2=0.999,amsgrad=False)

    return optimizer

def base_metric():

    return tf.keras.metrics.MeanSquaredError()
    
def performance_metric(logits, labels):

    metric = tf.reduce_mean(base_metric()(logits,labels))

    return metric

def get_gradient(Discriminator, Encoder, real_0, fake_0):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    '''
    mixed_images = tf.Variable(real_0,shape=real_0.get_shape())
    real = tf.Variable(real_0,shape=real_0.get_shape())
    fake = tf.Variable(fake_0,shape=fake_0.get_shape())
    epsilon_0 = tf.random.uniform(real_0.get_shape())
    epsilon = tf.Variable(epsilon_0,shape=epsilon_0.get_shape())
    with tf.GradientTape() as tape_gradient:
        mixed_images = real*epsilon + fake*(1 - epsilon)
        t = Encoder(mixed_images,training=False)
        mixed_scores = Discriminator(mixed_images,t)   # Calculate the critic's scores on the mixed images
    gradient = tape_gradient.gradient(target=mixed_scores,sources=mixed_images)

    return gradient


def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    input_shape = gradient.get_shape()[1:].num_elements()
    batch_size = gradient.get_shape()[0]
    gradient = tf.reshape(gradient,(batch_size,input_shape))

    # Calculate the magnitude of every row
    gradient_norm = tf.math.reduce_euclidean_norm(gradient,axis=1)
    # Penalize the mean squared distance of the gradient norms from 1
    penalty = tf.reduce_mean((gradient_norm - tf.ones_like(gradient_norm))**2)

    return penalty

class Conv2D_block(tf.keras.Model):
    def __init__(self, num_channels, kernel_size, padding, stride, **kwargs):
        super(Conv2D_block,self).__init__()
        if kwargs:
            parameters = list(kwargs.values())[0]
            self.l2_reg = parameters['l2_reg']
            self.l1_reg = parameters['l1_reg']
            dropout = parameters['dropout']
            activation = parameters['activation']
        else:
            self.l2_reg = 0.0
            self.l1_reg = 0.0
            dropout = 0.0
            activation = 'relu'

        # Apply padding
        if padding == 'same':
            self.Padding = tf.keras.layers.ZeroPadding2D(0)
            self.Conv2D = tf.keras.layers.Conv2D(num_channels,kernel_size,stride,padding='same',use_bias=True,
                                                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_reg,l2=self.l2_reg),
                                                 kernel_initializer='glorot_normal')
        elif type(padding) == int:
            self.Padding = tf.keras.layers.ZeroPadding2D(padding)
            self.Conv2D = tf.keras.layers.Conv2D(num_channels,kernel_size,stride,padding='valid',use_bias=True,
                                                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_reg,l2=self.l2_reg),
                                                 kernel_initializer='glorot_normal')

        self.BatchNorm = tf.keras.layers.BatchNormalization()

        if activation == 'leakyrelu':
            rate = 0.2
            self.Activation = tf.keras.layers.LeakyReLU(rate)
        elif activation == 'swish':
            self.Activation = tf.keras.layers.Activation('swish')
        elif activation == 'elu':
            self.Activation = tf.keras.activations.elu
        elif activation == 'tanh':
            self.Activation = tf.keras.activations.tanh
        elif activation == 'sigmoid':
            self.Activation = tf.keras.activations.sigmoid
        elif activation == 'linear':
            self.Activation = tf.keras.activations('linear')
        else:
            self.Activation = tf.keras.layers.Activation('relu')
        self.Dropout = tf.keras.layers.Dropout(dropout)

    def __call__(self, X):

        net = self.Padding(X)
        net = self.Conv2D(net)
        net = self.BatchNorm(net)
        net = self.Activation(net)
        net = self.Dropout(net)

        return net

class Conv2DTranspose_block(tf.keras.Model):
    def __init__(self, num_channels, kernel_size, stride, **kwargs):
        super(Conv2DTranspose_block,self).__init__()

        if kwargs:
            parameters = list(kwargs.values())[0]
            self.l2_reg = parameters['l2_reg']
            self.l1_reg = parameters['l1_reg']
            dropout = parameters['dropout']
            activation = parameters['activation']
        else:
            self.l2_reg = 0.0
            self.l1_reg = 0.0
            dropout = parameters['dropout']
            activation = 'relu'

        self.Conv2DTranspose = tf.keras.layers.Conv2DTranspose(num_channels,kernel_size,stride,padding='same',
                                                               kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_reg,l2=self.l2_reg),
                                                               kernel_initializer='glorot_normal')
        self.BatchNorm = tf.keras.layers.BatchNormalization()

        if activation == 'leakyrelu':
            rate = 0.2
            self.Activation = tf.keras.layers.LeakyReLU(rate)
        elif activation == 'swish':
            self.Activation = tf.keras.layers.Activation('swish')
        elif activation == 'elu':
            self.Activation = tf.keras.activations.elu
        elif activation == 'tanh':
            self.Activation = tf.keras.activations.tanh
        elif activation == 'sigmoid':
            self.Activation = tf.keras.activations.sigmoid
        elif activation == 'linear':
            self.Activation = tf.keras.activations('linear')
        else:
            self.Activation = tf.keras.layers.Activation('relu')
        self.Dropout = tf.keras.layers.Dropout(dropout)

    def __call__(self, X):

        net = self.Conv2DTranspose(X)
        net = self.Dropout(net)
        net = self.BatchNorm(net)
        net = self.Activation(net)

        return net

class Dense_layer(tf.keras.Model):
    def __init__(self, units, activation, l1_reg, l2_reg, dropout):
        super(Dense_layer, self).__init__()
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        self.Dense = tf.keras.layers.Dense(units=units,activation=None,kernel_initializer='glorot_normal',
                                           kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1_reg,l2=self.l2_reg))
        self.Dropout = tf.keras.layers.Dropout(dropout)
        self.BatchNorm = tf.keras.layers.BatchNormalization()
        if activation == 'leakyrelu':
            rate = 0.2
            self.Activation = tf.keras.layers.LeakyReLU(rate)
        elif activation == 'elu':
            self.Activation = tf.keras.activations.elu
        else:
            self.Activation = tf.keras.layers.Activation(activation)

    def call(self, X):

        net = self.Dense(X)
        net = self.Dropout(net)
        net = self.BatchNorm(net)
        net = self.Activation(net)

        return net

class Discriminator(tf.keras.Model):

    """A simple linear model."""
    def __init__(self, activation, l2_reg, l1_reg, dropout):
        super(Discriminator, self).__init__()
        self.l1 = l1_reg
        self.l2 = l2_reg
        self.dropout = dropout

        self.Conv2D_1 = Conv2D_block(num_channels=64,kernel_size=5,padding='same',stride=2,
                                     kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'dropout':dropout,'activation':activation})
        self.Conv2D_2 = Conv2D_block(num_channels=128,kernel_size=3,padding='same',stride=2,
                                     kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'dropout':dropout,'activation':activation})
        self.Pool = tf.keras.layers.GlobalMaxPool2D()
        self.Dense_1 = Dense_layer(64,activation,l1_reg,l2_reg,dropout)
        self.Dense_2 = Dense_layer(1,'sigmoid',l1_reg,l2_reg,0.0)

        self.model = {
        'Conv2D': [
        self.Conv2D_1,
        self.Conv2D_2,
        ],
        'Dense':[
        self.Dense_1,
        self.Dense_2,
        ],
        'Pool': [
        self.Pool,
        ],
        }
    def set_up_CV_state(self):

        def set_up_conv2d_block(block):

            block.Conv2D.kernel_regularizer.l1 = 0.0
            block.Conv2D.kernel_regularizer.l2 = 0.0
            block.Dropout.rate = 0.0

            return block
        def set_up_dense_layer(layer):

            layer.Dense.kernel_regularizer.l1 = 0.0
            layer.Dense.kernel_regularizer.l2 = 0.0
            layer.Dropout.rate = 0.0

            return layer

        for layer_type, layers in self.model.items():
            for layer in layers:
                if layer_type == 'Conv2D':
                    layer = set_up_conv2d_block(layer)
                elif layer_type == 'Dense':
                    layer = set_up_dense_layer(layer)
                elif layer_type == 'Final':
                    layer.kernel_regularizer.l1 = 0.0
                    layer.kernel_regularizer.l2 = 0.0

    def set_up_training_state(self):

        def set_up_conv2d_block(block, l1, l2, dropout):

            block.Conv2D.kernel_regularizer.l1 = l1
            block.Conv2D.kernel_regularizer.l2 = l2
            block.Dropout.rate = dropout

            return block
        def set_up_dense_layer(layer, l1, l2, dropout):

            layer.Dense.kernel_regularizer.l1 = l1
            layer.Dense.kernel_regularizer.l2 = l2
            layer.Dropout.rate = dropout

            return layer

        for layer_type, layers in self.model.items():
            for layer in layers:
                if layer_type == 'Conv2D':
                    layer = set_up_conv2d_block(layer,self.l1,self.l2,self.dropout)
                elif layer_type == 'Dense':
                    layer = set_up_dense_layer(layer,self.l1,self.l2,self.dropout)
                elif layer_type == 'Final':
                    layer.kernel_regularizer.l1 = self.l1
                    layer.kernel_regularizer.l2 = self.l2

    def __call__(self, X):

        net = self.Conv2D_1(X)
        net = self.Conv2D_2(net)
        net = self.Pool(net)
        net = self.Dense_1(net)
        net = self.Dense_2(net)

        return net

class ConditionalDiscriminator(tf.keras.Model):

    """A simple linear model."""
    def __init__(self, latent_dim, activation, l2_reg, l1_reg, dropout):
        super(ConditionalDiscriminator, self).__init__()
        self.l1 = l1_reg
        self.l2 = l2_reg
        self.dropout = dropout

        self.Conv2D_1 = Conv2D_block(num_channels=64,kernel_size=5,padding='same',stride=2,
                                     kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'dropout':dropout,'activation':activation})
        self.Conv2D_2 = Conv2D_block(num_channels=128,kernel_size=3,padding='same',stride=2,
                                     kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'dropout':dropout,'activation':activation})
        self.Pool = tf.keras.layers.GlobalMaxPool2D()
        self.Dense_1 = Dense_layer(64,activation,l1_reg,l2_reg,dropout)
        self.Dense_2 = Dense_layer(2*latent_dim,activation,l1_reg,l2_reg,dropout)
        self.Dense_3 = Dense_layer(1,'sigmoid',l1_reg,l2_reg,0.0)

        self.model = {
        'Conv2D': [
        self.Conv2D_1,
        self.Conv2D_2,
        ],
        'Dense':[
        self.Dense_1,
        self.Dense_2,
        self.Dense_3,
        ],
        'Pool': [
        self.Pool,
        ],
        }
    def set_up_CV_state(self):

        def set_up_conv2d_block(block):

            block.Conv2D.kernel_regularizer.l1 = 0.0
            block.Conv2D.kernel_regularizer.l2 = 0.0
            block.Dropout.rate = 0.0

            return block
        def set_up_dense_layer(layer):

            layer.Dense.kernel_regularizer.l1 = 0.0
            layer.Dense.kernel_regularizer.l2 = 0.0
            layer.Dropout.rate = 0.0

            return layer

        for layer_type, layers in self.model.items():
            for layer in layers:
                if layer_type == 'Conv2D':
                    layer = set_up_conv2d_block(layer)
                elif layer_type == 'Dense':
                    layer = set_up_dense_layer(layer)
                elif layer_type == 'Final':
                    layer.kernel_regularizer.l1 = 0.0
                    layer.kernel_regularizer.l2 = 0.0

    def set_up_training_state(self):

        def set_up_conv2d_block(block, l1, l2, dropout):

            block.Conv2D.kernel_regularizer.l1 = l1
            block.Conv2D.kernel_regularizer.l2 = l2
            block.Dropout.rate = dropout

            return block
        def set_up_dense_layer(layer, l1, l2, dropout):

            layer.Dense.kernel_regularizer.l1 = l1
            layer.Dense.kernel_regularizer.l2 = l2
            layer.Dropout.rate = dropout

            return layer

        for layer_type, layers in self.model.items():
            for layer in layers:
                if layer_type == 'Conv2D':
                    layer = set_up_conv2d_block(layer,self.l1,self.l2,self.dropout)
                elif layer_type == 'Dense':
                    layer = set_up_dense_layer(layer,self.l1,self.l2,self.dropout)
                elif layer_type == 'Final':
                    layer.kernel_regularizer.l1 = self.l1
                    layer.kernel_regularizer.l2 = self.l2

    def __call__(self, X, t):

        net_1 = self.Conv2D_1(X)
        net_1 = self.Conv2D_2(net_1)
        net_1 = self.Pool(net_1)
        net_1 = self.Dense_1(net_1)
        net_2 = self.Dense_2(t)
        net = tf.keras.layers.Concatenate()([net_1,net_2])
        net = self.Dense_3(net)

        return net

class Generator(tf.keras.Model):
    def __init__(self, input_dim, activation, l2_reg, l1_reg, dropout):
        super(Generator,self).__init__()
        self.l1 = l1_reg
        self.l2 = l2_reg
        self.dropout = dropout

        filt_in = 128
        f = 4
        s = 2
        fh = int(input_dim[0]/(2*s))
        fw = int(input_dim[1]/(2*s))
        fc = 32
        self.Dense = Dense_layer(fh*fw*fc,activation,l1_reg,l2_reg,dropout)
        self.Reshape = tf.keras.layers.Reshape((fh,fw,fc))
        self.Conv2DTranspose_1 = Conv2DTranspose_block(num_channels=filt_in,kernel_size=f,stride=s,
                                                       kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'dropout':dropout,'activation':activation})
        self.Conv2DTranspose_2 = Conv2DTranspose_block(num_channels=filt_in//2,kernel_size=f,stride=s,
                                                       kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'dropout':dropout,'activation':activation})
        self.Conv2D = Conv2D_block(num_channels=1,kernel_size=7,padding='same',stride=1,
                                   kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'dropout':dropout,'activation':'sigmoid'})

        self.model = {
        'Conv2DTranspose': [
        self.Conv2DTranspose_1,
        self.Conv2DTranspose_2,
        ],
        'Conv2D':[
        self.Conv2D,
        ],
        'Dense':[
        self.Dense,
        ],
        'Reshape': [self.Reshape],
        }

        self.structure = [
            'Dense',
            'Reshape',
            'Conv2DTranspose_1',
            'Conv2DTranspose_2',
            'Conv2D',
        ]

    def set_up_training_state(self):
        def set_up_conv2dtranspose_block(block, l1, l2, dropout):

            block.Conv2DTranspose.kernel_regularizer.l1 = l1
            block.Conv2DTranspose.kernel_regularizer.l2 = l2
            block.Dropout.rate = dropout

            return block
        def set_up_conv2d_block(block, l1, l2, dropout):

            block.Conv2D.kernel_regularizer.l1 = l1
            block.Conv2D.kernel_regularizer.l2 = l2
            block.Dropout.rate = dropout

            return block

        def set_up_dense_layer(layer, l1, l2, dropout):

            layer.Dense.kernel_regularizer.l1 = l1
            layer.Dense.kernel_regularizer.l2 = l2
            layer.Dropout.rate = dropout

            return layer

        for layer_type, layers in self.model.items():
            for layer in layers:
                if layer_type == 'Conv2DTranspose':
                    layer = set_up_conv2dtranspose_block(layer,self.l1,self.l2,self.dropout)
                elif layer_type == 'Conv2D':
                    layer = set_up_conv2d_block(layer,self.l1,self.l2,self.dropout)
                elif layer_type == 'Dense':
                    layer = set_up_dense_layer(layer,self.l1,self.l2,self.dropout)

    def set_up_CV_state(self):

        def set_up_conv2dtranspose_block(block):

            block.Conv2DTranspose.kernel_regularizer.l1 = 0.0
            block.Conv2DTranspose.kernel_regularizer.l2 = 0.0
            block.Dropout.rate = 0.0

            return block
        def set_up_conv2d_block(block):

            block.Conv2D.kernel_regularizer.l1 = 0.0
            block.Conv2D.kernel_regularizer.l2 = 0.0
            block.Dropout.rate = 0.0

            return block
        def set_up_dense_layer(layer):

            layer.Dense.kernel_regularizer.l1 = 0.0
            layer.Dense.kernel_regularizer.l2 = 0.0
            layer.Dropout.rate = 0.0

            return layer

        for layer_type, layers in self.model.items():
            for layer in layers:
                if layer_type == 'Conv2DTranspose':
                    layer = set_up_conv2dtranspose_block(layer)
                elif layer_type == 'Conv2D':
                    layer = set_up_conv2d_block(layer)
                elif layer_type == 'Dense':
                    layer = set_up_dense_layer(layer)

    def __call__(self, X):

        net = self.Dense(X)
        net = self.Reshape(net)
        net = self.Conv2DTranspose_1(net)
        net = self.Conv2DTranspose_2(net)
        net = self.Conv2D(net)

        return net

def convert_to_keras_model(model, input_shape, output_shape, compilation_parameters):

    X_input = tf.keras.layers.Input(shape=input_shape)
    net = X_input
    for item in model.structure:
        net = getattr(model,item)(net)

    keras_model = tf.keras.Model(inputs=X_input,outputs=net,name='Keras_%s_Model'%model.name)
    keras_model.compile(optimizer=compilation_parameters['optimizer'],loss=compilation_parameters['loss'],metrics=compilation_parameters['metric'])

    return keras_model