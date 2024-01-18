import os
import numpy as np
import tensorflow as tf


def swish(x, beta=1):
    return x * tf.keras.backend.sigmoid(beta * x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sampling(args):
    """Returns sample from a distribution N(args[0], diag(args[1]))
    Sampling from the distribution q(t|x) = N(t_mean, exp(t_log_var)) with reparametrization trick.

    The sample should be computed with reparametrization trick.

    The inputs are tf.Tensor
        args[0]: (batch_size x latent_dim) mean of the desired distribution
        args[1]: (batch_size x latent_dim) logarithm of the variance vector of the desired distribution

    Returns:
        A tf.Tensor of size (batch_size x latent_dim), the samples.
    """
    t_mean, t_log_var = args

    epsilon = tf.random.normal(tf.shape(t_log_var),name="epsilon")

    return t_mean + epsilon * tf.exp(t_log_var/2)


#@tf.function
def loss_function(x, x_decoded, t_mean, t_log_var):
    """Returns the value of negative Variational Lower Bound

    The inputs are tf.Tensor
        x: (batch_size x number_of_pixels) matrix with one image per row with zeros and ones
        x_decoded: (batch_size x number_of_pixels) mean of the distribution p(x | t), real numbers from 0 to 1
        t_mean: (batch_size x latent_dim) mean vector of the (normal) distribution q(t | x)
        t_log_var: (batch_size x latent_dim) logarithm of the variance vector of the (normal) distribution q(t | x)

    Returns:
        A tf.Tensor with one element (averaged across the batch), VLB
    """

    loss = tf.reduce_sum(x * tf.math.log(x_decoded + 1e-19) + (1 - x) * tf.math.log(1 - x_decoded + 1e-19), axis=1)
    regularisation = 0.5 * tf.reduce_sum(-t_log_var + tf.math.exp(t_log_var) + tf.math.square(t_mean) - 1, axis=1)

    return tf.reduce_mean(-loss + regularisation, axis=0)

def conv2D_block(X, num_channels, f, p, s, dropout,bn=True, **kwargs):

    if kwargs:
        parameters = list(kwargs.values())[0]
        l2_reg = parameters['l2_reg']
        l1_reg = parameters['l1_reg']
        activation = parameters['activation']
        res = parameters['res']
        s_res = parameters['s_res']
    else:
        l2_reg = 0.0
        l1_reg = 0.0
        activation = 'relu'
        res = None
        s_res = None

    # Apply padding
    if p == 'same':
        net = tf.keras.layers.Conv2D(num_channels,kernel_size=f,strides=s,padding='same',use_bias=True,
                                     kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(X)
    elif type(p) == int:
        net = tf.keras.layers.ZeroPadding2D(p)(X)
        net = tf.keras.layers.Conv2D(num_channels,kernel_size=f,strides=s,padding='valid',use_bias=True,
                                     kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)

    # Apply Batch-Normalization
    if bn == True:
        net = tf.keras.layers.BatchNormalization()(net)

    if res != None:
        res = tf.keras.layers.Conv2D(num_channels,kernel_size=1,strides=s_res,padding='same',use_bias=True,
                                     kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(res)
        net = tf.keras.layers.Add()([net,res])

    if activation == 'leakyrelu':
        rate = 0.3
        net = tf.keras.layers.LeakyReLU(rate)(net)
    elif activation == 'swish':
        net = tf.keras.layers.Activation('swish')(net)
    elif activation == 'elu':
        net = tf.keras.layers.ELU(net)
    elif activation == 'tanh':
        net = tf.keras.activations.tanh(net)
    elif activation == 'sigmoid':
        net = tf.keras.activations.sigmoid(net)
    elif activation == 'linear':
        net = tf.keras.activations('linear')(net)
    else:
        net = tf.keras.layers.Activation('relu')(net)

    return net

def conv2Dtranspose_block(X, num_channels, f, p, s, dropout, **kwargs):

    if kwargs:
        parameters = list(kwargs.values())[0]
        l2_reg = parameters['l2_reg']
        l1_reg = parameters['l1_reg']
        activation = parameters['activation']
    else:
        l2_reg = 0.0
        l1_reg = 0.0
        activation = 'relu'

    # Apply padding
    if type(p) == str:
        net = tf.keras.layers.Conv2DTranspose(num_channels,kernel_size=f,strides=s,padding=p,use_bias=True,
                                              kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg),
                                              kernel_initializer='glorot_normal')(X)
    elif type(p) == int:
        net = tf.keras.layers.Conv2D(num_channels,kernel_size=f,strides=s,output_padding=p,use_bias=True,
                                     kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(X)

    net = tf.keras.layers.BatchNormalization()(net)

    if activation == 'leakyrelu':
        rate = 0.2
        net = tf.keras.layers.LeakyReLU(rate)(net)
    elif activation == 'swish':
        net = tf.keras.layers.Activation('swish')(net)
    elif activation == 'elu':
        net = tf.keras.layers.ELU(net)
    elif activation == 'tanh':
        net = tf.keras.activations.tanh(net)
    elif activation == 'sigmoid':
        net = tf.keras.activations.sigmoid(net)
    elif activation == 'linear':
        net = tf.keras.activations('linear')(net)
    else:
        net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.Dropout(dropout)(net)

    return net

def dense_layer(X, units, activation, dropout, l1_reg, l2_reg):

    net = tf.keras.layers.Dense(units=units,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(X)
    net = tf.keras.layers.BatchNormalization()(net)
    if activation == 'leakyrelu':
        rate = 0.1
        net = tf.keras.layers.LeakyReLU(rate)(net)
    elif activation == 'elu':
        net = tf.keras.layers.ELU()(net)
    else:
        net = tf.keras.layers.Activation(activation)(net)
    net = tf.keras.layers.Dropout(dropout)(net)

    return net

def encoder_cnn(input_dim, latent_dim, hidden_layers, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):
        
    X_input = tf.keras.layers.Input(shape=input_dim)
    net = conv2D_block(X_input,num_channels=12,f=5,p='same',s=2,dropout=dropout,
                       kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation,'res':None,'s_res':None})
    res_net_1 = net
    net = conv2D_block(net,num_channels=16,f=3,p='same',s=2,dropout=dropout,
                       kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation,'res':None,'s_res':None})
    net = conv2D_block(net,num_channels=32,f=3,p='same',s=1,dropout=dropout,
                       kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation,'res':res_net_1,'s_res':2})
    net = tf.keras.layers.GlobalMaxPool2D()(net)
    '''
    for layer in hidden_layers:
        net = dense_layer(net,layer,activation,dropout,l1_reg,l2_reg)
    '''
    net = tf.keras.layers.Dense(units=2*latent_dim,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)

    encoder = tf.keras.Model(inputs=X_input,outputs=net,name='encoder_cnn')
    #encoder.summary()
    
    return encoder

def decoder_cnn(output_dim, latent_dim, hidden_layers, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):

    f0 = 32
    dense_layer0 = (output_dim[0]//2,output_dim[1]//2,f0)
    X_input = tf.keras.layers.Input(shape=latent_dim)
    net = dense_layer(X_input,np.prod(dense_layer0),activation,dropout,l1_reg,l2_reg)
    net = tf.keras.layers.Reshape(dense_layer0)(net)
    net = conv2Dtranspose_block(net,num_channels=f0//2,f=3,p='same',s=2,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation})
    net = tf.keras.layers.UpSampling2D()(net)
    net = conv2D_block(net,num_channels=1,f=5,p='same',s=2,dropout=0.0,bn=False,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':'sigmoid','res':None,'s_res':None})

    decoder = tf.keras.Model(inputs=X_input,outputs=net,name='decoder_cnn')
    #decoder.summary()

    return decoder
    
def encoder(input_dim, hidden_dim, latent_dim, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):
    '''
    Encoder network.
    Returns the mean and the log variances of the latent distribution
    '''

    X_input = tf.keras.Input(shape=input_dim)
    net = tf.keras.layers.Reshape((np.prod(input_dim),))(X_input)
    for hidden_layer_dim in hidden_dim:
        net = dense_layer(net,hidden_layer_dim,activation,dropout,l1_reg,l2_reg)
    net = tf.keras.layers.Dense(2*latent_dim,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)

    encoder = tf.keras.Model(inputs=X_input,outputs=net,name='encoder')

    return encoder

def decoder(latent_dim, hidden_dim, output_dim, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):
    '''
    Decoder network
    It assumes that the image is a normalized black & white image so each pixel ranges between 0 and 1
    '''
    
    decoder = tf.keras.Sequential(name='decoder')
    decoder.add(tf.keras.Input(shape=(latent_dim,)))
    for hidden_layer_dim in hidden_dim:
        decoder.add(tf.keras.layers.Dense(hidden_layer_dim,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg)))
        decoder.add(tf.keras.layers.BatchNormalization())
        if activation == 'leakyrelu':
            rate = 0.3
            decoder.add(tf.keras.layers.LeakyReLU(rate))
        else:
            decoder.add(tf.keras.layers.Activation(activation))
        decoder.add(tf.keras.layers.Dropout(dropout))
    decoder.add(tf.keras.layers.Dense(output_dim,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg),activation='sigmoid'))
    
    return decoder

def VAE(input_dim, latent_dim, encoder_hidden_layers, decoder_hidden_layers, alpha, l2_reg=0.0, l1_reg=0.0, dropout=0.0,
        activation='relu', mode='train'):

    in_shape_unrolled = np.prod(input_dim)

    ## DEFINE MODEL ##
    if mode == 'train':
        # Set models
        # e = encoder(input_dim,encoder_hidden_layers,latent_dim,l2_reg,l1_reg,dropout,activation)
        e = encoder_cnn(input_dim,latent_dim,encoder_hidden_layers,l2_reg,l1_reg,dropout,activation)
        # d = decoder(latent_dim,decoder_hidden_layers,in_shape_unrolled,l2_reg,l1_reg,dropout,activation)
        d = decoder_cnn(input_dim,latent_dim,decoder_hidden_layers,l2_reg,l1_reg,dropout,activation)

        # Encoder
        #x = tf.keras.Input(shape=(in_shape_unrolled,))
        x = tf.keras.Input(shape=input_dim)
        h = e(x)

        # Decoder
        get_t_mean = tf.keras.layers.Lambda(lambda h: h[:,:latent_dim])
        get_t_log_var = tf.keras.layers.Lambda(lambda h: h[:,latent_dim:])
        t_mean = get_t_mean(h)
        t_log_var = get_t_log_var(h)
        t = tf.keras.layers.Lambda(sampling)([t_mean,t_log_var])
        x_decoded = d(t)

        # Declare inputs/outputs for the model
        input = x
        output = x_decoded
    elif mode == 'decoder':
        # Set model
        # d = decoder(latent_dim,decoder_hidden_layers,in_shape_unrolled,l2_reg,l1_reg,dropout,activation)
        d = decoder_cnn(input_dim,latent_dim,decoder_hidden_layers,l2_reg,l1_reg,dropout,activation)

        # Sample
        t = tf.keras.Input(shape=(latent_dim,))
        t_mean = tf.zeros_like(t)
        t_log_var = tf.zeros_like(t)
        x_decoded = d(t)

        # Declare inputs/outputs for the model
        input = t
        output = x_decoded
        x = x_decoded
    elif mode == 'encoder':
        # Set encoder
        # e = encoder(input_dim,encoder_hidden_layers,latent_dim,l2_reg,l1_reg,dropout,activation)
        e = encoder_cnn(input_dim,latent_dim,encoder_hidden_layers,l2_reg,l1_reg,dropout,activation)

        # Encoder
        #x = tf.keras.Input(shape=(in_shape_unrolled,))
        x = tf.keras.Input(shape=input_dim)
        h = e(x)

        # Decoder
        get_t_mean = tf.keras.layers.Lambda(lambda h: h[:,:latent_dim])
        get_t_log_var = tf.keras.layers.Lambda(lambda h: h[:,latent_dim:])
        t_mean = get_t_mean(h)
        t_log_var = get_t_log_var(h)
        t = tf.keras.layers.Lambda(sampling)([t_mean,t_log_var])

        # Declare inputs/outputs for the model
        input = x
        output = t
        x_decoded = x

    loss = loss_function(x,x_decoded,t_mean,t_log_var)
    model = tf.keras.Model(input,output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),loss=lambda x,y: loss,
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    return model

def VAE_npi(input_dim, latent_dim, encoder_hidden_layers, decoder_hidden_layers, alpha, l2_reg=0.0, l1_reg=0.0, dropout=0.0,
        activation='relu', mode='train'):

    in_shape_unrolled = np.prod(input_dim)

    e = encoder(in_shape_unrolled,encoder_hidden_layers,latent_dim,l2_reg,l1_reg,dropout,activation)
    #e = encoder_cnn(input_dim,latent_dim,encoder_hidden_layers,l2_reg,l1_reg,dropout,activation)
    d = decoder(latent_dim,decoder_hidden_layers,in_shape_unrolled,l2_reg,l1_reg,dropout,activation)
    #d = decoder_cnn(input_dim,latent_dim,decoder_hidden_layers,l2_reg,l1_reg,dropout,activation)

    ## DEFINE MODEL ##
    if mode == 'train':
        # Encoder
        x = tf.keras.Input(shape=(in_shape_unrolled,))
        h = e(tf.keras.layers.concatenate([x,x_p]))

        # Decoder
        get_t_mean = tf.keras.layers.Lambda(lambda h: h[:,:latent_dim])
        get_t_log_var = tf.keras.layers.Lambda(lambda h: h[:,latent_dim:])
        t_mean = get_t_mean(h)
        t_log_var = get_t_log_var(h)
        t = tf.keras.layers.Lambda(sampling)([t_mean,t_log_var])
        x_decoded = d(t)

        # Declare inputs/outputs for the model
        input = [x,x_p]
        output = x_decoded
    elif mode == 'sample':
        # Decoder
        t = tf.keras.Input(shape=(latent_dim,))
        t_mean = tf.zeros_like(t)
        t_log_var = tf.zeros_like(t)
        x_decoded = d(t)

        # Declare inputs/outputs for the model
        input = t
        output = x_decoded
        x = x_decoded

    loss = loss_function(x,x_decoded,t_mean,t_log_var)
    model = tf.keras.Model(input,output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),loss=lambda x,y: loss,
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    return model

