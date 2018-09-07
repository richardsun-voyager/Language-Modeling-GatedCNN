import numpy as np
import tensorflow as tf
import functools
from nlp_blocks.encode_blocks import TCN_encode
from nlp_blocks.nn import regularizer
def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

class tcn_model:
    '''
    A Lanugage Model Based on Gated CNN
    '''
    def __init__(self, conf, is_train=True, is_bidirectional=False):
        #Input data place holders
        #tf.reset_default_graph()
        self.X = tf.placeholder(tf.int32, shape=(None, 
                                                 None), name="input_x")
        
        self.y = tf.placeholder(tf.int32, shape=(None, 
                                                 None), name="input_y")
        
        
        if is_bidirectional:
            self.X_reverse = tf.placeholder(tf.int32, shape=(None, 
                                                 None), name="input_x_reverse")
            self.y_reverse = tf.placeholder(tf.int32, shape=(None, 
                                                 None), name="input_y_reverse")

        self.is_train = is_train
        self.is_bidirectional = is_bidirectional
        self.conf = conf
        #Get hidden layer
        hidden_layers, hidden_layers_reverse = self.build_model
        self.out_layer = hidden_layers
        #print(hidden_layers)
        #self.hidden_layer = hidden_layers[-1]
        #Concatenate hidden layers
        #self.hidden_layers =  tf.concat(hidden_layers, 3) 
        #For the reverse part
        if hidden_layers_reverse:
            self.hidden_layer_reverse = hidden_layers_reverse[-1]
            #self.hidden_layers_reverse = tf.concat(hidden_layers_reverse, 3) 
        
        #Get the loss
        self.loss = self.build_loss
        if is_train:
            self.optimizer = self.optimize
            #self.cnn_hidden_emb = self.get_hidden_emb()
            self.create_summaries()
        
        
    def depthwise_conv(self, padded_input, kernel_shape, name, **kwargs):
        '''Depth Convolutional Layer'''
        # Kaiming intialization
        size = (kernel_shape[1] * kernel_shape[2], 
                kernel_shape[2] * kernel_shape[3])
        stddev = tf.constant(np.sqrt(2.0/size[0]), dtype=tf.float32)
        W_g = tf.get_variable(name='Wg'+name, 
                              initializer=stddev, dtype=tf.float32)
        init = tf.random_normal(kernel_shape, stddev=stddev, dtype=tf.float32)
        W_v = tf.get_variable(initializer=init, name='Wv'+name, dtype=tf.float32)
        W =  (W_g / tf.nn.l2_normalize(W_v, 0)) * W_v
        b = tf.get_variable(initializer=tf.zeros([size[1]]), 
                            name="b"+ name)
        
        conv = tf.nn.depthwise_conv2d(
            padded_input,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv"+name)
        conv = tf.nn.bias_add(conv, b)
        return conv
        
    
    def glu(self, kernel_shape, layer_input, layer_name, residual=None):
        """ Gated Linear Unit """
        # Pad the left side to prevent kernels from viewing future context
        #print(kernel_shape)
        kernel_width = kernel_shape[1]
        left_pad = kernel_width - 1
        paddings = [[0,0],[0,0],[left_pad,0],[0,0]]
        padded_input = tf.pad(layer_input, paddings, "CONSTANT")
        
        with tf.variable_scope('gated_cnn_block' + str(layer_name)):
            conv1 = self.depthwise_conv(padded_input, kernel_shape, name='linear')
            conv2 = self.depthwise_conv(padded_input, kernel_shape, name='gate')

            # Preactivation residual
            if residual is not None:
                conv1 = tf.add(conv1, residual)
                conv2 = tf.add(conv2, residual)

            h = tf.multiply(conv1, tf.sigmoid(conv2, name="sig"), name='gated_cnn')
            
            h = tf.layers.batch_normalization(h, training=self.is_train)

        return h
    
    @lazy_property
    def build_model(self):
        """ Setup the model after we have imported the data and know the vocabulary size """

        vocab_size = self.conf.vocab_size
        embedding_size = self.conf.embedding_size
        #Embddings
        with tf.variable_scope('embedding'):
            init = tf.random_normal([vocab_size, embedding_size], stddev=.01, dtype=tf.float32)
            self.word_embeddings = tf.get_variable(name='word_embedding',
                                               initializer=init)
            input_embeddings = tf.nn.embedding_lookup(self.word_embeddings, self.X)
            if self.is_train:
                input_embeddings = tf.nn.dropout(input_embeddings, 0.5)
                
        
        dropout_keep_rate = 1
        if self.is_train:
            dropout_keep_rate= 0.8
        
        num_filters_list = [128, 256, 256]
        outputs = TCN_encode(input_embeddings, num_filters_list,
                             dropout_keep_rate=dropout_keep_rate)
        
        return outputs, None
    
    def build_graph(self, input_embeddings_expanded):
        # [height, width, in_channels, out_channels]
        embedding_size = self.conf.embedding_size
        #Keep the hidden layers
        hidden_layers = []
        with tf.variable_scope('conv_layers'):
            hidden_layers.append(input_embeddings_expanded)
            kernel_shape = [1, 3, embedding_size, 1]
            h0 = self.glu(kernel_shape, input_embeddings_expanded, 0)
            h1 = self.glu(kernel_shape, h0, 1)
            h2 = self.glu(kernel_shape, h1, 2, h0)
            h3 = self.glu(kernel_shape, h2, 3)
            h4 = self.glu(kernel_shape, h3, 4, h2)
            hidden_layers.append(h4)

           #  For larger models with output projections:
            kernel_shape = [1, 3, 128, 2]
            h4a = self.glu(kernel_shape, h4, '14a')
            

            kernel_shape = [1, 3, 256, 1]
            h5 = self.glu(kernel_shape, h4a, 5)
            h6 = self.glu(kernel_shape, h5, 6, h4a)
            h7 = self.glu(kernel_shape, h6, 7)
            h8 = self.glu(kernel_shape, h7, 8, h6)
            h9 = self.glu(kernel_shape, h8, 9)
            hidden_layers.append(h9)
        return hidden_layers
        
    @lazy_property    
    def build_loss(self):
        # Output embeddings
        kernel_shape = [1, 3, 128, 2]
        last_hidden = self.out_layer
        vocab_size = self.conf.vocab_size
        output_weights_size = last_hidden.get_shape().as_list()[-1]
        stddev = 0.05
        #Output layer
        with tf.variable_scope('softmax'):
            init = tf.random_normal([vocab_size, output_weights_size], 
                                stddev=stddev, dtype=tf.float32)
            softmax_w = tf.get_variable(initializer=init, 
                                        name="output_weights", regularizer=regularizer)
            softmax_b = tf.get_variable(initializer=tf.zeros([vocab_size]), name="output_bias")

        labels = tf.reshape(self.y, (-1, 1))
        

        
        h = tf.reshape(last_hidden, (-1, output_weights_size))
        if self.is_train:
            h = tf.nn.dropout(h, 0.5)
        
        loss_forward = tf.nn.sampled_softmax_loss(
                                   softmax_w, softmax_b,
                                   labels, h,
                                   8192,
                                   vocab_size,
                                   num_true=1)
        loss = tf.reduce_mean(loss_forward)
        
        if self.is_bidirectional:
            last_hidden_reverse = self.hidden_layer_reverse
            labels_reverse = tf.reshape(self.y_reverse, (-1, 1))
            h_reverse = tf.reshape(last_hidden_reverse, (-1, output_weights_size))
            if self.is_train:
                h_reverse = tf.nn.dropout(h_reverse, 0.5)
            loss_backward = tf.nn.sampled_softmax_loss(
                                       softmax_w, softmax_b,
                                       labels_reverse, h_reverse,
                                       num_sampled=8192,
                                       num_classes=vocab_size,
                                       num_true=1)
        
            loss += tf.reduce_mean(loss_backward)
        self.perplexity = tf.exp(loss)

        return loss
    
    @lazy_property
    def optimize(self):
        cost = self.loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        #optimizer = tf.train.MomentumOptimizer(self.conf.learning_rate, self.conf.momentum)
        #gvs = optimizer.compute_gradients(cost)
        #capped_gvs = [(tf.clip_by_norm(grad, .1), var) for grad, var in gvs if grad is not None]
        #train_step = optimizer.apply_gradients(capped_gvs, self.global_step)
        #self._lr = tf.Variable(0.0, trainable=False)
        
        #self.lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        #self._lr_update = tf.assign(self._lr, self._new_lr)
        #opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        with tf.variable_scope('optimizer'):
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            cost += tf.reduce_sum(reg_losses)
            with tf.control_dependencies(update_ops):
                opt = tf.train.AdagradOptimizer(learning_rate=self.conf.learning_rate,
                                                initial_accumulator_value=1.0)
                
                
                grads = opt.compute_gradients(
                                cost * 20,
                                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

                capped_gvs = [(tf.clip_by_norm(grad, 1), var) for grad, 
                              var in grads if grad is not None]
                #To check the gradients
                self.gradients = grads
                self.variables = tf.trainable_variables()
                train_step = opt.apply_gradients(capped_gvs, self.global_step)
        
        #optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
        return train_step
  
    
    def create_summaries(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("perplexity", self.perplexity)
        self.merged_summary_op = tf.summary.merge_all()
        
    def get_hidden_emb(self):
        '''Remove the beginning and ending holders'''
        cnn_hidden_emb = self.hidden_layers[:, 0, :, :]
        cnn_hidden_emb = cnn_hidden_emb[:, 1:, :]
        cnn_hidden_emb = cnn_hidden_emb[:, :-1, :]
        return cnn_hidden_emb