import numpy as np
import tensorflow as tf
import functools
from utils import positional_encoding
from tensorflow.contrib.layers.python.layers import encoders
def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

class gated_cnn_model:
    '''
    A Lanugage Model Based on Gated CNN
    '''
    def __init__(self, conf,  is_train=True, is_bidirectional=False):
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
        self.hidden_layer = hidden_layers[-1]
        #Concatenate hidden layers
        self.hidden_layers =  tf.concat(hidden_layers, 3) 
        #For the reverse part
        if hidden_layers_reverse:
            self.hidden_layer_reverse = hidden_layers_reverse[-1]
            self.hidden_layers_reverse = tf.concat(hidden_layers_reverse, 3) 
        
        #Get the loss
        self.loss = self.build_loss
        self.optimizer = self.optimize
        self.cnn_hidden_emb = self.get_hidden_emb()
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
        
        #l2 Regularization
        tf.add_to_collection('losses', tf.nn.l2_loss(W))
        
        conv = tf.nn.depthwise_conv2d(
            padded_input,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv"+name)
        conv = tf.nn.bias_add(conv, b)
        return conv
        
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
            #Position embedding
            pos_emb = positional_encoding(self.X, embedding_size)
            input_embeddings += pos_emb
            #if self.is_train:
                #input_embeddings = tf.nn.dropout(input_embeddings, 0.8)
            
        #Note for the convolutional layer, we need expand the dimension to 4
        input_embeddings_expanded = tf.expand_dims(input_embeddings, 1)
        with tf.variable_scope('forward_layer'):
            hidden_layers = self.build_graph(input_embeddings_expanded)
            
        #Make it bidirectional, just like that in biLSTM
        if self.is_bidirectional:
            input_embeddings_reverse = tf.nn.embedding_lookup(self.word_embeddings, 
                                                              self.X_reverse)
            input_embeddings_expanded_reverse = tf.expand_dims(input_embeddings_reverse, 1)
            with tf.variable_scope('backward_layer'):
                hidden_layers_reverse = self.build_graph(input_embeddings_expanded_reverse)
                #last_hidden_reverse = hidden_layers_reverse[-1]
                return hidden_layers, hidden_layers_reverse
        #If only forward network
        return hidden_layers, None
    
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
            h = tf.multiply(conv1, tf.sigmoid(conv2, name="sig"), name='gated_cnn')

            # Preactivation residual
            if residual is not None:
                h += residual
                #conv1 = tf.add(conv1, residual)
                #conv2 = tf.add(conv2, residual)
            #Gated CNN    
            
            #Batch normalization
            h = tf.layers.batch_normalization(h, training=self.is_train)

        return h
    
    def build_graph(self, input_embeddings_expanded):
        # [height, width, in_channels, out_channels]
        embedding_size = self.conf.embedding_size
        #Keep the hidden layers
        hidden_layers = []
        with tf.variable_scope('conv_layers'):
            hidden_layers.append(input_embeddings_expanded)
            kernel_shape = [1, 3, embedding_size, 5]
            h0 = self.glu(kernel_shape, input_embeddings_expanded, 0)
            kernel_shape = [1, 3, embedding_size*5, 1]
            h1 = self.glu(kernel_shape, h0, 1)
            h2 = self.glu(kernel_shape, h1, 2)
            h3 = self.glu(kernel_shape, h2, 3, h0)
            #batch_size, 1, word_count, 256
            h4 = self.glu(kernel_shape, h3, 4)
            hidden_layers.append(h4)
            

            #  For larger models with output projections:
            #kernel_shape = [1, 3, 256, 1]
            h4a = self.glu(kernel_shape, h4, '14a', h2)
            

            #kernel_shape = [1, 3, 256, 1]
            h5 = self.glu(kernel_shape, h4a, 5)
            h6 = self.glu(kernel_shape, h5, 6, h4)
            h7 = self.glu(kernel_shape, h6, 7)
            h8 = self.glu(kernel_shape, h7, 8, h5)
            h9 = self.glu(kernel_shape, h8, 9)
            hidden_layers.append(h9)
        return hidden_layers
        
    @lazy_property    
    def build_loss(self):
        # Output embeddings
        kernel_shape = [1, 3, 128, 5]
        last_hidden = self.hidden_layer
        vocab_size = self.conf.vocab_size
        output_weights_size = kernel_shape[2] * kernel_shape[3]
        stddev = np.sqrt(2.0 / (kernel_shape[2] * kernel_shape[3]))
        #Output layer
        with tf.variable_scope('softmax'):
            init = tf.random_normal([vocab_size, output_weights_size], 
                                stddev=stddev, dtype=tf.float32)
            softmax_w = tf.get_variable(initializer=init, name="output_weights")
            softmax_b = tf.get_variable(initializer=tf.zeros([vocab_size]), name="output_bias")
            
            tf.add_to_collection('losses', tf.nn.l2_loss(softmax_w))

        labels = tf.reshape(self.y, (-1, 1))
        
        h = tf.reshape(last_hidden, (-1, output_weights_size))
        #Prevent against overfitting
        if self.is_train:
            h = tf.nn.dropout(h, 0.8)
        
        loss_forward = tf.nn.sampled_softmax_loss(
                                   softmax_w, softmax_b,
                                   labels, h,
                                   8192,
                                   self.conf.vocab_size,
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
                                       8192,
                                       self.conf.vocab_size,
                                       num_true=1)
        
            loss += tf.reduce_mean(loss_backward)
        self.perplexity = tf.exp(loss)

        return loss
    
    @lazy_property
    def optimize(self):
        cost = self.loss
        #Take regularization intp account
        cost += tf.reduce_sum(tf.get_collection('losses')) * 0.0001
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        #optimizer = tf.train.MomentumOptimizer(self.conf.learning_rate, self.conf.momentum)
        #gvs = optimizer.compute_gradients(cost)
        #capped_gvs = [(tf.clip_by_norm(grad, .1), var) for grad, var in gvs if grad is not None]
        #train_step = optimizer.apply_gradients(capped_gvs, self.global_step)
        #self._lr = tf.Variable(0.0, trainable=False)
        with tf.control_dependencies(update_ops):
            #opt = tf.train.AdagradOptimizer(learning_rate=self.conf.learning_rate,
                                            #initial_accumulator_value=1.0)

            #opt = tf.train.MomentumOptimizer(self.conf.learning_rate, 
                                             #self.conf.momentum, use_nesterov=True)
            opt = tf.train.AdamOptimizer(self.conf.learning_rate)
            grads = opt.compute_gradients(
                            cost*10 ,
                            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
            clip = self.conf.grad_clip
            capped_gvs = [(tf.clip_by_norm(grad, clip), var) for grad, var in grads if grad is not None]
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
    
        

 
        
# class gated_char_cnn_model(object):
#     '''
#     The input are word characters, to make full use of subword information
#     '''
#     def __init__(self, conf):
#         tf.reset_default_graph()
#         #Configuration
#         options = {'char_cnn': {'activation': 'relu',
#                     'embedding': {'dim': 4},
#                     'filters': [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
#                     'max_characters_per_token': 50,
#                     'n_characters': 262,
#                     'n_highway': 2},
#                     'lstm': {'cell_clip': 3,
#                     'dim': 64,
#                     'n_layers': 2,
#                     'proj_clip': 3,
#                     'projection_dim': conf.embedding_size,
#                     'use_skip_connections': False},
#                   'n_negative_samples_batch': 8192,
#                   'n_train_tokens': 768648884}
        
#         #Input sentence characters, batch_size * word_num * char_num   
#         self.X = tf.placeholder(shape=[None, None, 50], dtype=tf.int32, name="X")
#         #Label words
#         self.y = tf.placeholder(shape=[None, None], dtype=tf.int32, name="y")
#         #Create word Embeddings for input sentences
#         #embed = self.create_embeddings(self.X, conf)
#         embed = self._build_word_char_embeddings(self.X, options)
#         #print(embed)
#         embed = tf.expand_dims(embed, 1)
#         #print(embed)

#         #Initialize the input of each layer
#         self.internal_states = []
        
#         h, res_input = embed, embed
#         for i in np.arange(conf.num_layers):
#             self.internal_states.append(h)
#             shape = (1, conf.filter_h, conf.filter_w, 1)
#             h = self.glu(h, shape, conf, i)
#             if i % conf.block_size == 2:
#                 h += res_input
#                 res_input = h
        

#         #Flatten the output, (batch_size*max_len) * emb_size
#         #print(h)
#         self.out_layer = tf.squeeze(h, 1)
#         #h = tf.reshape(h, (-1, conf.embedding_size))
        
#         h = tf.reshape(h, (-1, conf.embedding_size))

#         y_shape = self.y.get_shape().as_list()
#         #Flatten the label, (batch_size*max_len) * 1
#         self.y = tf.reshape(self.y, (-1, 1))
#         #print(self.y)
#         #Nce loss for the softmax
#         softmax_w = tf.get_variable("softmax_w", [conf.vocab_size, conf.embedding_size], tf.float32, 
#                                     tf.random_normal_initializer(0.0, 0.1))
#         softmax_b = tf.get_variable("softmax_b", [conf.vocab_size], tf.float32, tf.constant_initializer(1.0))

#         #Preferance: NCE Loss, heirarchial softmax, adaptive softmax
#         #Note tf.nn.nce_loss has changed in new versions
#         #self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=softmax_w, biases=softmax_b, 
#                                                   #inputs=h, labels=self.y, 
#                                                   #num_sampled=conf.num_sampled,
#                                                   #num_classes=conf.vocab_size))
        
#         self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
#                                    softmax_w, softmax_b,
#                                    self.y, h,
#                                    options['n_negative_samples_batch'],
#                                    conf.vocab_size,
#                                    num_true=1))
        
#         #Optimizer
#         #trainer = tf.train.MomentumOptimizer(conf.learning_rate, conf.momentum)
#         #gradients = trainer.compute_gradients(self.loss)
#         #clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
#         #self.optimizer = trainer.apply_gradients(clipped_gradients)
#         self.perplexity = tf.exp(self.loss)
#         self.optimizer = tf.train.AdamOptimizer(conf.learning_rate).minimize(self.loss)

#         self.create_summaries()
        
#     def glu(self, inputs, shape, conf, layer_index):
#         #Initialize kernel size
#         height = conf.filter_h
#         width = conf.filter_w
#         #h, res_input = embed, embed
#         kernel_width = shape[1]
#         left_pad = kernel_width - 1
#         paddings = [[0,0],[0,0],[left_pad,0],[0,0]]
#         padded_input = tf.pad(inputs, paddings, "CONSTANT")

#         with tf.variable_scope("layer_%d"%layer_index):
#             #Linear layer
#             conv_w = self.conv_op(padded_input, shape, "linear")
#             #conv_w = tf.layers.conv2d(h, filter_size,
#                                    #kernel_size=(height, width),
#                                    #strides=(1, width), padding='same',
#                                     #name='linear')
#             #Gate layer
#             conv_v = self.conv_op(padded_input, shape, "gated")
#             #conv_v = tf.layers.conv2d(h, filter_size,
#                                    #kernel_size=(height, width),
#                                    #strides=(1, width), padding='same',
#                                     #name='gate')
#             #Elementwise multiplication
#             #batch_size, max_len, 1, filter_size
#             h = conv_w * tf.sigmoid(conv_v)
#             #print(h)
#         return h
    
#     def _build_word_char_embeddings(self, inputs, options):
#         '''
#         options contains key 'char_cnn': {

#         'n_characters': 262,

#         # includes the start / end characters
#         'max_characters_per_token': 50,

#         'filters': [
#             [1, 32],
#             [2, 32],
#             [3, 64],
#             [4, 128],
#             [5, 256],
#             [6, 512],
#             [7, 512]
#         ],
#         'activation': 'tanh',

#         # for the character embedding
#         'embedding': {'dim': 16}

#         # for highway layers
#         # if omitted, then no highway layers
#         'n_highway': 2,
#         }
#         '''
        
#         projection_dim = options['lstm']['projection_dim']

#         cnn_options = options['char_cnn']
#         filters = cnn_options['filters']
#         n_filters = sum(f[1] for f in filters)
#         max_chars = cnn_options['max_characters_per_token']
#         char_embed_dim = cnn_options['embedding']['dim']
#         n_chars = cnn_options['n_characters']
#         if n_chars != 262:
#             raise InvalidNumberOfCharacters(
#                 "Set n_characters=262 after training see the README.md"
#             )
#         if cnn_options['activation'] == 'tanh':
#             activation = tf.nn.tanh
#         elif cnn_options['activation'] == 'relu':
#             activation = tf.nn.relu

#         # the character embeddings
#         with tf.device("/cpu:0"):
#             self.embedding_weights = tf.get_variable(
#                     "char_embed", [n_chars, char_embed_dim],
#                     dtype=tf.float32,
#                     initializer=tf.random_uniform_initializer(-1.0, 1.0)
#             )
#             # shape (batch_size, unroll_steps, max_chars, embed_dim)
#             char_embedding = tf.nn.embedding_lookup(self.embedding_weights,
#                                                     inputs)

#         # the convolutions
#         def make_convolutions(inp):
#             with tf.variable_scope('CNN') as scope:
#                 convolutions = []
#                 for i, (width, num) in enumerate(filters):
#                     if cnn_options['activation'] == 'relu':
#                         # He initialization for ReLU activation
#                         # with char embeddings init between -1 and 1
#                         #w_init = tf.random_normal_initializer(
#                         #    mean=0.0,
#                         #    stddev=np.sqrt(2.0 / (width * char_embed_dim))
#                         #)

#                         # Kim et al 2015, +/- 0.05
#                         w_init = tf.random_uniform_initializer(
#                             minval=-0.05, maxval=0.05)
#                     elif cnn_options['activation'] == 'tanh':
#                         # glorot init
#                         w_init = tf.random_normal_initializer(
#                             mean=0.0,
#                             stddev=np.sqrt(1.0 / (width * char_embed_dim))
#                         )
#                     w = tf.get_variable(
#                         "W_cnn_%s" % i,
#                         [1, width, char_embed_dim, num],
#                         initializer=w_init,
#                         dtype=tf.float32)
#                     b = tf.get_variable(
#                         "b_cnn_%s" % i, [num], dtype=tf.float32,
#                         initializer=tf.constant_initializer(0.0))

#                     conv = tf.nn.conv2d(
#                             inp, w,
#                             strides=[1, 1, 1, 1],
#                             padding="VALID") + b
#                     # now max pool
#                     conv = tf.nn.max_pool(
#                             conv, [1, 1, max_chars-width+1, 1],
#                             [1, 1, 1, 1], 'VALID')

#                     # activation
#                     conv = activation(conv)
#                     conv = tf.squeeze(conv, axis=2)

#                     convolutions.append(conv)

#             return tf.concat(convolutions, 2)

#         embedding = make_convolutions(char_embedding)
#         #print(embedding)

#         # for highway and projection layers
#         n_highway = cnn_options.get('n_highway')
#         use_highway = n_highway is not None and n_highway > 0
#         use_proj = n_filters != projection_dim

#         if use_highway or use_proj:
#             #   reshape from (batch_size, n_tokens, dim) to (-1, dim)
#             batch_size_n_tokens = tf.shape(embedding)[0:2]
#             embedding = tf.reshape(embedding, [-1, n_filters])

#         # set up weights for projection
#         if use_proj:
#             assert n_filters > projection_dim#???why is it?
#             with tf.variable_scope('CNN_proj') as scope:
#                     W_proj_cnn = tf.get_variable(
#                         "W_proj", [n_filters, projection_dim],
#                         initializer=tf.random_normal_initializer(
#                             mean=0.0, stddev=np.sqrt(1.0 / n_filters)),
#                         dtype=tf.float32)
#                     b_proj_cnn = tf.get_variable(
#                         "b_proj", [projection_dim],
#                         initializer=tf.constant_initializer(0.0),
#                         dtype=tf.float32)

#         # apply highways layers
#         def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
#             carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
#             transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
#             return carry_gate * transform_gate + (1.0 - carry_gate) * x

#         if use_highway:
#             highway_dim = n_filters

#             for i in range(n_highway):
#                 with tf.variable_scope('CNN_high_%s' % i) as scope:
#                     W_carry = tf.get_variable(
#                         'W_carry', [highway_dim, highway_dim],
#                         # glorit init
#                         initializer=tf.random_normal_initializer(
#                             mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
#                         dtype=tf.float32)
#                     b_carry = tf.get_variable(
#                         'b_carry', [highway_dim],
#                         initializer=tf.constant_initializer(-2.0),
#                         dtype=tf.float32)
#                     W_transform = tf.get_variable(
#                         'W_transform', [highway_dim, highway_dim],
#                         initializer=tf.random_normal_initializer(
#                             mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
#                         dtype=tf.float32)
#                     b_transform = tf.get_variable(
#                         'b_transform', [highway_dim],
#                         initializer=tf.constant_initializer(0.0),
#                         dtype=tf.float32)

#                 embedding = high(embedding, W_carry, b_carry,
#                                  W_transform, b_transform)

#         # finally project down if needed
#         if use_proj:
#             embedding = tf.matmul(embedding, W_proj_cnn) + b_proj_cnn

#         # reshape back to (batch_size, tokens, dim)
#         if use_highway or use_proj:
#             shp = tf.concat([batch_size_n_tokens, [projection_dim]], axis=0)
#             embedding = tf.reshape(embedding, shp)

#         # at last assign attributes for remainder of the model
#         #self.embedding = embedding
#         return embedding

#     def create_embeddings(self, X, conf):
#         #Create initial embeddings
#         embeddings = tf.get_variable("embeds",(conf.vocab_size, conf.embedding_size), tf.float32, tf.random_uniform_initializer(-1.0,1.0))
#         #Find embeddings for X
#         embed = tf.nn.embedding_lookup(embeddings, X)
#         #The original sentence was padding with k-1 zero in the beginning
#         embed = tf.expand_dims(embed, 1)
#         return embed


#     def conv_op(self, fan_in, shape, name):
#         '''
#         Depthwise convolution layer
#         '''
#         W = tf.get_variable("%s_W"%name, shape, tf.float32, tf.random_normal_initializer(0.0, 0.1))
#         b = tf.get_variable("%s_b"%name, shape[-1], tf.float32, tf.constant_initializer(1.0))
#         return tf.add(tf.nn.depthwise_conv2d(fan_in, W, strides=[1,1,1,1], padding='VALID'), b)
    
#     def create_summaries(self):
#         tf.summary.scalar("loss", self.loss)
#         tf.summary.scalar("perplexity", self.perplexity)
#         self.merged_summary_op = tf.summary.merge_all()
