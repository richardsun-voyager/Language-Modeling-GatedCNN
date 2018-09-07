import numpy as np
import tensorflow as tf
import functools
from utils import positional_encoding
from nlp_blocks.nn import regularizer, spatial_dropout
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

class gated_char_cnn_model:
    '''
    A Lanugage Model Based on Gated CNN
    '''
    def __init__(self, conf, is_char_input=True, is_train=True, is_bidirectional=False):
        #Input data place holders
        #tf.reset_default_graph()
        if is_char_input:#Char as input
            shape = (conf.batch_size, conf.text_size, 50)
        else:#Words as input
            shape = (conf.batch_size, conf.text_size)
        self.create_placeholders(shape, is_bidirectional)
        self.is_train = is_train
        self.is_bidirectional = is_bidirectional
        self.is_char_input = is_char_input
        self.conf = conf
        #Get hidden layer
        hidden_layers, hidden_layers_reverse = self.build_model
        self.hidden_layer = hidden_layers[-1]
        #Concatenate hidden layers
        hidden_layers = tf.stack(hidden_layers, axis=3)
        self.hidden_layers =  tf.reduce_mean(hidden_layers, 3) 
        #For the reverse part
        if hidden_layers_reverse:
            self.hidden_layer_reverse = hidden_layers_reverse[-1]
            hidden_layers_reverse = tf.stack(hidden_layers_reverse, axis=3)
            self.hidden_layers_reverse = tf.reduce_mean(hidden_layers_reverse, 3)
        
        #Get the loss
        self.loss = self.build_loss
        self.optimizer = self.optimize
        self.cnn_hidden_emb = self.get_hidden_emb()
        self.create_summaries()
        
    def create_placeholders(self, shape, is_bidirectional=False):
        '''Create Input Shape'''
        self.X = tf.placeholder(tf.int32, shape=shape, name="input_x")
        self.y = tf.placeholder(tf.int32, shape=(None, None), name="input_y")
        if is_bidirectional:
            self.X_reverse = tf.placeholder(tf.int32, shape=shape, name="input_x_reverse")
            self.y_reverse = tf.placeholder(tf.int32, shape=(None, None), name="input_y_reverse")
        
        
    def conv_1d(self, padded_input, num_filters, filter_size, name, **kwargs):
        '''1-Dimension Convolution'''
        #Default  Kaiming initializer
        fain = num_filters * filter_size
        in_channel = padded_input.get_shape().as_list()[-1]
        kernel_shape = [filter_size, in_channel, num_filters]
        var = np.sqrt(4.0/fain)
        
#         stddev = tf.constant(var, dtype=tf.float32)
#         W_g = tf.get_variable(name='Wg'+name, 
#                               initializer=stddev, dtype=tf.float32)
#         init = tf.random_normal(kernel_shape, stddev=stddev, dtype=tf.float32)
#         W_v = tf.get_variable(initializer=init, name='Wv'+name, dtype=tf.float32)
#         #Weight normailization
#         W =  (W_g / tf.nn.l2_normalize(W_v, 0)) * W_v
#         b = tf.get_variable(initializer=tf.zeros([num_filters]), 
#                             name="b"+ name)
#         tf.add_to_collection('losses', tf.nn.l2_loss(W))
        
#         conv = tf.nn.conv1d(padded_input, W, stride=1,padding='VALID', name='conv'+name)
        
        conv = tf.layers.conv1d(padded_input, num_filters, filter_size, 
                                activation=None, 
                                kernel_initializer=tf.random_normal_initializer(0, var),
                                bias_initializer=tf.zeros_initializer(),
                                kernel_regularizer=regularizer, name='conv'+name)
        return conv
    
    def build_graph(self, inputs):
        '''
        This graph refers to dauphin's graph
        '''
        # [height, width, in_channels, out_channels]
        #Set the conv layers
        layers = '[(350, 3)] * 3'
        layers += ' + [(350, 1)] * 1'
        layers += ' + [(350, 3)] * 4'
        layers += ' + [(350, 1)] * 1'
        layers += ' + [(350, 3)] * 3'
        #layers += ' + [(1024, 4)] * 1'
        #layers += ' + [(2048, 4)] * 1'
        layers = eval(layers)
        embedding_size = self.conf.embedding_size
        #Keep the hidden layers
        hidden_layers = []
        #inputs, batch_size, context_size, emb_size
        x = inputs
        with tf.variable_scope('proj_layer'):
            #Project inputs to the dimension of next layer
            x = tf.layers.dense(x, layers[0][0], 
                                kernel_regularizer=regularizer,
                                name='projection')
        hidden_layers.append(x)     
        with tf.variable_scope('gated_conv_layers'):
            for i, params in enumerate(layers):
                #params[0]: out_dim, params[1]:kernel_width
                #Dropout
                if self.is_train:
                    x = spatial_dropout(x)
                #Give the third one a highway
                if i%3 == 2:
                    res = hidden_layers[i-1]
                else:
                    res = None
                #Gated cnn unit    
                x = self.glu(x, params[0], params[1], i, res) 
                hidden_layers.append(x) 
        return hidden_layers
    
    def glu(self, layer_input, num_filters, kernel_width, layer_name, residual=None):
        """ 
        Gated Linear Unit
        Args:
        layer_input: tensor, batch_size, text_len, emb_dim
        num_filters: output dim, int
        kernel_width: width of a convnet, int
        """
        # Pad the left side to prevent kernels from viewing future context
        #print(kernel_shape)
        left_pad = kernel_width - 1
        paddings = [[0,0], [left_pad,0],[0,0]]
        padded_input = tf.pad(layer_input, paddings, "CONSTANT")
        
        with tf.variable_scope('gated_cnn_block' + str(layer_name)):
            conv1 = self.conv_1d(padded_input, num_filters, kernel_width, name='linear')
            conv2 = self.conv_1d(padded_input, num_filters, kernel_width, name='gate')
            h = tf.multiply(conv1, tf.sigmoid(conv2, name="sig"), name='gated_cnn')

            # Preactivation residual
            if residual is not None:
                #h = (h + residual) * np.sqrt(0.5)
                h += residual
            
            #Batch normalization
            h = tf.layers.batch_normalization(h, training=self.is_train)

        return h

    def build_pipeline(self, X, name):
        '''
        Build a pipeline to compute output embeddings for each word
        Args: 
        X: input word sequence
        name: scope name
        '''
        vocab_size = self.conf.vocab_size
        embedding_size = self.conf.embedding_size
        with tf.variable_scope('layer_'+name):
            if not self.is_char_input:
                with tf.variable_scope('embedding'):
                    init = tf.random_normal([vocab_size, embedding_size], stddev=.02, dtype=tf.float32)
                    self.word_embeddings = tf.get_variable(name='word_embedding',
                                                       initializer=init)
                    input_embeddings = tf.nn.embedding_lookup(self.word_embeddings, X)
                    #Position embedding
                    pos_emb = positional_encoding(self.X.get_shape().as_list(), embedding_size)
                    input_embeddings += pos_emb
            else:
                with tf.variable_scope('char_embedding'):
                    input_embeddings = self._build_word_char_embeddings(X)
                    pos_emb = positional_encoding(self.X.get_shape().as_list()[:2], embedding_size)
                    #Position embedding
                    input_embeddings += pos_emb
                
            hidden_layers = self.build_graph(input_embeddings) 
        return hidden_layers
        
    
    @lazy_property
    def build_model(self):
        """ Setup the model after we have imported the data and know the vocabulary size """

        hidden_layers = self.build_pipeline(self.X, 'forward')
        hidden_layers_reverses = None
        if self.is_bidirectional:
            hidden_layers_reverses = self.build_pipeline(self.X_reverse, 'backward')
            
        #If only forward network
        return hidden_layers, hidden_layers_reverses
    

    @lazy_property    
    def build_loss(self):
        # Output embeddings
        kernel_shape = [self.conf.batch_size, 
                        self.conf.text_size, 350]
        last_hidden = self.hidden_layer
        #print(last_hidden)
        vocab_size = self.conf.vocab_size
        output_weights_size = kernel_shape[2]
        stddev = np.sqrt(2.0 / kernel_shape[2])
        #Output layer
        #GPU doesn't have so much memory for this output matrix
        with tf.device('/cpu:0'):
            with tf.variable_scope('softmax'):
                init = tf.random_normal_initializer(0.0, stddev)
                softmax_w = tf.get_variable(shape=[vocab_size, output_weights_size], 
                                            initializer=init, 
                                            name="output_weights", 
                                            regularizer=regularizer, 
                                            dtype=tf.float32)
                softmax_b = tf.get_variable(initializer=tf.zeros([vocab_size]), 
                                            name="output_bias")

        labels = tf.reshape(self.y, (-1, 1))
        
        h = tf.reshape(last_hidden, (-1, output_weights_size))
        #Prevent against overfitting
        if self.is_train:
            h = tf.nn.dropout(h, 0.5)
        
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
        cost += tf.losses.get_regularization_loss() * 1000
        #Update batch normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        with tf.control_dependencies(update_ops):
            opt = tf.train.AdagradOptimizer(learning_rate=self.conf.learning_rate,
                                            initial_accumulator_value=1.0)

            #opt = tf.train.MomentumOptimizer(self.conf.learning_rate, self.conf.momentum)
            # opt = tf.train.AdamOptimizer(0.01)
            grads = opt.compute_gradients(
                            cost*10,
                            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
            clip = self.conf.grad_clip
            capped_gvs = [(tf.clip_by_norm(grad, clip), var) for grad, var in grads if grad is not None]
            train_step = opt.apply_gradients(capped_gvs, self.global_step)
        
        #optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
        return train_step
    
    
    def _build_word_char_embeddings(self, inputs):
        '''
        options contains key 'char_cnn': {

        'n_characters': 262,

        # includes the start / end characters
        'max_characters_per_token': 50,

        'filters': [
            [1, 32],
            [2, 32],
            [3, 64],
            [4, 128],
            [5, 256],
            [6, 512],
            [7, 512]
        ],
        'activation': 'tanh',

        # for the character embedding
        'embedding': {'dim': 16}

        # for highway layers
        # if omitted, then no highway layers
        'n_highway': 2,
        }
        '''
        #Configuration
        options = {'char_cnn': {'activation': 'relu',
                    'embedding': {'dim': 4},
                    'filters': [[1, 4], [2, 8], [3, 16], [4, 32], [5, 96]],
                    'max_characters_per_token': 50,
                    'n_characters': 262,
                    'n_highway': 2},
                    'lstm': {'cell_clip': 3,
                    'dim': 64,
                    'n_layers': 2,
                    'proj_clip': 3,
                    'projection_dim': self.conf.embedding_size,
                    'use_skip_connections': False},
                  'n_negative_samples_batch': 8192,
                  'n_train_tokens': 768648884}
        
        projection_dim = options['lstm']['projection_dim']

        cnn_options = options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        max_chars = cnn_options['max_characters_per_token']
        char_embed_dim = cnn_options['embedding']['dim']
        n_chars = cnn_options['n_characters']
        if n_chars != 262:
            raise InvalidNumberOfCharacters(
                "Set n_characters=262 after training see the README.md"
            )
        if cnn_options['activation'] == 'tanh':
            activation = tf.nn.tanh
        elif cnn_options['activation'] == 'relu':
            activation = tf.nn.relu

        # the character embeddings
        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable(
                    "char_embed", [n_chars, char_embed_dim],
                    dtype=tf.float32,
                    initializer=tf.random_uniform_initializer(-1.0, 1.0)
            )
            # shape (batch_size, unroll_steps, max_chars, embed_dim)
            char_embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                    inputs)

        # the convolutions
        def make_convolutions(inp):
            with tf.variable_scope('CNN') as scope:
                convolutions = []
                for i, (width, num) in enumerate(filters):
                    if cnn_options['activation'] == 'relu':
                        # He initialization for ReLU activation
                        # with char embeddings init between -1 and 1
                        #w_init = tf.random_normal_initializer(
                        #    mean=0.0,
                        #    stddev=np.sqrt(2.0 / (width * char_embed_dim))
                        #)

                        # Kim et al 2015, +/- 0.05
                        w_init = tf.random_uniform_initializer(
                            minval=-0.05, maxval=0.05)
                    elif cnn_options['activation'] == 'tanh':
                        # glorot init
                        w_init = tf.random_normal_initializer(
                            mean=0.0,
                            stddev=np.sqrt(1.0 / (width * char_embed_dim))
                        )
                    w = tf.get_variable(
                        "W_cnn_%s" % i,
                        [1, width, char_embed_dim, num],
                        initializer=w_init,
                        dtype=tf.float32)
                    b = tf.get_variable(
                        "b_cnn_%s" % i, [num], dtype=tf.float32,
                        initializer=tf.constant_initializer(0.0))

                    conv = tf.nn.conv2d(
                            inp, w,
                            strides=[1, 1, 1, 1],
                            padding="VALID") + b
                    # now max pool
                    conv = tf.nn.max_pool(
                            conv, [1, 1, max_chars-width+1, 1],
                            [1, 1, 1, 1], 'VALID')

                    # activation
                    conv = activation(conv)
                    conv = tf.squeeze(conv, axis=2)

                    convolutions.append(conv)

            return tf.concat(convolutions, 2)

        embedding = make_convolutions(char_embedding)
        #print(embedding)

        # for highway and projection layers
        n_highway = cnn_options.get('n_highway')
        use_highway = n_highway is not None and n_highway > 0
        use_proj = n_filters != projection_dim

        if use_highway or use_proj:
            #   reshape from (batch_size, n_tokens, dim) to (-1, dim)
            batch_size_n_tokens = tf.shape(embedding)[0:2]
            embedding = tf.reshape(embedding, [-1, n_filters])

        # set up weights for projection
        if use_proj:
            assert n_filters > projection_dim#???why is it?
            with tf.variable_scope('CNN_proj') as scope:
                    W_proj_cnn = tf.get_variable(
                        "W_proj", [n_filters, projection_dim],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / n_filters)),
                        dtype=tf.float32)
                    b_proj_cnn = tf.get_variable(
                        "b_proj", [projection_dim],
                        initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)

        # apply highways layers
        def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
            carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
            transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
            return carry_gate * transform_gate + (1.0 - carry_gate) * x

        if use_highway:
            highway_dim = n_filters

            for i in range(n_highway):
                with tf.variable_scope('CNN_high_%s' % i) as scope:
                    W_carry = tf.get_variable(
                        'W_carry', [highway_dim, highway_dim],
                        # glorit init
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=tf.float32)
                    b_carry = tf.get_variable(
                        'b_carry', [highway_dim],
                        initializer=tf.constant_initializer(-2.0),
                        dtype=tf.float32)
                    W_transform = tf.get_variable(
                        'W_transform', [highway_dim, highway_dim],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=tf.float32)
                    b_transform = tf.get_variable(
                        'b_transform', [highway_dim],
                        initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)

                embedding = high(embedding, W_carry, b_carry,
                                 W_transform, b_transform)

        # finally project down if needed
        if use_proj:
            embedding = tf.matmul(embedding, W_proj_cnn) + b_proj_cnn

        # reshape back to (batch_size, tokens, dim)
        if use_highway or use_proj:
            shp = tf.concat([batch_size_n_tokens, [projection_dim]], axis=0)
            embedding = tf.reshape(embedding, shp)

        # at last assign attributes for remainder of the model
        #self.embedding = embedding
        return embedding
  
    
    def create_summaries(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("perplexity", self.perplexity)
        self.merged_summary_op = tf.summary.merge_all()
        
    def get_hidden_emb(self):
        '''Remove the beginning and ending holders'''
        cnn_hidden_emb = self.hidden_layers#[:, 0, :, :]
        cnn_hidden_emb = cnn_hidden_emb[:, 1:, :]
        cnn_hidden_emb = cnn_hidden_emb[:, :-1, :]
        return cnn_hidden_emb
    
    
    

# class gated_char_cnn_model2:
#     '''
#     A Lanugage Model Based on Gated CNN
#     '''
#     def __init__(self, conf, vocab_mapping=None, is_train=True, is_char_input=True):
#         #Input data place holders
#         tf.reset_default_graph()
#         if is_char_input:
#             self.X = tf.placeholder(tf.int32, shape=(None, 
#                                                  None, 50), name="input_x")
#         else:
#             self.X = tf.placeholder(tf.int32, shape=(None, 
#                                                  None), name="input_x")
        
#         self.y = tf.placeholder(tf.int32, shape=(None, 
#                                                  None), name="input_y")
#         self.is_train = is_train
#         self.is_char_input = is_char_input
#         self.vocab_mapping = vocab_mapping
#         self.conf = conf
#         self.hidden_layer = self.build_model
#         self.loss = self.build_loss
#         self.optimizer = self.optimize
#         self.create_summaries()
        
    
#     def glu(self, kernel_shape, layer_input, layer_name, residual=None):
#         """ Gated Linear Unit """
#         # Pad the left side to prevent kernels from viewing future context
#         #print(kernel_shape)
#         kernel_width = kernel_shape[1]
#         #print(kernel_width)
#         left_pad = kernel_width - 1
#         paddings = [[0,0],[0,0],[left_pad,0],[0,0]]
#         padded_input = tf.pad(layer_input, paddings, "CONSTANT")

#         # Kaiming intialization
#         stddev = tf.constant(np.sqrt(2.0 / (kernel_shape[1] * kernel_shape[2])), dtype=tf.float32)
#         # First conv layer
#         W_g = tf.get_variable(name='Wg'+str(layer_name), 
#                               initializer=stddev, dtype=tf.float32)
#         init = tf.random_normal(kernel_shape, stddev=stddev, dtype=tf.float32)
#         W_v = tf.get_variable(initializer=init, name='Wv'+str(layer_name), dtype=tf.float32)
#         W =  (W_g / tf.nn.l2_normalize(W_v, 0)) * W_v
#         b = tf.get_variable(initializer=tf.zeros([kernel_shape[2] * kernel_shape[3]]), 
#                             name="b%s" % layer_name)
#         conv1 = tf.nn.depthwise_conv2d(
#             padded_input,
#             W,
#             strides=[1, 1, 1, 1],
#             padding="VALID",
#             name="conv1")
#         conv1 = tf.nn.bias_add(conv1, b)

#         # Second gating sigmoid layer
#         V_g = tf.get_variable(name="Vg"+str(layer_name), initializer=stddev, dtype=tf.float32)
#         V_v = tf.get_variable(initializer=tf.random_normal(kernel_shape, stddev=stddev), 
#                               name="Vv"+str(layer_name))
#         V = (V_g / tf.nn.l2_normalize(V_v, 0)) * V_v
#         c = tf.get_variable(initializer=tf.zeros([kernel_shape[2] * kernel_shape[3]]), 
#                             name="c%s" % layer_name, dtype=tf.float32)
#         conv2 = tf.nn.depthwise_conv2d(
#             padded_input,
#             V,
#             strides=[1, 1, 1, 1],
#             padding="VALID",
#             name="conv2")
#         conv2 = tf.nn.bias_add(conv2, c)

#         # Preactivation residual
#         if residual is not None:
#             conv1 = tf.add(conv1, residual)
#             conv2 = tf.add(conv2, residual)

#         h = tf.multiply(conv1, tf.sigmoid(conv2, name="sig"))

#         return h
    
#     @lazy_property
#     def build_model(self):
#         """ Setup the model after we have imported the data and know the vocabulary size """
        
#         #Configuration
#         options = {'char_cnn': {'activation': 'relu',
#                     'embedding': {'dim': 4},
#                     'filters': [[1, 4], [2, 8], [3, 16], [4, 32], [5, 96]],
#                     'max_characters_per_token': 50,
#                     'n_characters': 262,
#                     'n_highway': 2},
#                     'lstm': {'cell_clip': 3,
#                     'dim': 64,
#                     'n_layers': 2,
#                     'proj_clip': 3,
#                     'projection_dim': self.conf.embedding_size,
#                     'use_skip_connections': False},
#                   'n_negative_samples_batch': 8192,
#                   'n_train_tokens': 768648884}
#         embedding_size = self.conf.embedding_size
#         if not self.is_char_input:
#             vocab_size = self.conf.vocab_size
#             init = tf.random_normal([vocab_size, embedding_size], stddev=.01, dtype=tf.float32)
#             with tf.variable_scope('embedding'):
#                 self.word_embeddings = tf.get_variable(name='word_embedding',
#                                                    initializer=init)
#                 input_embeddings = tf.nn.embedding_lookup(self.word_embeddings, self.X)
#             input_embeddings_expanded = tf.expand_dims(input_embeddings, 1)
#         else:
#             #embed = self.create_embeddings(self.X, conf)
#             input_embeddings = self._build_word_char_embeddings(self.X, options)
#             #print(input_embeddings)
#             input_embeddings_expanded = tf.expand_dims(input_embeddings, 1)


#         # Remove the last element, as the next word is in a new sequence and we do not predict it
#         # Reshape labels and hidden layer as we're only interested in scoring at the word level
#         last_hidden = self.set_graph(input_embeddings_expanded)
#         return last_hidden
    
    
#     def set_graph(self, input_embeddings_expanded):
#         # [height, width, in_channels, out_channels]
#         embedding_size = self.conf.embedding_size
#         with tf.variable_scope('conv_layers'):
#             kernel_shape = [1, 3, embedding_size, 1]
#             h0 = self.glu(kernel_shape, input_embeddings_expanded, 0)
#             h1 = self.glu(kernel_shape, h0, 1)
#             h2 = self.glu(kernel_shape, h1, 2)
#             h3 = self.glu(kernel_shape, h2, 3)
#             h4 = self.glu(kernel_shape, h3, 4, h0)

#            #  For larger models with output projections:
#             kernel_shape = [1, 3, 128, 2]
#             h4a = self.glu(kernel_shape, h4, '14a')
            

#             kernel_shape = [1, 3, 256, 1]
#             h5 = self.glu(kernel_shape, h4a, 5)
#             h6 = self.glu(kernel_shape, h5, 6)
#             h7 = self.glu(kernel_shape, h6, 7)
#             h8 = self.glu(kernel_shape, h7, 8)
#             h9 = self.glu(kernel_shape, h8, 9, h4a)
#         return h9
        
#     @lazy_property    
#     def build_loss(self):
#         # Output embeddings
#         kernel_shape = [1, 3, 128, 2]
#         last_hidden = self.hidden_layer
#         vocab_size = self.conf.vocab_size
#         output_weights_size = kernel_shape[2] * kernel_shape[3]
#         stddev = np.sqrt(2.0 / (kernel_shape[1] * kernel_shape[2]))
#         init = tf.random_normal([vocab_size, output_weights_size], stddev=stddev, dtype=tf.float32)
#         softmax_w = tf.get_variable(initializer=init, name="output_weights")
#         softmax_b = tf.get_variable(initializer=tf.zeros([vocab_size]), name="output_bias")
#         #last_hidden = tf.slice(last_hidden, [0, 0, 0, 0], [-1, -1, sequence_length-1, -1])
#         #last_hidden = tf.squeeze(last_hidden)
#         #last_hidden = tf.reshape(last_hidden, [minibatch_size * (sequence_length - 1), output_weights_size])
#         labels = tf.reshape(self.y, (-1, 1))
#         h = tf.reshape(last_hidden, (-1, output_weights_size))

#        # Todo: sampled softmax for larger vocabularies
#        # losses = tf.nn.sampled_softmax_loss(output_weights, output_bias, last_hidden, labels, candidates, vocab_size, num_true=1, partition_strategy='mod', name='ssl')

#         losses = tf.nn.sampled_softmax_loss(
#                                    softmax_w, softmax_b,
#                                    labels, h,
#                                    8192,
#                                    self.conf.vocab_size,
#                                    num_true=1)
#         loss = tf.reduce_mean(losses)
#         self.perplexity = tf.exp(loss)

#         return loss
    
#     @lazy_property
#     def optimize(self):
#         cost = self.loss
#         #trainer = tf.train.MomentumOptimizer(self.conf.learning_rate, self.conf.momentum)
#         #gradients = trainer.compute_gradients(cost)
#         #clipped_gradients = [(tf.clip_by_value(_[0], -self.conf.grad_clip, self.conf.grad_clip), 
#                               #_[1]) for _ in gradients]
#         #optimizer = trainer.apply_gradients(clipped_gradients)
        
#         self.global_step = tf.Variable(0, name='global_step', trainable=False)
#         #learning_rate = tf.train.exponential_decay(.5, self.global_step, 1, 0.9, staircase=False)
#         #optimizer = tf.train.MomentumOptimizer(self.conf.learning_rate, self.conf.momentum)
#         #gvs = optimizer.compute_gradients(cost)
#         #capped_gvs = [(tf.clip_by_norm(grad, .1), var) for grad, var in gvs if grad is not None]
#         #train_step = optimizer.apply_gradients(capped_gvs, self.global_step)
        
#         lr = self.conf.learning_rate
#         opt = tf.train.AdagradOptimizer(learning_rate=lr,
#                                         initial_accumulator_value=1.0)
#         grads = opt.compute_gradients(
#                         cost * 20,
#                         aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
#                     )
#         capped_gvs = [(tf.clip_by_norm(grad, 1), var) for grad, var in grads if grad is not None]
#         train_step = opt.apply_gradients(capped_gvs, self.global_step)
        
        
#         #optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
#         return train_step
    
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
    
#     def create_summaries(self):
#         tf.summary.scalar("loss", self.loss)
#         tf.summary.scalar("perplexity", self.perplexity)
#         self.merged_summary_op = tf.summary.merge_all()