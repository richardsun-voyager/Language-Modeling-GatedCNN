import numpy as np
import tensorflow as tf
import functools
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
    def __init__(self, conf, vocab_mapping=None, is_train=True):
        #Input data place holders
        tf.reset_default_graph()
        self.X = tf.placeholder(tf.int32, shape=(None, 
                                                 None), name="input_x")
        
        self.y = tf.placeholder(tf.int32, shape=(None, 
                                                 None), name="input_y")
        self.is_train = is_train
        self.vocab_mapping = vocab_mapping
        self.conf = conf
        self.hidden_layer = self.build_model
        self.loss = self.build_loss
        self.optimizer = self.optimize
        self.create_summaries()
        
    
    def glu(self, kernel_shape, layer_input, layer_name, residual=None):
        """ Gated Linear Unit """
        # Pad the left side to prevent kernels from viewing future context
        #print(kernel_shape)
        kernel_width = kernel_shape[1]
        #print(kernel_width)
        left_pad = kernel_width - 1
        paddings = [[0,0],[0,0],[left_pad,0],[0,0]]
        padded_input = tf.pad(layer_input, paddings, "CONSTANT")

        # Kaiming intialization
        stddev = tf.constant(np.sqrt(2.0 / (kernel_shape[1] * kernel_shape[2])), dtype=tf.float32)
        # First conv layer
        W_g = tf.get_variable(name='Wg'+str(layer_name), 
                              initializer=stddev, dtype=tf.float32)
        init = tf.random_normal(kernel_shape, stddev=stddev, dtype=tf.float32)
        W_v = tf.get_variable(initializer=init, name='Wv'+str(layer_name), dtype=tf.float32)
        W =  (W_g / tf.nn.l2_normalize(W_v, 0)) * W_v
        b = tf.get_variable(initializer=tf.zeros([kernel_shape[2] * kernel_shape[3]]), 
                            name="b%s" % layer_name)
        conv1 = tf.nn.depthwise_conv2d(
            padded_input,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv1")
        conv1 = tf.nn.bias_add(conv1, b)

        # Second gating sigmoid layer
        V_g = tf.get_variable(name="Vg"+str(layer_name), initializer=stddev, dtype=tf.float32)
        V_v = tf.get_variable(initializer=tf.random_normal(kernel_shape, stddev=stddev), 
                              name="Vv"+str(layer_name))
        V = (V_g / tf.nn.l2_normalize(V_v, 0)) * V_v
        c = tf.get_variable(initializer=tf.zeros([kernel_shape[2] * kernel_shape[3]]), 
                            name="c%s" % layer_name, dtype=tf.float32)
        conv2 = tf.nn.depthwise_conv2d(
            padded_input,
            V,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv2")
        conv2 = tf.nn.bias_add(conv2, c)

        # Preactivation residual
        if residual is not None:
            conv1 = tf.add(conv1, residual)
            conv2 = tf.add(conv2, residual)

        h = tf.multiply(conv1, tf.sigmoid(conv2, name="sig"))

        return h
    
    @lazy_property
    def build_model(self):
        """ Setup the model after we have imported the data and know the vocabulary size """

        vocab_size = self.conf.vocab_size
        embedding_size = self.conf.embedding_size
        init = tf.random_normal([vocab_size, embedding_size], stddev=.01, dtype=tf.float32)
        with tf.variable_scope('embedding'):
            self.word_embeddings = tf.get_variable(name='word_embedding',
                                               initializer=init)
            input_embeddings = tf.nn.embedding_lookup(self.word_embeddings, self.X)
        input_embeddings_expanded = tf.expand_dims(input_embeddings, 1)

        # [height, width, in_channels, out_channels]
        with tf.variable_scope('conv_layers'):
            kernel_shape = [1, 3, embedding_size, 1]
            h0 = self.glu(kernel_shape, input_embeddings_expanded, 0)
            h1 = self.glu(kernel_shape, h0, 1)
            h2 = self.glu(kernel_shape, h1, 2)
            h3 = self.glu(kernel_shape, h2, 3)
            h4 = self.glu(kernel_shape, h3, 4, h0)

           #  For larger models with output projections:
            kernel_shape = [1, 3, 128, 2]
            h4a = self.glu(kernel_shape, h4, '14a')
            

            kernel_shape = [1, 3, 256, 1]
            h5 = self.glu(kernel_shape, h4a, 5)
            h6 = self.glu(kernel_shape, h5, 6)
            h7 = self.glu(kernel_shape, h6, 7)
            h8 = self.glu(kernel_shape, h7, 8)
            h9 = self.glu(kernel_shape, h8, 9, h4a)

        # Remove the last element, as the next word is in a new sequence and we do not predict it
        # Reshape labels and hidden layer as we're only interested in scoring at the word level
        last_hidden = h9
        return last_hidden
        
    @lazy_property    
    def build_loss(self):
        # Output embeddings
        kernel_shape = [1, 3, 128, 2]
        last_hidden = self.hidden_layer
        vocab_size = self.conf.vocab_size
        output_weights_size = kernel_shape[2] * kernel_shape[3]
        stddev = np.sqrt(2.0 / (kernel_shape[1] * kernel_shape[2]))
        init = tf.random_normal([vocab_size, output_weights_size], stddev=stddev, dtype=tf.float32)
        softmax_w = tf.get_variable(initializer=init, name="output_weights")
        softmax_b = tf.get_variable(initializer=tf.zeros([vocab_size]), name="output_bias")
        #last_hidden = tf.slice(last_hidden, [0, 0, 0, 0], [-1, -1, sequence_length-1, -1])
        #last_hidden = tf.squeeze(last_hidden)
        #last_hidden = tf.reshape(last_hidden, [minibatch_size * (sequence_length - 1), output_weights_size])
        labels = tf.reshape(self.y, (-1, 1))
        h = tf.reshape(last_hidden, (-1, output_weights_size))

       # Todo: sampled softmax for larger vocabularies
       # losses = tf.nn.sampled_softmax_loss(output_weights, output_bias, last_hidden, labels, candidates, vocab_size, num_true=1, partition_strategy='mod', name='ssl')

        losses = tf.nn.sampled_softmax_loss(
                                   softmax_w, softmax_b,
                                   labels, h,
                                   8192,
                                   self.conf.vocab_size,
                                   num_true=1)
        loss = tf.reduce_mean(losses)
        self.perplexity = tf.exp(loss)

        return loss
    
    @lazy_property
    def optimize(self):
        cost = self.loss
        #trainer = tf.train.MomentumOptimizer(self.conf.learning_rate, self.conf.momentum)
        #gradients = trainer.compute_gradients(cost)
        #clipped_gradients = [(tf.clip_by_value(_[0], -self.conf.grad_clip, self.conf.grad_clip), 
                              #_[1]) for _ in gradients]
        #optimizer = trainer.apply_gradients(clipped_gradients)
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        #learning_rate = tf.train.exponential_decay(.5, self.global_step, 1, 0.9, staircase=False)
        #optimizer = tf.train.MomentumOptimizer(self.conf.learning_rate, self.conf.momentum)
        #gvs = optimizer.compute_gradients(cost)
        #capped_gvs = [(tf.clip_by_norm(grad, .1), var) for grad, var in gvs if grad is not None]
        #train_step = optimizer.apply_gradients(capped_gvs, self.global_step)
        
        lr = self.conf.learning_rate
        opt = tf.train.AdagradOptimizer(learning_rate=lr,
                                        initial_accumulator_value=1.0)
        grads = opt.compute_gradients(
                        cost * 20,
                        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                    )
        capped_gvs = [(tf.clip_by_norm(grad, 1), var) for grad, var in grads if grad is not None]
        train_step = opt.apply_gradients(capped_gvs, self.global_step)
        
        #optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
        return train_step
    
    def create_summaries(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("perplexity", self.perplexity)
        self.merged_summary_op = tf.summary.merge_all()
    
        

class GatedCNN(object):

    def __init__(self, conf):
        tf.reset_default_graph()
        #Input sentence, batch_size * word_num    
        self.X = tf.placeholder(shape=[None, conf.text_size], dtype=tf.int32, name="X")
        #Label words
        self.y = tf.placeholder(shape=[None, conf.text_size], dtype=tf.int32, name="y")
        #Create word Embeddings for input sentences
        embed = self.create_embeddings(self.X, conf)
        #Initialize the input of each layer
        self.internal_states = []
        
        h, res_input = embed, embed
        for i in np.arange(conf.num_layers):
            self.internal_states.append(h)
            shape = (1, conf.filter_h, conf.filter_w, 1)
            h = self.glu(h, shape, conf, i)
            if i % conf.block_size == 2:
                h += res_input
                res_input = h
        

        #Flatten the output, (batch_size*max_len) * emb_size
        #print(h)
        self.out_layer = tf.squeeze(h, 1)
        #h = tf.reshape(h, (-1, conf.embedding_size))
        
        h = tf.reshape(h, (-1, conf.embedding_size))
        
        #residual layer, prevent weight diminishing
        #Note residual block should avoid the last layer
        
        
        #print(h)
        y_shape = self.y.get_shape().as_list()
        #Flatten the label, (batch_size*max_len) * 1
        self.y = tf.reshape(self.y, (-1, 1))
        #print(self.y)
        #Nce loss for the softmax
        softmax_w = tf.get_variable("softmax_w", [conf.vocab_size, conf.embedding_size], tf.float32, 
                                    tf.random_normal_initializer(0.0, 0.1))
        softmax_b = tf.get_variable("softmax_b", [conf.vocab_size], tf.float32, tf.constant_initializer(1.0))
        #softmax_w = tf.get_variable("softmax_w", [conf.embedding_size, conf.vocab_size], tf.float32, 
                                    #tf.random_normal_initializer(0.0, 0.1))
        
        #logits = tf.matmul(h, softmax_w) + softmax_b
        
        #output = tf.nn.softmax(tf.matmul(h, softmax_w) + softmax_b)
        #labels = tf.one_hot(self.y, conf.vocab_size)
        
        #self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        #cross_entropy = -tf.reduce_mean(labels*tf.log(output))
        #self.loss = cross_entropy

        #Preferance: NCE Loss, heirarchial softmax, adaptive softmax
        #Note tf.nn.nce_loss has changed in new versions
        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=softmax_w, biases=softmax_b, 
                                                  inputs=h, labels=self.y, 
                                                  num_sampled=conf.num_sampled,
                                                  num_classes=conf.vocab_size))
        
        #Optimizer
        trainer = tf.train.MomentumOptimizer(conf.learning_rate, conf.momentum)
        gradients = trainer.compute_gradients(self.loss)
        clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
        self.optimizer = trainer.apply_gradients(clipped_gradients)
        self.perplexity = tf.exp(self.loss)
        #self.optimizer = tf.train.AdamOptimizer(conf.learning_rate).minimize(self.loss)

        self.create_summaries()
        
    def glu(self, inputs, shape, conf, layer_index):
        #Initialize kernel size
        height = conf.filter_h
        width = conf.filter_w
        #h, res_input = embed, embed
        kernel_width = shape[1]
        left_pad = kernel_width - 1
        paddings = [[0,0],[0,0],[left_pad,0],[0,0]]
        padded_input = tf.pad(inputs, paddings, "CONSTANT")

        with tf.variable_scope("layer_%d"%layer_index):
            #Linear layer
            conv_w = self.conv_op(padded_input, shape, "linear")
            #conv_w = tf.layers.conv2d(h, filter_size,
                                   #kernel_size=(height, width),
                                   #strides=(1, width), padding='same',
                                    #name='linear')
            #Gate layer
            conv_v = self.conv_op(padded_input, shape, "gated")
            #conv_v = tf.layers.conv2d(h, filter_size,
                                   #kernel_size=(height, width),
                                   #strides=(1, width), padding='same',
                                    #name='gate')
            #Elementwise multiplication
            #batch_size, max_len, 1, filter_size
            h = conv_w * tf.sigmoid(conv_v)
            #print(h)
        return h

    def create_embeddings(self, X, conf):
        #Create initial embeddings
        embeddings = tf.get_variable("embeds",(conf.vocab_size, conf.embedding_size), tf.float32, tf.random_uniform_initializer(-1.0,1.0))
        #Find embeddings for X
        embed = tf.nn.embedding_lookup(embeddings, X)
        #The original sentence was padding with k-1 zero in the beginning
        #So the first k-1
        #mask_layer = np.ones((conf.batch_size, conf.text_size, conf.embedding_size))
        #In the original paper, the first k-1 word will be padded
        #In a convolutional network, the kernel size is k, if we choose padding mode as 'SAME'
        #That means k/2 will be padded
        #mask_size = int(conf.filter_h/2)
        #mask_layer[:,0:mask_size,:] = 0
        #embed *= mask_layer
        
        #embed_shape = embed.get_shape().as_list()
        #Expand a dimension for convolutional layer
        #embed = tf.reshape(embed, (embed_shape[0], embed_shape[1], embed_shape[2], 1))
        embed = tf.expand_dims(embed, 1)
        return embed


    def conv_op(self, fan_in, shape, name):
        '''
        Depthwise convolution layer
        '''
        W = tf.get_variable("%s_W"%name, shape, tf.float32, tf.random_normal_initializer(0.0, 0.1))
        b = tf.get_variable("%s_b"%name, shape[-1], tf.float32, tf.constant_initializer(1.0))
        return tf.add(tf.nn.depthwise_conv2d(fan_in, W, strides=[1,1,1,1], padding='VALID'), b)
    
    def create_summaries(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("perplexity", self.perplexity)
        self.merged_summary_op = tf.summary.merge_all()
        
        
class GatedCharCNN(object):
    '''
    The input are word characters, to make full use of subword information
    '''
    def __init__(self, conf):
        tf.reset_default_graph()
        #Configuration
        options = {'char_cnn': {'activation': 'relu',
                    'embedding': {'dim': 4},
                    'filters': [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                    'max_characters_per_token': 50,
                    'n_characters': 262,
                    'n_highway': 2},
                    'lstm': {'cell_clip': 3,
                    'dim': 64,
                    'n_layers': 2,
                    'proj_clip': 3,
                    'projection_dim': conf.embedding_size,
                    'use_skip_connections': False},
                  'n_negative_samples_batch': 8192,
                  'n_train_tokens': 768648884}
        
        #Input sentence characters, batch_size * word_num * char_num   
        self.X = tf.placeholder(shape=[None, None, 50], dtype=tf.int32, name="X")
        #Label words
        self.y = tf.placeholder(shape=[None, None], dtype=tf.int32, name="y")
        #Create word Embeddings for input sentences
        #embed = self.create_embeddings(self.X, conf)
        embed = self._build_word_char_embeddings(self.X, options)
        #print(embed)
        embed = tf.expand_dims(embed, 1)
        #print(embed)

        #Initialize the input of each layer
        self.internal_states = []
        
        h, res_input = embed, embed
        for i in np.arange(conf.num_layers):
            self.internal_states.append(h)
            shape = (1, conf.filter_h, conf.filter_w, 1)
            h = self.glu(h, shape, conf, i)
            if i % conf.block_size == 2:
                h += res_input
                res_input = h
        

        #Flatten the output, (batch_size*max_len) * emb_size
        #print(h)
        self.out_layer = tf.squeeze(h, 1)
        #h = tf.reshape(h, (-1, conf.embedding_size))
        
        h = tf.reshape(h, (-1, conf.embedding_size))

        y_shape = self.y.get_shape().as_list()
        #Flatten the label, (batch_size*max_len) * 1
        self.y = tf.reshape(self.y, (-1, 1))
        #print(self.y)
        #Nce loss for the softmax
        softmax_w = tf.get_variable("softmax_w", [conf.vocab_size, conf.embedding_size], tf.float32, 
                                    tf.random_normal_initializer(0.0, 0.1))
        softmax_b = tf.get_variable("softmax_b", [conf.vocab_size], tf.float32, tf.constant_initializer(1.0))

        #Preferance: NCE Loss, heirarchial softmax, adaptive softmax
        #Note tf.nn.nce_loss has changed in new versions
        #self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=softmax_w, biases=softmax_b, 
                                                  #inputs=h, labels=self.y, 
                                                  #num_sampled=conf.num_sampled,
                                                  #num_classes=conf.vocab_size))
        
        self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                                   softmax_w, softmax_b,
                                   self.y, h,
                                   options['n_negative_samples_batch'],
                                   conf.vocab_size,
                                   num_true=1))
        
        #Optimizer
        #trainer = tf.train.MomentumOptimizer(conf.learning_rate, conf.momentum)
        #gradients = trainer.compute_gradients(self.loss)
        #clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
        #self.optimizer = trainer.apply_gradients(clipped_gradients)
        self.perplexity = tf.exp(self.loss)
        self.optimizer = tf.train.AdamOptimizer(conf.learning_rate).minimize(self.loss)

        self.create_summaries()
        
    def glu(self, inputs, shape, conf, layer_index):
        #Initialize kernel size
        height = conf.filter_h
        width = conf.filter_w
        #h, res_input = embed, embed
        kernel_width = shape[1]
        left_pad = kernel_width - 1
        paddings = [[0,0],[0,0],[left_pad,0],[0,0]]
        padded_input = tf.pad(inputs, paddings, "CONSTANT")

        with tf.variable_scope("layer_%d"%layer_index):
            #Linear layer
            conv_w = self.conv_op(padded_input, shape, "linear")
            #conv_w = tf.layers.conv2d(h, filter_size,
                                   #kernel_size=(height, width),
                                   #strides=(1, width), padding='same',
                                    #name='linear')
            #Gate layer
            conv_v = self.conv_op(padded_input, shape, "gated")
            #conv_v = tf.layers.conv2d(h, filter_size,
                                   #kernel_size=(height, width),
                                   #strides=(1, width), padding='same',
                                    #name='gate')
            #Elementwise multiplication
            #batch_size, max_len, 1, filter_size
            h = conv_w * tf.sigmoid(conv_v)
            #print(h)
        return h
    
    def _build_word_char_embeddings(self, inputs, options):
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

    def create_embeddings(self, X, conf):
        #Create initial embeddings
        embeddings = tf.get_variable("embeds",(conf.vocab_size, conf.embedding_size), tf.float32, tf.random_uniform_initializer(-1.0,1.0))
        #Find embeddings for X
        embed = tf.nn.embedding_lookup(embeddings, X)
        #The original sentence was padding with k-1 zero in the beginning
        embed = tf.expand_dims(embed, 1)
        return embed


    def conv_op(self, fan_in, shape, name):
        '''
        Depthwise convolution layer
        '''
        W = tf.get_variable("%s_W"%name, shape, tf.float32, tf.random_normal_initializer(0.0, 0.1))
        b = tf.get_variable("%s_b"%name, shape[-1], tf.float32, tf.constant_initializer(1.0))
        return tf.add(tf.nn.depthwise_conv2d(fan_in, W, strides=[1,1,1,1], padding='VALID'), b)
    
    def create_summaries(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("perplexity", self.perplexity)
        self.merged_summary_op = tf.summary.merge_all()
