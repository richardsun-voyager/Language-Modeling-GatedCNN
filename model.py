import numpy as np
import tensorflow as tf

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
