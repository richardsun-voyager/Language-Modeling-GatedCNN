import numpy as np
import tensorflow as tf

class GatedCNN(object):

    def __init__(self, conf):
        tf.reset_default_graph()
        #Input sentence, batch_size * word_num    
        self.X = tf.placeholder(shape=[conf.batch_size, conf.text_size], dtype=tf.int32, name="X")
        #Label words
        self.y = tf.placeholder(shape=[conf.batch_size, conf.text_size], dtype=tf.int32, name="y")
        #Create word Embeddings for input sentences
        embed = self.create_embeddings(self.X, conf)
        #Initialize the input of each layer
        h, res_input = embed, embed

        for i in range(conf.num_layers):
            #Get the last dimension, channels of last layer
            fanin_depth = h.get_shape()[-1]
            #filter size, note the size in the last layer is 1
            filter_size = conf.filter_size if i < conf.num_layers-1 else 1
            #print(filter_size)
            #shape of the filter
            shape = (conf.filter_h, conf.filter_w, fanin_depth, filter_size)
            
            with tf.variable_scope("layer_%d"%i):
                #Linear layer
                conv_w = self.conv_op(h, shape, "linear")
                #Gate layer
                conv_v = self.conv_op(h, shape, "gated")
                #Elementwise multiplication
                h = conv_w * tf.sigmoid(conv_v)
                #print(h)
                #residual layer, prevent weight diminishing
                #Note residual block should avoid the last layer
                if i % conf.block_size == 1:
                    h += res_input
                    res_input = h
        #Flatten the output, (batch_size*max_len) * emb_size
        #print(h)
        h = tf.reshape(h, (-1, conf.embedding_size))
        #print(h)
        y_shape = self.y.get_shape().as_list()
        #Flatten the label, (batch_size*max_len) * 1
        self.y = tf.reshape(self.y, (y_shape[0] * y_shape[1], 1))
        #print(self.y)
        #Nce loss for the softmax
        softmax_w = tf.get_variable("softmax_w", [conf.vocab_size, conf.embedding_size], tf.float32, 
                                    tf.random_normal_initializer(0.0, 0.1))
        softmax_b = tf.get_variable("softmax_b", [conf.vocab_size], tf.float32, tf.constant_initializer(1.0))

        #Preferance: NCE Loss, heirarchial softmax, adaptive softmax
        #Note tf.nn.nce_loss has changed in new versions
        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=softmax_w, biases=softmax_b, 
                                                  inputs=h, labels=self.y, 
                                                  partition_strategy="div",
                                                  num_sampled=conf.num_sampled,
                                                  num_classes=conf.vocab_size))
        
        #Optimizer
        trainer = tf.train.MomentumOptimizer(conf.learning_rate, conf.momentum)
        gradients = trainer.compute_gradients(self.loss)
        clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
        self.optimizer = trainer.apply_gradients(clipped_gradients)
        self.perplexity = tf.exp(self.loss)

        self.create_summaries()

    def create_embeddings(self, X, conf):
        #Create initial embeddings
        embeddings = tf.get_variable("embeds",(conf.vocab_size, conf.embedding_size), tf.float32, tf.random_uniform_initializer(-1.0,1.0))
        #Find embeddings for X
        embed = tf.nn.embedding_lookup(embeddings, X)
        #The original sentence was padding with k-1 zero in the beginning
        #So the first k-1
        mask_layer = np.ones((conf.batch_size, conf.text_size, conf.embedding_size))
        #In the original paper, the first k-1 word will be padded
        #In a convolutional network, the kernel size is k, if we choose padding mode as 'SAME'
        #That means k/2 will be padded
        mask_size = int(conf.filter_h/2)
        mask_layer[:,0:mask_size,:] = 0
        embed *= mask_layer
        
        #embed_shape = embed.get_shape().as_list()
        #Expand a dimension for convolutional layer
        #embed = tf.reshape(embed, (embed_shape[0], embed_shape[1], embed_shape[2], 1))
        embed = tf.expand_dims(embed, 3)
        return embed


    def conv_op(self, fan_in, shape, name):
        W = tf.get_variable("%s_W"%name, shape, tf.float32, tf.random_normal_initializer(0.0, 0.1))
        b = tf.get_variable("%s_b"%name, shape[-1], tf.float32, tf.constant_initializer(1.0))
        return tf.add(tf.nn.conv2d(fan_in, W, strides=[1,1,1,1], padding='SAME'), b)
    
    def create_summaries(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("perplexity", self.perplexity)
        self.merged_summary_op = tf.summary.merge_all()
