import numpy as np
import tensorflow as tf

class RNN(object):

    def __init__(self, conf):
        tf.reset_default_graph()
        #Input sentence, batch_size * word_num    
        self.X = tf.placeholder(shape=[conf.batch_size, conf.text_size], dtype=tf.int32, name="X")
        #Label words
        self.y = tf.placeholder(shape=[conf.batch_size, conf.text_size], dtype=tf.int32, name="y")
        #Create word Embeddings for input sentences
        embed = self.create_embeddings(self.X, conf)
        #Initialize the input of each layer
        
        output, state = self._build_rnn_graph_lstm(embed, conf, True)

        softmax_w = tf.get_variable(
            "softmax_w", [128, conf.vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [conf.vocab_size], dtype=tf.float32)
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
         # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [conf.batch_size, conf.text_size, conf.vocab_size])

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self.y,
            tf.ones([conf.batch_size, conf.text_size], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
        self.loss = tf.reduce_sum(loss)
        
        #Optimizer
        trainer = tf.train.MomentumOptimizer(conf.learning_rate, conf.momentum)
        gradients = trainer.compute_gradients(self.loss)
        clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
        self.optimizer = trainer.apply_gradients(clipped_gradients)
        self.perplexity = tf.exp(self.loss)
        #self.optimizer = tf.train.AdamOptimizer(conf.learning_rate).minimize(self.loss)

        self.create_summaries()
        
    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonical LSTM cells."""
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        def make_cell():
          cell = self._get_lstm_cell(is_training)
          if is_training:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=0.5)
          return cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(2)], state_is_tuple=True)

        self._initial_state = cell.zero_state(config.batch_size, tf.float32)
        state = self._initial_state
        # Simplified version of tf.nn.static_rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
        # outputs, state = tf.nn.static_rnn(cell, inputs,
        #                                   initial_state=self._initial_state)
        outputs = []
        with tf.variable_scope("RNN"):
          for time_step in range(config.text_size):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)
        #outputs: batch_size, max_len, hidden_size
        #output: batch_size*max_len, hidden_size
        output = tf.reshape(tf.concat(outputs, 1), [-1, 128])
        return output, state    
    
    def _get_lstm_cell(self, is_training):
    
          return tf.contrib.rnn.BasicLSTMCell(
              128, forget_bias=0.0, state_is_tuple=True,
              reuse=not is_training)

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
        #embed = tf.expand_dims(embed, 3)
        return embed


    def conv_op(self, fan_in, shape, name):
        W = tf.get_variable("%s_W"%name, shape, tf.float32, tf.random_normal_initializer(0.0, 0.1))
        b = tf.get_variable("%s_b"%name, shape[-1], tf.float32, tf.constant_initializer(1.0))
        return tf.add(tf.nn.conv2d(fan_in, W, strides=[1,1,1,1], padding='SAME'), b)
    
    def create_summaries(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("perplexity", self.perplexity)
        self.merged_summary_op = tf.summary.merge_all()
