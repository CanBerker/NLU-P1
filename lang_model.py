import time
import numpy as np
import tensorflow as tf
import sys

import reader

#Initial weight scale
init_scale = 0.1
#Initial learning rate
learning_rate = 1.0
#Maximum permissible norm for the gradient (For gradient clipping -- another measure against Exploding Gradients)
max_grad_norm = 5
#The number of layers in our model
num_layers = 2
#The total number of recurrence steps, also known as the number of layers when our RNN is "unfolded"
num_steps = 30
#The number of processing units (neurons) in the hidden layers
hidden_size = 200
#The maximum number of epochs trained with the initial learning rate
max_epoch = 4
#The total number of epochs in training
max_max_epoch = 13
#The probability for keeping data in the Dropout Layer (This is an optimization, but is outside our scope for this notebook!)
#At 1, we ignore the Dropout Layer wrapping.
keep_prob = 1
#The decay for the learning rate
decay = 0.5
#The size for each batch of data
batch_size = 64
#The size of our vocabulary
vocab_size = 10000
#Training flag to separate training from testing
is_training = 1
#Data directory for our dataset
data_dir= "/Users/renatomarroquin/Downloads/data"

class LangModel(object):

    def __init__(self, is_training):
        ######################################
        # Setting parameters for ease of use #
        ######################################
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        ###############################################################################
        # Creating placeholders for our input data and expected outputs (target data) #
        ###############################################################################
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps]) #[64#30]
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps]) #[64#30]

        ##########################################################################
        # Creating the LSTM cell structure and connect it with the RNN structure #
        ##########################################################################
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
        self._initial_state = lstm_cell.zero_state(batch_size, tf.float32)

        ####################################################################
        # Creating the word embeddings and pointing them to the input data #
        ####################################################################
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, hidden_size])  
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        ###############################
        # Instanciating our RNN model #
        ###############################
        state = self._initial_state
        outputs = []
        with tf.variable_scope("RNN"):
          for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = lstm_cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])

        #########################################################################
        # Creating a logistic unit to return the probability of the output word #
        #########################################################################
        output = tf.reshape(outputs, [-1, hidden_size])
        softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size]) #[200x1000]
        softmax_b = tf.get_variable("softmax_b", [vocab_size]) #[1x1000]
        logits = tf.matmul(output, softmax_w) + softmax_b

        #########################################################################
        # Defining the loss and cost functions for the model's learning to work #
        #########################################################################
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(self._targets, [-1])],
                                                      [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        # Store the final state
        self._final_state = state

        #Everything after this point is relevant only for training
        if not is_training:
            return

        #################################################
        # Creating the Training Operation for our Model #
        #################################################
        # Create a variable for the learning rate
        self._lr = tf.Variable(0.0, trainable=False)
        # Get all TensorFlow variables marked as "trainable" (i.e. all of them except _lr, which we just created)
        tvars = tf.trainable_variables()
        # Define the gradient clipping threshold
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
        # Create the gradient descent optimizer with our learning rate
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        # Create the training TensorFlow Operation through our optimizer
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    # Helper functions for our LSTM RNN class

    # Assign the learning rate for this model
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    # Returns the input data for this model at a point in time
    @property
    def input_data(self):
        return self._input_data

    # Returns the targets for this model at a point in time
    @property
    def targets(self):
        return self._targets

    # Returns the initial state for this model
    @property
    def initial_state(self):
        return self._initial_state

    # Returns the defined Cost
    @property
    def cost(self):
        return self._cost

    # Returns the final state for this model
    @property
    def final_state(self):
        return self._final_state

    # Returns the current learning rate for this model
    @property
    def lr(self):
        return self._lr

    # Returns the training operation defined for this model
    @property
    def train_op(self):
        return self._train_op

def run_epoch(session, m, data, eval_op):

    #Define the epoch size based on the length of the data, batch size and the number of steps
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(m.initial_state)
    
    for step, (x, y) in enumerate(reader.reader_iterator(data, m.batch_size, m.num_steps)):
        #Evaluate and return cost, state by running cost, final_state and the function passed as parameter
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        
        #keeps track of the total costs for this epoch
        costs += cost
        
        #Add number of steps to iteration counter
        iters += m.num_steps

        if step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / epoch_size, np.exp(costs / iters),
              iters * m.batch_size / (time.time() - start_time)))

    # Returns the Perplexity rating for us to keep track of how the model is evolving
    return np.exp(costs / iters)

# Reads the data and separates it into training data, validation data and testing data
raw_data = reader.read_raw_data(data_dir)
train_data, test_data, _ = raw_data

#Initializes the Execution Graph and the Session
with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-init_scale,init_scale)
    
    # Instantiates the model for training
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        m = LangModel(is_training=True)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        mtest = LangModel(is_training=False)

    #Initialize all variables
    tf.global_variables_initializer().run()

    for i in range(max_max_epoch):
        # Define the decay for this epoch
        lr_decay = decay ** max(i - max_epoch, 0.0)
        
        # Set the decayed learning rate as the learning rate for this epoch
        m.assign_lr(session, learning_rate * lr_decay)
        print("Epoch %d : Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        
        # Run the loop for this epoch in the training model
        train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                   verbose=True)
        print("Epoch %d : Train Perplexity: %.3f" % (i + 1, train_perplexity))
    
    # Run the loop in the testing model to see how effective was our training
    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
    print("Test Perplexity: %.3f" % test_perplexity)
