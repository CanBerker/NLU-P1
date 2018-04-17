import time
import numpy as np
import tensorflow as tf
import os
import sys
import argparse

import reader

parser = argparse.ArgumentParser(description='Language model.')
parser.add_argument('--max-grad-norm', action='store', type=int, default=5, help='maximum permissible norm for gradient clipping')
parser.add_argument('--num-layers',    action='store', type=int, default=2, help='number of layers in our model')
parser.add_argument('--num-steps',     action='store', type=int, default=30, help='total number of recurrence steps, also known as the number of layers when our RNN is "unfolded"')
parser.add_argument('--hidden-size',   action='store', type=int, default=512, help='number of processing units (neurons) in the hidden layers')
parser.add_argument('--embedding-size',action='store', type=int, default=100, help='word embedding size')
parser.add_argument('--max-epoch',     action='store', type=int, default=1, help='maximum number of epochs trained with the initial learning rate')
parser.add_argument('--max-max-epoch', action='store', type=int, default=2, help='total number of epochs in training')
parser.add_argument('--batch-size',    action='store', type=int, default=64, help='size for each batch of data')
parser.add_argument('--vocab-size',    action='store', type=int, default=20000, help='size of our vocabulary')
parser.add_argument('--init-scale',    action='store', type=float, default=0.1, help='initial weight scale')
parser.add_argument('--learning-rate', action='store', type=float, default=1.0, help='initial learning rate')
parser.add_argument('--decay',         action='store', type=float, default=0.5, help='decay for the learning rate')
parser.add_argument('--ckpt-dir',      action='store', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ckpt'), help='directory for checkpointing model')
parser.add_argument('--data-dir',      action='store', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'), help='directory of our dataset')
parser.add_argument('--embedding-dir', action='store', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'), help='directory of our predefined embeddings')
parser.add_argument('--is-training',   action='store_true', default=False, help='flag to separate training from testing')
parser.add_argument('--use_gpu',       action='store_true', default=False, help='use GPU instead of CPU')
parser.add_argument('--predefined-emb',action='store_true', default=False, help='Indicates if we use predefined word embeddings')
parser.add_argument('--do_validation', action='store_true', default=False, help='Run validation over test set')

args = parser.parse_args()
init_scale    = args.init_scale
learning_rate = args.learning_rate
max_grad_norm = args.max_grad_norm
num_layers    = args.num_layers
num_steps     = args.num_steps
hidden_size   = args.hidden_size
embedding_size= args.embedding_size
max_epoch     = args.max_epoch
max_max_epoch = args.max_max_epoch
decay         = args.decay
batch_size    = args.batch_size
vocab_size    = args.vocab_size
is_training   = args.is_training
data_dir      = args.data_dir
embedding_dir = args.embedding_dir
predef_emb    = args.predefined_emb
ckpt_dir      = args.ckpt_dir
do_validation = args.do_validation
processor     = '/device:GPU:0' if args.use_gpu else '/cpu:0'

class LangModel(object):

    def __init__(self, is_training, predef_emb=None):
        ######################################
        # Setting parameters for ease of use #
        ######################################
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.predefined_embedding = predef_emb
        
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
        with tf.device(processor):
            #
            if self.predefined_embedding is None:                
                # No predefined embedding so we train our own embedding.
                # Note that in this case the embeddings are TRAINABLE VARIABLES!
                embedding = tf.get_variable("embedding", [vocab_size, embedding_size])
            else:
                print("Found predefined embedding, will use this embedding.")
                initial_weights = tf.constant(self.predefined_embedding, dtype = tf.float32)
                embedding = tf.get_variable("embedding", initializer=initial_weights)
                
            # Create a lookup for the embedding matrix. Given an index get a column.
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        ###############################
        # Instanciating our RNN model #
        ###############################
        state = self._initial_state
        outputs = []
        states = []
        with tf.variable_scope("RNN"):
          for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            
            # Should be batch * embedding. Taking a timeslice in batch*time*emb.
            d_slice = inputs[:, time_step, :] 
            
            # Chain cells by inserting the data slice (i.e. batch of t'th word embeddings)
            # and the previous cell's state or zero if it's the first cell.
            (new_h, state) = lstm_cell(d_slice, state)
            
            # Keep track of output at step t, might be usefull?
            outputs.append(new_h)
            
            # Keep track of the cell states at timestep t, only the last one is needed.
            states.append(state)
            
        # Get the state of last cell.
        state_f = states[-1]
        # Get the hidden state of the last cell, we will next apply softmax weights to it.
        hidden_state_f = state_f.h
        
        
        #########################################################################
        # Creating a logistic unit to return the probability of the output word #
        #########################################################################
        output = tf.reshape(hidden_state_f, [-1, hidden_size])
        softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size]) #[512x20000]
        softmax_b = tf.get_variable("softmax_b", [vocab_size]) #[1x20000]
        logits = tf.matmul(output, softmax_w) + softmax_b
        
        logits_per_s = []
        for i in range(len(outputs)):
            logits_per_s.append(tf.matmul(outputs[i], softmax_w) + softmax_b)
        
        # Find perp
        out_final = tf.reshape(hidden_state_f, [-1, hidden_size])
        loss_2 = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                logits_per_s, 
                [tf.transpose(self._targets)[i] for i in range(num_steps)], 
                [tf.ones([batch_size]) for i in range(num_steps)]
                )

        #########################################################################
        # Defining the loss and cost functions for the model's learning to work #
        #########################################################################
        
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._targets[:,-1], logits=logits )
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._cost_2 = cost_2 = loss_2
        self._eval_op = loss_2
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
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
        #grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
        # Create the gradient descent optimizer with our learning rate
        optimizer = tf.train.AdamOptimizer(self.lr)
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
        
    @property
    def eval_op(self):
        return self._eval_op

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
        
    # Returns the defined Cost
    @property
    def cost_2(self):
        return self._cost_2

def process_embedding(mapping, id_to_words):
    # Makes sure that the embedding matrix corresponds with the 
    # ID's we computed earlier, otherwise words will map to wrong embeddings!
    
    # Contains all the words that are in the predef. embedding.
    vocab_of_mapping = mapping.vocab

    # Initially all zeros, we will fill in the ones we actually know.
    embedding = np.random.uniform(-0.2, 0.2, (vocab_size, embedding_size))
    
    missing_words = 0
    for item in id_to_words.items():
        id = item[0]
        word = item[1].strip()
        
        if word in vocab_of_mapping:
            # We found a word so the row at index id should be the embedding.
            embedding[id] = mapping[word]
        else:
            #Nothing found, stays random.
            missing_words+=1
    
    print("In total, %d words were not found." % missing_words)
    
    return embedding
    
    
def run_eval(session, m, data, op):

    perp_list = []

    #Define the epoch size based on the length of the data, batch size and the number of steps   
    batch_size = 1
    epoch_size = ((len(data) // batch_size) - 1) // m.num_steps
    
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(m.initial_state)
    perp_p_s = None
    for step, (x_batch, y_batch) in enumerate(reader.reader_iterator(data, batch_size, m.num_steps)):

        cost, state, _ = session.run([m.cost_2, m.final_state, op],
                                {m.input_data: x_batch,
                                 m.targets: y_batch,
                                 m.initial_state: state})
        perp_list.extend(cost)
        
        
        #Add number of steps to iteration counter
        iters += m.num_steps
        

    # Returns the Perplexity rating for us to keep track of how the model is evolving
    return perp_list
    
def run_epoch(session, m, data, op, is_train=False):

    #Define the epoch size based on the length of the data, batch size and the number of steps   
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(m.initial_state)
    perp_p_s = None
    for step, (x_batch, y_batch) in enumerate(reader.reader_iterator(data, m.batch_size, m.num_steps)):
                
        #Evaluate and return cost, state by running cost, final_state and the function passed as parameter
        if is_train:
            cost, state, _  = session.run([m.cost, m.final_state, op],
                                    {m.input_data: x_batch,
                                     m.targets: y_batch,
                                     m.initial_state: state})
            #keeps track of the total costs for this epoch
            costs += cost
        else:
            cost, state, _ = session.run([m.cost_2, m.final_state, op],
                                    {m.input_data: x_batch,
                                     m.targets: y_batch,
                                     m.initial_state: state})
            costs = cost
        
        
        #Add number of steps to iteration counter
        iters += m.num_steps

        #Don't know what this part here is?
        if step % 20 == 0 and is_train:
            print("%d perplexity: %.3f speed: %.0f wps" % (step, np.exp(costs / iters),
              iters * m.batch_size / (time.time() - start_time)))
            
             

    # Returns the Perplexity rating for us to keep track of how the model is evolving
    if is_train:
        return np.exp(costs / iters)
    else:
        return costs

def evaluate_model(test_data, ckpt_dir):
    #Initializes the Execution Graph and the Session
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-init_scale,init_scale)

        # Instantiates the model for training
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = LangModel(is_training=True, predef_emb=embedding_matrix)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = LangModel(is_training=False, predef_emb=embedding_matrix)
        saver = tf.train.Saver()
        saver.restore(session, ckpt_dir)
        print("---")#m.predef_emb.eval())
        for v in tf.get_default_graph().as_graph_def().node:
            print (v.name)
        valid_perplexity = run_eval(session, m, test_data, m.eval_op)
        print("Valid shape:", len(valid_perplexity))
        return valid_perplexity
        #Initialize all variables
        #tf.global_variables_initializer().run()
        #perp_p_s = np.power(np.array(perp_p_s),2)

def get_embedding():
    if predef_emb:
        # Obtain the word -> vec mapping from reader.
        raw_embedding = reader.load_embedding(embedding_dir)
        
        # optional but unethical: also match test data in embedding matrix.
        
        # Process all our seen data (in ID's) to create an embedding matrix
        # such that the rows contain the correct embedding!
        embedding_matrix = process_embedding(raw_embedding, id_to_words)
            
    else:
        embedding_matrix = None
    return embedding_matrix
    

def do_training():
    #Initializes the Execution Graph and the Session
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-init_scale,init_scale)
        
        if predef_emb:
            # Obtain the word -> vec mapping from reader.
            raw_embedding = reader.load_embedding(embedding_dir)
            
            # optional but unethical: also match test data in embedding matrix.
            
            # Process all our seen data (in ID's) to create an embedding matrix
            # such that the rows contain the correct embedding!
            embedding_matrix = process_embedding(raw_embedding, id_to_words)
            
        else:
            embedding_matrix = None
        
        # Instantiates the model for training
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = LangModel(is_training=True, predef_emb=embedding_matrix)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = LangModel(is_training=False, predef_emb=embedding_matrix)
    
        #Initialize all variables
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
    
        for i in range(max_max_epoch):
            # Define the decay for this epoch
            lr_decay = decay ** max(i - max_epoch, 0.0)
            
            # Set the decayed learning rate as the learning rate for this epoch
            m.assign_lr(session, learning_rate * lr_decay)
            print("Epoch %d : Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            
            # Run the loop for this epoch in the training model
            train_perplexity = run_epoch(session, m, train_data, m.train_op, True)
            print("Epoch %d : Train Perplexity: %.3f" % (i + 1, train_perplexity))
        
        print("Checkpointing")
        saver.save(session, "{0}/lang_model.ckpt".format(ckpt_dir, i))
        
        # Run the loop in the testing model to see how effective was our training
        test_perplexity = run_epoch(session, mtest, test_data, tf.no_op(),True)
        print("Test Perplexity: %.3f" % test_perplexity)

# Reads the data and separates it into training data, validation data and testing data
raw_data = reader.read_raw_data(vocab_size, data_dir)
train_data, test_data, id_to_words, voc_size = raw_data
if do_validation:
    ckpt_file = "{0}/lang_model.ckpt".format(ckpt_dir)
    files = [f for f in os.listdir(ckpt_dir)]
    if (len(files) > 0):
        print("Starting validation")
        start_time = time.time()
        #perp_list = evaluate_model(test_data, ckpt_file)
        end_time = time.time()
        #print(perp_list)
        print("Validation took: %.3f secs" % ((end_time-start_time)/1000))
else:
    embedding_matrix = get_embedding()
    print("Starting training")
    start_time = time.time()
    do_training()
    end_time = time.time()
    print("Training took: %.3f secs" % ((end_time-start_time)/1000))
