import time
import numpy as np
import tensorflow as tf
import os
import argparse

import reader

TRAIN = 1
EVALUATE = 2
BOTH = 3
GEN_SENTENCES = 4

parser = argparse.ArgumentParser(description='Language model.')
parser.add_argument('--num-steps', action='store', type=int, default=30,
                    help='total number of recurrence steps, also known as the number of layers when our RNN is "unfolded"')
parser.add_argument('--max-grad-norm', action='store', type=int, default=5,
                    help='maximum permissible norm for gradient clipping')
parser.add_argument('--hidden-size', action='store', type=int, default=512,
                    help='number of processing units (neurons) in the hidden layers')
parser.add_argument('--embedding-size', action='store', type=int, default=100,
                    help='word embedding size')
parser.add_argument('--max-epoch', action='store', type=int, default=1,
                    help='maximum number of epochs trained with the initial learning rate')
parser.add_argument('--batch-size', action='store', type=int, default=64,
                    help='size for each batch of data')
parser.add_argument('--vocab-size', action='store', type=int, default=20000,
                    help='size of our vocabulary')
parser.add_argument('--ckpt-dir', action='store', default=os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'ckpt'),
                    help='directory for checkpointing model')
parser.add_argument('--data-dir', action='store', default=os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'data'),
                    help='directory of our dataset')
parser.add_argument('--out-dir', action='store', default=os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'output'),
                    help='output directory')
parser.add_argument('--embedding-dir', action='store', default=os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'data'),
                    help='directory of our predefined embeddings')
parser.add_argument('--is-training', action='store_true', default=True,
                    help='flag to separate training from testing')
parser.add_argument('--use_gpu', action='store_true', default=False,
                    help='use GPU instead of CPU')
parser.add_argument('--predefined-emb', action='store_true', default=False,
                    help='Indicates if we use predefined word embeddings')
parser.add_argument('--do_validation', action='store_true', default=False,
                    help='Run validation over test set')
parser.add_argument('--exp_c', action='store_true', default=False,
                    help='Run experiment C')
parser.add_argument('--action', action='store', type=int, default=3,
                    help='What action should be done when running?')
parser.add_argument('--base-lr', action='store', type=float, default=2,
                    help='Base learning rate')

args = parser.parse_args()
num_steps = args.num_steps
hidden_size = args.hidden_size
embedding_size = args.embedding_size
max_epoch = args.max_epoch
max_grad_norm = args.max_grad_norm
batch_size = args.batch_size
vocab_size = args.vocab_size
is_training = args.is_training
data_dir = args.data_dir
embedding_dir = args.embedding_dir
predef_emb = args.predefined_emb
ckpt_dir = args.ckpt_dir
do_validation = args.do_validation
action = args.action
base_lr = args.base_lr
exp_c = args.exp_c
out_dir = args.out_dir
processor = '/device:GPU:0' if args.use_gpu else '/cpu:0'


class LangModel(object):

    def __init__(self, is_training, predef_emb=None):
        """Setting parameters for ease of use"""
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        lstm_hidden_size = hidden_size if not exp_c else 1024
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.predefined_embedding = predef_emb

        # Actually 29 not 30
        self.num_steps = self.num_steps - 1

        # Creating placeholders for our input data and expected outputs (target data)
        self._input_data = tf.placeholder(tf.int32, [None, self.num_steps],
                                          name="inputs")  # [64#30]
        self._targets = tf.placeholder(tf.int32, [None, self.num_steps],
                                       name="targets")  # [64#30]

        batch_size_t = tf.shape(self._input_data)[0]

        # Creating the LSTM cell structure and connect it with the RNN structure
        with tf.device(processor):
            with tf.variable_scope("cell", reuse=tf.AUTO_REUSE,
                                   initializer=tf.contrib.layers.xavier_initializer()):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size,
                                                         forget_bias=0.0,
                                                         state_is_tuple=True)
                self._initial_state = lstm_cell.zero_state(batch_size_t,
                                                           tf.float32)

            # Creating the word embeddings and pointing them to the input data
            if self.predefined_embedding is None:
                # No predefined embedding so we train our own embedding.
                # Note that in this case the embeddings are TRAINABLE VARIABLES!
                with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
                    embedding = tf.get_variable("embedding",
                                                [vocab_size, embedding_size])
            else:
                print("Found predefined embedding, will use this embedding.")
                initial_weights = tf.constant(self.predefined_embedding,
                                              dtype=tf.float32)
                with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
                    embedding = tf.get_variable("embedding",
                                                initializer=initial_weights)

        # Create a lookup for the embedding matrix. Given an index get a column.
        embedded_inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE):
            softmax_b = tf.get_variable("softmax_b", [vocab_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
            softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
            if exp_c:
                softmax_wp = tf.get_variable("softmax_wp",
                                             [lstm_hidden_size, hidden_size],
                                             initializer=tf.contrib.layers.xavier_initializer())

        # Instanciating our RNN model
        state = self._initial_state
        outputs = []
        states = []
        losses = []
        logits = []
        with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
            for time_step in range(self.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()

                # Should be batch * embedding. Taking a timeslice in batch*time*emb.
                d_slice = embedded_inputs[:, time_step,
                          :]  # 64*29*100 ==> 64*1*100
                l_column = self._targets[:, time_step]  # 64*29 ==> 64*1

                # Chain cells by inserting the data slice (i.e. batch of t'th word embeddings)
                # and the previous cell's state or zero if it's the first cell.
                (new_h, state) = lstm_cell(d_slice, state)

                if exp_c:
                    # downscaling
                    new_h = tf.matmul(new_h, softmax_wp)
                # print("new_h",new_h.get_shape())
                logits_for_t = tf.matmul(new_h,
                                         softmax_w) + softmax_b  # 64*100 ==> 64*20 000

                loss_for_t = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=l_column,
                    logits=logits_for_t)  # computes cross-entopy
                losses.append(loss_for_t)
                logits.append(logits_for_t)

                # Keep track of output at step t, might be usefull?
                outputs.append(new_h)

                # Keep track of the cell states at timestep t, only the last one could be used
                states.append(state)

        self._first_state = states[0]
        # output of first
        self._distribution = logits_for_t[0]
        self._logits = logits
        # Get the state of last cell.
        state_f = states[-1]
        # Get the hidden state of the last cell, we will next apply softmax weights to it.
        cross_entropies_matrix = losses  # 30*64*1

        # Creating a logistic unit to return the probability of the output word
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
            logits = tf.matmul(output, softmax_w) + softmax_b

            # 30 * 64 * 20000
            logits_per_s = []
            for i in range(len(outputs)):
                logits_per_s.append(
                    tf.matmul(outputs[i], softmax_w) + softmax_b)

            # Find perp
            # out_final = tf.reshape(hidden_state_f, [-1, hidden_size])
            loss_2 = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                logits_per_s,
                [tf.transpose(self._targets)[i] for i in
                 range(self.num_steps)],
                [tf.ones([batch_size_t]) for i in range(self.num_steps)],
                name="loss_2")

            # Defining the loss and cost functions for the model's learning to work

            targets = tf.reshape(self._targets, [-1])
            weights = tf.ones([batch_size_t * self.num_steps])
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                                                                      [
                                                                          targets],
                                                                      [
                                                                          weights])
            self._cost = tf.reduce_sum(loss) / tf.cast(batch_size_t,
                                                       tf.float32)
            self._cost_2 = loss_2
            self._eval_op = loss_2
            # Store the final state
            self._final_state = state
            self._cross_entropies_m = cross_entropies_matrix
            self._batch_size_t = batch_size_t

        # Everything after this point is relevant only for training
        if not is_training:
            return

        # Creating the Training Operation for our Model
        # Create a variable for the learning rate
        self._lr = tf.Variable(0.0, trainable=False)
        # Get all TensorFlow variables marked as "trainable" (i.e. all of them except _lr, which we just created)
        tvars = tf.trainable_variables()
        # Define the gradient clipping threshold
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                          max_grad_norm)
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

    @property
    def cross_loss_m(self):
        return self._cross_entropies_m

    @property
    def distribution(self):
        return self._distribution

    @property
    def batch_size_t(self):
        return self._batch_size_t

    @property
    def first_state(self):
        return self._first_state


def process_embedding(mapping, id_to_words):
    """Makes sure that the embedding matrix corresponds with the ID's we
    computed earlier, otherwise words will map to wrong embeddings!"""

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
            # Nothing found, stays random.
            missing_words += 1

    print("In total, %d words were not found." % missing_words)

    return embedding


def evaluate(raw_data, model, session):
    train_data, test_data, id_to_words, voc_size = raw_data

    # Initializes the Execution Graph and the Session
    with session as session:
        state = session.run(model.initial_state)
        cost, state = session.run([model.cost_2, model.final_state],
                                  {model.input_data: test_data[0],
                                   model.targets: test_data[1],
                                   model.initial_state: state})
        print(np.array(cost).shape)


def run_training_epoch(session, model, data, op, is_train=False,
                       id_to_word=None):
    costs = 0
    iters = 0
    start_time = time.time()

    batches = reader.reader_iterator(data, batch_size, num_steps)
    for step, batch in enumerate(batches):

        x_batch = batch[0]
        y_batch = batch[1]

        # if state == None:
        state = session.run(model.initial_state, {model.input_data: x_batch})

        cost, state, _, cross_loss, bs, logs = session.run(
            [model.cost, model.final_state, op, model.cross_loss_m,
             model.batch_size_t, model._logits],
            {model.input_data: x_batch,
             model.targets: y_batch,
             model.initial_state: state})
        costs += cost

        logs = np.array(logs)
        pred = np.reshape(np.argmax(logs, axis=2), (bs, num_steps - 1))
        for y in range(len(pred)):
            pre_s = ""
            act_s = ""

            for x in range(len(pred[y])):
                try:
                    pre_s += " " + id_to_word[pred[y][x]]
                    act_s += " " + id_to_word[y_batch[y][x]]
                except:
                    pass

        # Add number of steps to iteration counter
        iters += model.num_steps

        if (step + 1) % 1 == 0 and is_train:
            print("Batches done: {0} \n".format(step + 1) +
                  "---cross-entopy: {0} \n".format(
                      costs / (iters * model.batch_size)) +
                  "---speed: {0} wps\n".format(
                      iters * model.batch_size / (time.time() - start_time))
                  )

    return costs


def get_embedding(id_to_words):
    if predef_emb:
        raw_embedding = reader.load_embedding(embedding_dir)
        embedding_matrix = process_embedding(raw_embedding, id_to_words)
    else:
        embedding_matrix = None
    return embedding_matrix


def write_values_to_file(values, dirname, filename):
    out_dir = "{0}_{1}".format(dirname, hidden_size)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = "{0}/{1}_{2}".format(out_dir, filename, hidden_size)

    with open(out_file, 'w') as of:
        for e in values:
            of.write("{0}\n".format(e))


def evaluate_model(model, session, test_data):
    batches = reader.reader_iterator(test_data, batch_size, num_steps)
    losses_list = []

    for b, (x_batch, y_batch) in enumerate(batches):
        # if state == None:
        state = session.run(model.initial_state, {model.input_data: x_batch})

        cost, state = session.run([model.cost_2, model.final_state],
                                  {model.input_data: x_batch,
                                   model.targets: y_batch,
                                   model.initial_state: state})

        losses_list.extend(np.power(2, cost))  # List of size batch
    return losses_list


def evaluate_model_from_file(test_data, ckpt_file, ckpt_dir):
    """Initializes the Execution Graph and the Session"""
    tf.reset_default_graph()
    with tf.Session() as session:
        model = LangModel(is_training=False)
        saver = tf.train.Saver()
        saver.restore(session, "{}/{}".format(ckpt_dir, ckpt_file))

        train_perplexity = evaluate_model(model, session, test_data)
        print(train_perplexity)


def train_model(raw_data):
    train_data, val_data, test_data, word_to_id, id_to_word, voc_size = raw_data
    embedding_matrix = get_embedding(id_to_word)

    # Initializes the Execution Graph and the Session
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.contrib.layers.xavier_initializer()

        with tf.name_scope("Train"):
            with tf.variable_scope("model", reuse=None,
                                   initializer=initializer):
                lstm_model = LangModel(is_training=True,
                                       predef_emb=embedding_matrix)

        # Initialize all variables
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()

        for i in range(max_epoch):
            learning_rate = base_lr / (i + 1)
            lstm_model.assign_lr(session, learning_rate)

            # Run training epoch with the data.
            train_perplexity = run_training_epoch(session, lstm_model,
                                                  train_data,
                                                  lstm_model.train_op, True,
                                                  id_to_word)
            print("Epoch %d : Train : %.3f" % (i + 1, train_perplexity))

        print("Checkpointing")
        saver.save(session, "{0}/lang_model".format(ckpt_dir, i),
                   global_step=hidden_size)

        complete_sentences(lstm_model, session, id_to_word)


def dict_map(word, mapping_dict, default):
    id = default
    try:
        id = mapping_dict[word]
    except:
        pass
    return id


def complete_sentences(model, session, id_to_words):
    words_to_id = dict(zip(id_to_words.values(), id_to_words.keys()))

    unk_id = words_to_id["<unk>"]

    incomplete_sentences = reader.load_incomplete_set(data_dir)
    incomplete_sentences = [[dict_map(x, words_to_id, unk_id) for x in y] for y
                            in incomplete_sentences]
    complete_sentences = [complete(model, session, x, words_to_id)
                          for x in incomplete_sentences]
    complete_sentences = [
        [dict_map(x, id_to_words, id_to_words[unk_id]) for x in y] for y in
        complete_sentences]
    complete_sentences_string = [" ".join(x) for x in complete_sentences]
    write_values_to_file(complete_sentences_string, out_dir,
                         "completed_sentences.txt")


def complete(model, session, incomplete_sentence, words_to_id):
    next_words = []
    max_completion_length = 20
    input_word = [[0] * (num_steps - 1)]
    state = session.run(model.initial_state, {model._input_data: input_word})
    for i in range(max_completion_length):

        if i < len(incomplete_sentence):
            input_word[0][0] = incomplete_sentence[i]

        distr, state = session.run([model.distribution, model._first_state],
                                   {model.input_data: input_word,
                                    model.targets: [[0] * 29],
                                    model.initial_state: state})

        most_probable_word = np.argmax(np.array(distr))

        if i >= len(incomplete_sentence):
            next_words.append(most_probable_word)

        if most_probable_word == words_to_id["<eos>"]:
            break

        input_word[0][0] = most_probable_word
    incomplete_sentence.extend(next_words)
    return incomplete_sentence


def do_sentence_completion(id_to_word):
    with tf.Session() as session:
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = LangModel(is_training=False)
        saver = tf.train.import_meta_graph("./ckpt2/lang_model-512_2.meta")
        print("Restoring from {0}".format(ckpt_dir))
        saver.restore(session, tf.train.latest_checkpoint("./ckpt2"))

        complete_sentences(model, session, id_to_word)


def main():
    """Reads the data and separates it into training data, validation data and
    testing data"""
    raw_data = reader.read_raw_data(vocab_size, data_dir)
    train_data, val_data, test_data, word_to_id, id_to_word, voc_size = raw_data

    print("Actual size of vocabulary: ", voc_size)

    start_time = time.time()
    if action == TRAIN:
        print("\n---------------Training start---------------")
        train_model(raw_data)
        # make_sentences(model, id_to_words)
        print("----------------Training end----------------\n")
    elif action == EVALUATE:
        print("\n---------------Evaluating start---------------")
        evaluate_model_from_file(test_data,
                                 "{0}/lang_model-{1}.meta".format(ckpt_dir,
                                                                  hidden_size),
                                 ckpt_dir)
        print("----------------Evaluating end----------------\n")
    elif action == BOTH:
        print("\n---------------Train/eval start---------------")
        train_model(raw_data)
        print("----------------Train/eval end----------------\n")
    elif action == GEN_SENTENCES:
        print("\n---------------Generate sentences begin---------------")
        do_sentence_completion(id_to_word)
        print("\n---------------Generate sentences end  ---------------")
    end_time = time.time()
    print("Total time: {0}".format(end_time - start_time))


if __name__ == "__main__":
    main()
