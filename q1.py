import collections
import os
import sys

import tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", "data", "Path to training/test data.")
flags.DEFINE_string("save_path", None, "Path to save model.")
flags.DEFINE_integer("vocab_sz", 20000, "Vocabulary size. Default: 20k.")
FLAGS = flags.FLAGS

sentence_sz = 30

def preprocess(content):
     words = []
     nlines = 0
     for iline in content:
         nlines += 1
         iwords = iline.rstrip().split(" ")
         print("{0} {1} {2}".format(len(iwords), sentence_sz, len(iwords) + 2 <= sentence_sz))
         if (len(iwords) + 2 <= sentence_sz):
             padding = " <pad>" * (sentence_sz - (len(iwords) + 2))
             oline = "<bos> {0} <eos>{1}".format(" ".join(iwords), padding)
             words.extend(oline.split(" "))
     return words

def build_vocab(filename, vocab_sz):
    # read words from preproc file
    words = []
    with open(filename) as f:
        fcontent = f.readlines()
    words = preprocess(fcontent)
        #for line in f:
        #    words.extend(line.split(" "))

    # build dictionary  
    count = [['unk', -1]]
    counter = collections.Counter()
    for w in words:
        counter[w] = counter[w] + 1
    # only taking the popular ones
    count.extend(counter.most_common(vocab_sz - 1))
    dictionary = dict()
    for word, _ in count:
      dictionary[word] = len(dictionary)

    # replace rare words with unk token
    word_ids = list()
    unk_count = 0
    for word in words:
      index = dictionary.get(word, 0)
      # if not found
      if index == 0:
        unk_count += 1
      word_ids.append(index)
    # updating number of unknowns
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    #print(count)
    #print(dictionary)
    return word_ids, count, dictionary, reversed_dictionary


def load_data(data_path, vocab_sz):
    """Returns: tuple (train_data, vocabulary)"""

    #train_path = os.path.join(data_path, "sentences.train.small")
    train_path = os.path.join(data_path, "sentences.train")
    print("Building {0} vocabulary from {1}".format(vocab_sz, train_path))
    word_ids, count, dictionary, reverse_dictionary = build_vocab(train_path, vocab_sz)


def main(_):
    load_data(FLAGS.data_path, FLAGS.vocab_sz)

if __name__ == "__main__":
    tf.app.run()
