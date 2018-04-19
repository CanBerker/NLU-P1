from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import itertools
import gensim

import numpy as np
import tensorflow as tf


def load_embedding(embedding_dir):
    print("Starting to load embeddings...")
    
    # Better to define as var!
    filename = os.path.join(embedding_dir, "wordembeddings-dim100.word2vec")
    
    # Load at location, this is a mapping of words-->embedding of 100!
    predef_emb = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=False)
    
    print("Done loading the embeddings!")
    
    # No more computation here!
    return predef_emb

def _read_words(filename):
    sentence_sz = 30
    with open(filename) as f:
        file_content = f.readlines()
    words2 = []

    rejected = 0
    nlines = 0
    for iline in file_content:
        nlines += 1
        iwords = iline.rstrip().split(" ")
        if (len(iwords) + 2 <= sentence_sz):
            npads = (sentence_sz - (len(iwords) + 2))
            oline2 = ["<bos>"]
            oline2.extend(iwords)
            oline2.append("<eos>")
            oline2.extend(["<pad>"]*npads)
            words2.extend(oline2)
        else:
            rejected += 1
    print("Read {0} lines from {1}".format(nlines, filename))
    print("Rejected {0} lines".format(rejected))
    return words2


def build_vocab(filename, vocab_size):
  data = _read_words(filename)

  counter = collections.Counter(data)

  # to make sure the 'unk' token is always in the first position
  count = [['<unk>', float('inf')]]
  count.extend(counter.most_common(vocab_size - 4)) # bos+pad+unk+eos
  count_pairs = sorted(count, key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))

  return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  # we know it is always in the first position
  return [word_to_id[word] if word in word_to_id else 0 for word in data]


def read_raw_data(vocab_size, data_path=None):

  train_path = os.path.join(data_path, "sentences.train")
  val_path = os.path.join(data_path, "sentences.val")
  test_path = os.path.join(data_path, "sentences.test")

  word_to_id, id_to_word = build_vocab(train_path, vocab_size)
  train_data_id = _file_to_word_ids(train_path, word_to_id)
  val_data_id = _file_to_word_ids(val_path, word_to_id)
  test_data_id = _file_to_word_ids(test_path, word_to_id)
  vocab_size = len(word_to_id)
  return train_data_id, val_data_id, test_data_id, word_to_id, id_to_word, vocab_size


def reader_iterator(raw_data, batch_size, num_steps):
  raw_data = np.array(raw_data, dtype=np.int32)
  
  #print(num_steps)
  
  n_words = int(len(raw_data))
  n_sentences = int(n_words / num_steps)
  
  sentence_list = np.reshape(raw_data, (n_sentences, num_steps))
  
  batches = []
  batch = [] 
  for sentence in sentence_list:
    if len(batch) >= batch_size and batch_size > 0:
        batches.append(np.array(batch))
        batch = []
    batch.append(sentence)
    
  # Non full batches also deserve a chance
  if len(batch) > 0:
   batches.append(np.array(batch))
 
  for batch in batches:
    x_batch = batch[:,:num_steps-1]
    y_batch = batch[:,1:]

    yield (x_batch, y_batch)
"""
    
  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps :(i+1)*num_steps ]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)
"""
