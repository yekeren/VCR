from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

from absl import app
from absl import flags
from absl import logging

import hashlib
import io
import zipfile
import numpy as np
import PIL.Image
import tensorflow as tf

from bert import tokenization
from bert.modeling import BertConfig
from bert.modeling import BertModel
from bert.modeling import get_assignment_map_from_checkpoint

from modeling.layers import token_to_id

flags.DEFINE_string('bert_vocab_file',
                    'data/bert/tf1.x/cased_L-12_H-768_A-12/vocab.txt',
                    'Path to the Bert vocabulary file.')

flags.DEFINE_string('bert_config_file',
                    'data/bert/tf1.x/cased_L-12_H-768_A-12/bert_config.json',
                    'Path to the Bert configuration file.')

flags.DEFINE_string('bert_checkpoint_file',
                    'data/bert/tf1.x/cased_L-12_H-768_A-12/bert_model.ckpt',
                    'Path to the Bert checkpoint file.')

flags.DEFINE_boolean('do_lower_case', False,
                     'To be passed to the bert tokenizer.')

flags.DEFINE_string('annotations_jsonl_file', 'data/vcr1annots/val.jsonl',
                    'Path to the annotations file in jsonl format.')

flags.DEFINE_integer('num_shards', 10,
                     'Number of shards of the output tfrecord files.')

flags.DEFINE_integer('shard_id', 0, 'Shard id of the current process.')

flags.DEFINE_string('output_bert_feature_dir',
                    'output/bert-tmp/cased_L-12_H-768_A-12',
                    'Path to the directory saving features.')

FLAGS = flags.FLAGS

# GENDER_NEUTRAL_NAMES = [
#     'Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry',
#     'Jody', 'Kendall', 'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn'
# ]
UNK = 100
NUM_CHOICES = 4
GENDER_NEUTRAL_NAMES = [
    'Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry',
    'Kendall', 'Frankie', 'Pat', 'Quinn'
]

_NUM_PARTITIONS = 100


def get_partition_id(annot_id, num_partitions=_NUM_PARTITIONS):
  split, number = annot_id.split('-')
  return int(number) % num_partitions


def _load_annotations(filename):
  """Loads annotations from file.

  Args:
    filename: Path to the jsonl annotations file.

  Returns:
    A list of python dictionary, each is parsed from a json object.
  """
  with tf.io.gfile.GFile(filename, 'r') as f:
    return [json.loads(x.strip('\n')) for x in f]


def _fix_tokenization(tokenized_sent, obj_to_type, bert_tokenizer,
                      do_lower_case):
  """Converts tokenized annotations into tokenized sentence.

  Args:
    tokenized_sent: Tokenized sentence with detections collapsed to a list.
    obj_to_type: [person, person, pottedplant] indexed by the labels.

  Returns:
    tokenized_sent: A list of string tokens.
  """
  case_fn = lambda x: x.lower() if do_lower_case else x

  new_tokenization_with_tags = []
  for tok in tokenized_sent:
    if isinstance(tok, list):
      for idx in tok:
        obj_type = obj_to_type[idx]
        text_to_use = GENDER_NEUTRAL_NAMES[
            idx %
            len(GENDER_NEUTRAL_NAMES)] if obj_type == 'person' else obj_type
        new_tokenization_with_tags.append((case_fn(text_to_use), idx))
    else:
      for sub_tok in bert_tokenizer.wordpiece_tokenizer.tokenize(case_fn(tok)):
        new_tokenization_with_tags.append((sub_tok, -1))

  tokenized_sent, tags = zip(*new_tokenization_with_tags)
  return list(tokenized_sent), list(tags)


def _create_bert_embeddings(annot, bert_tokenizer, do_lower_case, bert_fn):
  """Creates an example from the annotation.

  Args:
    annot: A python dictionary parsed from the json object.
    bert_tokenizer: A tokenization.FullTokenizer object.
    do_lower_case: If true, convert text to lower case.
    bert_fn: Function used to extract bert features.

  Returns:
    A list of [sequence_len, embedding_dims] arrays.
  """

  obj_to_type = annot['objects']

  question = annot['question']
  answer_choices = annot['answer_choices']
  assert NUM_CHOICES == len(answer_choices)

  question_tokens, _ = _fix_tokenization(question, obj_to_type, bert_tokenizer,
                                         do_lower_case)

  # Encode answer choices.
  bert_outputs = []
  for idx, tokenized_sent in enumerate(answer_choices):
    answer_choice_tokens, _ = _fix_tokenization(tokenized_sent, obj_to_type,
                                                bert_tokenizer, do_lower_case)
    input_sequence = ['[CLS]'] + question_tokens + [
        '[SEP]'
    ] + answer_choice_tokens + ['[SEP]']
    sequence_output, pooled_output = bert_fn(input_sequence)
    import pdb
    pdb.set_trace()
    bert_outputs.append(bert_fn(input_sequence))
  return bert_outputs


def main(_):
  logging.set_verbosity(logging.INFO)

  for i in range(_NUM_PARTITIONS):
    tf.io.gfile.makedirs(os.path.join(FLAGS.output_bert_feature_dir,
                                      '%02d' % i))

  # Create Bert model.
  bert_tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.bert_vocab_file,
                                              do_lower_case=FLAGS.do_lower_case)

  # Bert prediction.
  input_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
  token_to_id_layer = token_to_id.TokenToIdLayer(FLAGS.bert_vocab_file,
                                                 unk_token_id=UNK)

  bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)
  bert_model = BertModel(bert_config,
                         is_training=False,
                         input_ids=token_to_id_layer(
                             tf.expand_dims(input_placeholder, 0)))
  sequence_output = bert_model.get_sequence_output()[0]
  pooled_output = bert_model.get_pooled_output()[0]
  saver = tf.compat.v1.train.Saver()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.compat.v1.Session(config=config)
  sess.run(tf.compat.v1.tables_initializer())
  saver.restore(sess, FLAGS.bert_checkpoint_file)

  for name in sess.run(tf.compat.v1.report_uninitialized_variables()):
    logging.warn('%s is uninitialized!', name)

  def _bert_fn(sequence):
    return sess.run([sequence_output, pooled_output],
                    feed_dict={input_placeholder: sequence})

  # Load annotations.
  annots = _load_annotations(FLAGS.annotations_jsonl_file)
  logging.info('Loaded %i annotations.', len(annots))

  shard_id, num_shards = FLAGS.shard_id, FLAGS.num_shards
  assert 0 <= shard_id < num_shards

  for idx, annot in enumerate(annots):
    if (idx + 1) % 1000 == 0:
      logging.info('On example %i/%i.', idx + 1, len(annots))

    annot_id = int(annot['annot_id'].split('-')[-1])
    if annot_id % num_shards != shard_id:
      continue

    # Check npy file.
    part_id = get_partition_id(annot['annot_id'])
    output_file = os.path.join(FLAGS.output_bert_feature_dir, '%02d' % part_id,
                               annot['annot_id'] + '.npy')
    if os.path.isfile(output_file):
      logging.info('%s is there.', output_file)
      continue

    annot_id = int(annot['annot_id'].split('-')[-1])
    if annot_id % num_shards != shard_id:
      continue

    # Create TF example.
    bert_outputs = _create_bert_embeddings(annot, bert_tokenizer,
                                           FLAGS.do_lower_case, _bert_fn)
    with open(output_file, 'wb') as f:
      np.save(f, bert_outputs)

  logging.info('Done')


if __name__ == '__main__':
  app.run(main)
