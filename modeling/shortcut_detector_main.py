from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import os
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from protos import pipeline_pb2
from modeling import trainer
from readers import reader
from readers.vcr_text_only_reader import InputFields
from readers.vcr_text_only_reader import NUM_CHOICES
from vcr import builder
from protos import pipeline_pb2
import json

flags.DEFINE_string('model_dir', None,
                    'Path to the directory which holds model checkpoints.')

flags.DEFINE_string('pipeline_proto', None, 'Path to the pipeline proto file.')

flags.DEFINE_string('output_jsonl_file', 'data/adversarial_train.json',
                    'Path to the output json file.')

FLAGS = flags.FLAGS

FIELD_ANSWER_PREDICTION = 'answer_prediction'


def _load_pipeline_proto(filename):
  """Loads pipeline proto from file.

  Args:
    filename: Path to the pipeline config file.

  Returns:
    An instance of pipeline_pb2.Pipeline.
  """
  with tf.io.gfile.GFile(filename, 'r') as fp:
    return text_format.Merge(fp.read(), pipeline_pb2.Pipeline())


def print_tensor_values(answer_choices, answer_choices_len, answer_label,
                        scores):
  """Prints tensor values."""
  for answer, answer_len, score in zip(answer_choices, answer_choices_len,
                                       scores):
    answer = ' '.join([x.decode('utf8') for x in answer[:answer_len]])
    print('[%.2lf] %s' % (score, answer))


def pack_tensor_values(answer_choices, answer_choices_len):
  """Prints tensor values."""
  results = []
  for answer, answer_len in zip(answer_choices, answer_choices_len):
    results.append([x.decode('utf8') for x in answer[:answer_len]])
  return results


def main(_):
  logging.set_verbosity(logging.DEBUG)

  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  pipeline_proto = _load_pipeline_proto(FLAGS.pipeline_proto)

  # Get `next_examples_ts' tensor.
  train_input_fn = reader.get_input_fn(pipeline_proto.train_reader,
                                       is_training=False)
  iterator = train_input_fn().make_initializable_iterator()
  next_examples_ts = iterator.get_next()

  # Build model that takes placeholder as inputs, and predicts the logits.
  (answer_label_pl, answer_choices_pl,
   answer_choices_len_pl) = (tf.placeholder(tf.int32, [1]),
                             tf.placeholder(tf.string, [1, NUM_CHOICES, None]),
                             tf.placeholder(tf.int32, [1, NUM_CHOICES]))
  model = builder.build(pipeline_proto.model, is_training=False)
  logits_ts = model.predict({
      InputFields.answer_choices_with_question: answer_choices_pl,
      InputFields.answer_choices_with_question_len: answer_choices_len_pl,
  })[FIELD_ANSWER_PREDICTION]
  scores_ts = logits_ts
  losses_ts = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_ts,
                                                      labels=tf.one_hot(
                                                          answer_label_pl,
                                                          depth=NUM_CHOICES))
  saver = tf.train.Saver()

  # Find the latest checkpoint file.
  ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
  assert ckpt_path is not None

  def _calc_answer_score_and_loss(answer_choices, answer_choices_len,
                                  answer_label):
    """Get answer prediction scores and losses.

    Args:
      answer_choices: A [NUM_CHOICES, max_answer_len] numpy array.
      answer_choices_len: A [NUM_CHOICES] numpy array.
      answer_label: An integer, the label.

    Returns:
      scores: A [NUM_CHOICES] numpy array.
      losses: A [NUM_CHOICES] numpy array.
    """
    (scores, losses) = sess.run(
        [scores_ts, losses_ts],
        feed_dict={
            answer_label_pl: np.expand_dims(answer_label, 0),
            answer_choices_pl: np.expand_dims(answer_choices, 0),
            answer_choices_len_pl: np.expand_dims(answer_choices_len, 0)
        })
    return scores[0], losses[0]

  # Run inference using the pretrained Bert model.
  with tf.Session() as sess, open(FLAGS.output_jsonl_file, 'w') as output_fp:
    sess.run(iterator.initializer)
    sess.run(tf.tables_initializer())
    saver.restore(sess, ckpt_path)
    logging.info('Restore from %s.', ckpt_path)

    batch_id = 0
    while True:
      batch_id += 1
      try:
        inputs_batched = sess.run(next_examples_ts)
        batch_size = len(inputs_batched[InputFields.annot_id])

        masks = np.array([[b'[MASK]'], [b'[MASK]'], [b'[MASK]'], [b'[MASK]']])

        for example_id in range(batch_size):
          annot_id = inputs_batched[InputFields.annot_id][example_id].decode(
              'utf8')
          answer_choices = inputs_batched[
              InputFields.answer_choices_with_question][example_id]
          answer_choices_len = inputs_batched[
              InputFields.answer_choices_with_question_len][example_id]
          answer_label = inputs_batched[InputFields.answer_label][example_id]

          # Scores of the original answer choices.
          original_scores, original_losses = _calc_answer_score_and_loss(
              answer_choices, answer_choices_len, answer_label)

          # print('=' * 128)
          # print('annot id=%s' % annot_id)
          # print('label=%i' % answer_label)
          # print_tensor_values(answer_choices, answer_choices_len, answer_label,
          #                     original_losses)

          # Adversarial atacking.
          max_losses = np.zeros(NUM_CHOICES)
          max_losses_answer_choices = answer_choices

          sep_pos = np.where(answer_choices == b'[SEP]')[1][[0, 2, 4, 6]]

          for pos_id in range(sep_pos.min() + 1, answer_choices_len.max()):
            # Calculate the new losses.
            new_answer_choices = np.concatenate([
                answer_choices[:, :pos_id], masks, answer_choices[:,
                                                                  pos_id + 1:]
            ], -1)
            scores, losses = _calc_answer_score_and_loss(
                new_answer_choices, answer_choices_len, answer_label)

            # Updating.
            token = answer_choices[:, pos_id]
            is_valid = np.logical_not(
                np.logical_or(
                    token == b'[PAD]',
                    np.logical_or(token == b'[CLS]', token == b'[SEP]')))

            # Maximize loss.
            adversarial_select_cond = np.logical_and(losses > max_losses,
                                                     is_valid)
            max_losses_answer_choices = np.where(
                np.expand_dims(adversarial_select_cond, -1), new_answer_choices,
                max_losses_answer_choices)
            max_losses = np.maximum(max_losses, losses)

          # print('Maximize loss')
          # print_tensor_values(max_losses_answer_choices, answer_choices_len,
          #                     answer_label, max_losses)

          answer_choices = pack_tensor_values(answer_choices,
                                              answer_choices_len)
          adversarial_answer_choices = pack_tensor_values(
              max_losses_answer_choices, answer_choices_len)
          results = {
              'annot_id': annot_id,
              'answer_label': int(answer_label),
              'answer_choices': answer_choices,
              'adversarial_answer_choices': adversarial_answer_choices,
          }
          output_fp.write(json.dumps(results) + '\n')

        if batch_id % 10 == 0:
          logging.info('batch_id=%i', batch_id)
      except tf.errors.OutOfRangeError as ex:
        logging.info('Done!')
        break

  output_fp.close()


if __name__ == '__main__':
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_proto')
  app.run(main)
