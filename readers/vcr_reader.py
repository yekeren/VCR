from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import tensorflow as tf

from protos import reader_pb2

PAD = '[PAD]'
NUM_CHOICES = 4


class TFExampleFields(object):
  """Fields in the tf.train.Example."""
  img_id = 'img_id'
  annot_id = 'annot_id'
  question = 'question'
  answer_label = 'answer_label'
  answer_choices = [
      'answer_choice_1', 'answer_choice_2', 'answer_choice_3', 'answer_choice_4'
  ]
  img_encoded = 'image/encoded'


class InputFields(object):
  """Names of the input tensors."""
  img_id = 'img_id'
  img_data = 'img_data'
  img_height = 'img_height'
  img_width = 'img_width'

  annot_id = 'annot_id'
  question = 'question'
  question_len = 'question_len'
  answer_label = 'answer_label'
  answer_choices = 'answer_choices'
  answer_choices_len = 'answer_choices_len'
  answer_choices_with_question = 'answer_choices_with_question'
  answer_choices_with_question_len = 'answer_choices_with_question_len'


def _parse_single_example(example):
  """Parses a single tf.Example proto.

  Args:
    example: An Example proto.

  Returns:
    A dictionary indexed by tensor name.
  """
  features = {
      TFExampleFields.img_id: tf.io.FixedLenFeature([], tf.string),
      TFExampleFields.img_encoded: tf.io.FixedLenFeature([], tf.string),
      TFExampleFields.annot_id: tf.io.FixedLenFeature([], tf.string),
      TFExampleFields.question: tf.io.VarLenFeature(tf.string),
      TFExampleFields.answer_label: tf.io.FixedLenFeature([], tf.int64),
  }
  for field in TFExampleFields.answer_choices:
    features[field] = tf.io.VarLenFeature(tf.string)
  parsed = tf.io.parse_single_example(example, features)

  # Parse image.
  image = tf.image.decode_jpeg(parsed[TFExampleFields.img_encoded], channels=3)
  image_shape = tf.shape(image)
  image_height, image_width = image_shape[0], image_shape[1]

  # Parse question.
  question = tf.sparse.to_dense(parsed[TFExampleFields.question],
                                default_value=PAD)
  question_len = tf.shape(question)[0]

  # Parse answer choices.
  def _pad_answer_choices(answer_choices):
    answer_choices_len = [tf.shape(x)[0] for x in answer_choices]
    padded_size = tf.reduce_max(answer_choices_len)
    answer_choices = tf.stack([
        tf.pad(x,
               paddings=[[0, padded_size - answer_choices_len[i]]],
               mode='CONSTANT',
               constant_values=PAD) for i, x in enumerate(answer_choices)
    ])
    return answer_choices, answer_choices_len

  answer_choices_list = [
      tf.sparse.to_dense(parsed[field], default_value=PAD)
      for field in TFExampleFields.answer_choices
  ]
  answer_choices_with_question_list = [
      tf.concat([['[CLS]'], question, ['[SEP]'], x, ['[SEP]']], 0)
      for x in answer_choices_list
  ]

  (answer_choices,
   answer_choices_len) = _pad_answer_choices(answer_choices_list)
  (answer_choices_with_question, answer_choices_with_question_len
  ) = _pad_answer_choices(answer_choices_with_question_list)

  return {
      InputFields.img_id:
          parsed[TFExampleFields.img_id],
      InputFields.img_data:
          image,
      InputFields.img_height:
          image_height,
      InputFields.img_width:
          image_width,
      InputFields.annot_id:
          parsed[TFExampleFields.annot_id],
      InputFields.question:
          question,
      InputFields.question_len:
          question_len,
      InputFields.answer_label:
          tf.dtypes.cast(parsed[TFExampleFields.answer_label], tf.int32),
      InputFields.answer_choices:
          answer_choices,
      InputFields.answer_choices_len:
          answer_choices_len,
      InputFields.answer_choices_with_question:
          answer_choices_with_question,
      InputFields.answer_choices_with_question_len:
          answer_choices_with_question_len,
  }


def _create_dataset(options, is_training, input_pipeline_context=None):
  """Creates dataset object based on options.

  Args:
    options: An instance of reader_pb2.Reader.
    is_training: If true, shuffle the dataset.
    input_pipeline_context: A tf.distribute.InputContext instance.

  Returns:
    A tf.data.Dataset object.
  """
  dataset = tf.data.Dataset.list_files(options.input_pattern[:],
                                       shuffle=is_training)

  batch_size = options.batch_size
  if input_pipeline_context:
    if input_pipeline_context.num_input_pipelines > 1:
      dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                              input_pipeline_context.input_pipeline_id)
    batch_size = input_pipeline_context.get_per_replica_batch_size(
        options.batch_size)

  if is_training:
    dataset = dataset.repeat()
    dataset = dataset.shuffle(options.shuffle_buffer_size)
  dataset = dataset.interleave(tf.data.TFRecordDataset,
                               cycle_length=options.interleave_cycle_length,
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(map_func=_parse_single_example,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  padded_shapes = {
      InputFields.img_id: [],
      InputFields.img_data: [None, None, 3],
      InputFields.img_height: [],
      InputFields.img_width: [],
      InputFields.annot_id: [],
      InputFields.question: [None],
      InputFields.question_len: [],
      InputFields.answer_label: [],
      InputFields.answer_choices: [NUM_CHOICES, None],
      InputFields.answer_choices_len: [NUM_CHOICES],
      InputFields.answer_choices_with_question: [NUM_CHOICES, None],
      InputFields.answer_choices_with_question_len: [NUM_CHOICES],
  }
  padding_values = {
      InputFields.img_id: '',
      InputFields.img_data: tf.constant(0, dtype=tf.uint8),
      InputFields.img_height: 0,
      InputFields.img_width: 0,
      InputFields.annot_id: '',
      InputFields.question: PAD,
      InputFields.question_len: 0,
      InputFields.answer_label: -1,
      InputFields.answer_choices: PAD,
      InputFields.answer_choices_len: 0,
      InputFields.answer_choices_with_question: PAD,
      InputFields.answer_choices_with_question_len: 0,
  }
  dataset = dataset.padded_batch(batch_size,
                                 padded_shapes=padded_shapes,
                                 padding_values=padding_values,
                                 drop_remainder=True)
  dataset = dataset.prefetch(options.prefetch_buffer_size)
  return dataset


def get_input_fn(options, is_training):
  """Returns a function that generate input examples.

  Args:
    options: an instance of reader_pb2.Reader.
    is_training: If true, shuffle the dataset.

  Returns:
    input_fn: a callable that returns a dataset.
  """
  if not isinstance(options, reader_pb2.VCRReader):
    raise ValueError('options has to be an instance of Reader.')

  def _input_fn(input_pipeline_context=None):
    """Returns a python dictionary.

    Returns:
      A dataset that can be fed to estimator.
    """
    return _create_dataset(options, is_training, input_pipeline_context)

  return _input_fn
