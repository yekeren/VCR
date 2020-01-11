from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import tensorflow as tf

from tf_slim import tfexample_decoder
from protos import reader_pb2

PAD = '[PAD]'
NUM_CHOICES = 4


class TFExampleFields(object):
  """Fields in the tf.train.Example."""
  img_id = 'img_id'
  annot_id = 'annot_id'
  answer_label = 'answer_label'

  img_encoded = 'image/encoded'
  img_format = 'image/format'
  img_bbox_scope = "image/object/bbox/"
  img_bbox_keys = ['ymin', 'xmin', 'ymax', 'xmax']
  img_bbox_label = "image/object/bbox/label"
  img_bbox_score = "image/object/bbox/score"
  img_bbox_feature = "image/object/bbox/feature"

  question = 'question'
  answer_choice_1 = 'answer_choice_1'
  answer_choice_2 = 'answer_choice_2'
  answer_choice_3 = 'answer_choice_3'
  answer_choice_4 = 'answer_choice_4'
  answer_choices = [
      answer_choice_1, answer_choice_2, answer_choice_3, answer_choice_4
  ]


class InputFields(object):
  """Names of the input tensors."""
  # Meta information.
  img_id = 'img_id'
  annot_id = 'annot_id'
  answer_label = 'answer_label'

  # Image data.
  img_data = 'img_data'
  img_height = 'img_height'
  img_width = 'img_width'

  # Objects.
  num_objects = 'num_objects'
  object_bboxes = 'object_bboxes'
  object_labels = 'object_labels'
  object_scores = 'object_scores'
  object_features = 'object_features'

  # Question and answer choices.
  question = 'question'
  question_len = 'question_len'
  answer_choices = 'answer_choices'
  answer_choices_len = 'answer_choices_len'
  answer_choices_with_question = 'answer_choices_with_question'
  answer_choices_with_question_len = 'answer_choices_with_question_len'


def _pad_sentences(sentences):
  """Pads sentences to the max-length.

  Args:
    sentences: A list of 1-D string tensor of size num_sentences, each elem in
      the 1-D tensor denotes a sentence.

  Returns:
    padded_sentences: A [num_sentences, max_sentence_len] string tensor.
    lengths: A [num_sentences] int tensor.
  """
  lengths = [tf.shape(x)[0] for x in sentences]
  padded_size = tf.reduce_max(lengths)
  padded_sentences = tf.stack([
      tf.pad(x,
             paddings=[[0, padded_size - lengths[i]]],
             mode='CONSTANT',
             constant_values=PAD) for i, x in enumerate(sentences)
  ])
  return padded_sentences, lengths


def _update_decoded_example(decoded_example, options):
  """Updates the decoded example, add size to the varlen feature.

  Args:
    decoded_example: A tensor dictionary keyed by name.
    options: An instance of reader_pb2.Reader.

  Returns:
    decoded_example: The same instance with content modified.
  """
  # Number of objects.
  object_bboxes = decoded_example[InputFields.object_bboxes]
  num_objects = tf.shape(object_bboxes)[0]

  # Object Fast-RCNN features.
  object_features = decoded_example.pop(TFExampleFields.img_bbox_feature)
  object_features = tf.reshape(object_features,
                               [-1, options.frcnn_feature_dims])

  # Question length.
  question = decoded_example[InputFields.question]
  question_len = tf.shape(question)[0]

  # Answer choices and lengths.
  answer_choices_list = [
      decoded_example.pop(field) for field in TFExampleFields.answer_choices
  ]
  answer_choices_with_question_list = [
      tf.concat([['[CLS]'], question, ['[SEP]'], x, ['[SEP]']], 0)
      for x in answer_choices_list
  ]
  (answer_choices, answer_choices_len) = _pad_sentences(answer_choices_list)
  (answer_choices_with_question, answer_choices_with_question_len
  ) = _pad_sentences(answer_choices_with_question_list)

  decoded_example.update({
      InputFields.num_objects:
          num_objects,
      InputFields.object_features:
          object_features,
      InputFields.question_len:
          question_len,
      InputFields.answer_choices:
          answer_choices,
      InputFields.answer_choices_len:
          answer_choices_len,
      InputFields.answer_choices_with_question:
          answer_choices_with_question,
      InputFields.answer_choices_with_question_len:
          answer_choices_with_question_len,
  })

  # Image shape.
  if InputFields.img_data in decoded_example:
    image = decoded_example[InputFields.img_data]
    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]
    decoded_example.update({
        InputFields.img_height: height,
        InputFields.img_width: width,
    })

  return decoded_example


def _parse_single_example(example, options):
  """Parses a single tf.Example proto.

  Args:
    example: An Example proto.
    options: An instance of reader_pb2.Reader.

  Returns:
    A dictionary indexed by tensor name.
  """
  # Initialize `keys_to_features`.
  keys_to_features = {
      TFExampleFields.img_id: tf.io.FixedLenFeature([], tf.string),
      TFExampleFields.annot_id: tf.io.FixedLenFeature([], tf.string),
      TFExampleFields.answer_label: tf.io.FixedLenFeature([], tf.int64),
      TFExampleFields.img_bbox_label: tf.io.VarLenFeature(tf.string),
      TFExampleFields.img_bbox_score: tf.io.VarLenFeature(tf.float32),
      TFExampleFields.img_bbox_feature: tf.io.VarLenFeature(tf.float32),
      TFExampleFields.question: tf.io.VarLenFeature(tf.string),
  }
  for bbox_key in TFExampleFields.img_bbox_keys:
    field = os.path.join(TFExampleFields.img_bbox_scope, bbox_key)
    keys_to_features[field] = tf.io.VarLenFeature(tf.float32)
  for field in TFExampleFields.answer_choices:
    keys_to_features[field] = tf.io.VarLenFeature(tf.string)

  # Initialize `items_to_handlers`.
  items_to_handlers = {
      InputFields.img_id:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.img_id,
                                   default_value=''),
      InputFields.annot_id:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.annot_id,
                                   default_value=''),
      InputFields.answer_label:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.answer_label,
                                   default_value=-1),
      InputFields.object_bboxes:
          tfexample_decoder.BoundingBox(keys=TFExampleFields.img_bbox_keys,
                                        prefix=TFExampleFields.img_bbox_scope),
      InputFields.object_labels:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.img_bbox_label,
                                   default_value=''),
      InputFields.object_scores:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.img_bbox_score,
                                   default_value=0),
      InputFields.question:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.question,
                                   default_value=PAD),
      TFExampleFields.img_bbox_feature:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.img_bbox_feature,
                                   default_value=0),
  }
  if options.decode_jpeg:
    keys_to_features.update({
        TFExampleFields.img_encoded: tf.io.FixedLenFeature([], tf.string),
        TFExampleFields.img_format: tf.io.FixedLenFeature([], tf.string),
    })
    items_to_handlers.update({
        InputFields.img_data:
            tfexample_decoder.Image(image_key=TFExampleFields.img_encoded,
                                    format_key=TFExampleFields.img_format,
                                    shape=None)
    })

  for field in TFExampleFields.answer_choices:
    items_to_handlers[field] = tfexample_decoder.Tensor(tensor_key=field,
                                                        default_value=PAD)

  # Decode example.
  example_decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                       items_to_handlers)

  output_keys = example_decoder.list_items()
  output_tensors = example_decoder.decode(example)
  output_tensors = [
      x if x.dtype != tf.int64 else tf.cast(x, tf.int32) for x in output_tensors
  ]
  decoded_example = dict(zip(output_keys, output_tensors))
  return _update_decoded_example(decoded_example, options)


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
    if options.cache_dataset:
      dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.shuffle(options.shuffle_buffer_size)
  dataset = dataset.interleave(tf.data.TFRecordDataset,
                               cycle_length=options.interleave_cycle_length)

  parse_fn = lambda x: _parse_single_example(x, options)
  dataset = dataset.map(map_func=parse_fn,
                        num_parallel_calls=options.num_parallel_calls)

  padded_shapes = {
      InputFields.img_id: [],
      InputFields.annot_id: [],
      InputFields.answer_label: [],
      InputFields.num_objects: [],
      InputFields.object_bboxes: [None, 4],
      InputFields.object_labels: [None],
      InputFields.object_scores: [None],
      InputFields.object_features: [None, options.frcnn_feature_dims],
      InputFields.question: [None],
      InputFields.question_len: [],
      InputFields.answer_choices: [NUM_CHOICES, None],
      InputFields.answer_choices_len: [NUM_CHOICES],
      InputFields.answer_choices_with_question: [NUM_CHOICES, None],
      InputFields.answer_choices_with_question_len: [NUM_CHOICES],
  }
  padding_values = {
      InputFields.img_id: '',
      InputFields.annot_id: '',
      InputFields.answer_label: -1,
      InputFields.num_objects: 0,
      InputFields.object_bboxes: 0.0,
      InputFields.object_labels: '',
      InputFields.object_scores: 0.0,
      InputFields.object_features: 0.0,
      InputFields.question: PAD,
      InputFields.question_len: 0,
      InputFields.answer_choices: PAD,
      InputFields.answer_choices_len: 0,
      InputFields.answer_choices_with_question: PAD,
      InputFields.answer_choices_with_question_len: 0,
  }
  if options.decode_jpeg:
    padded_shapes.update({
        InputFields.img_data: [None, None, 3],
        InputFields.img_height: [],
        InputFields.img_width: [],
    })
    padding_values.update({
        InputFields.img_data: tf.constant(0, dtype=tf.uint8),
        InputFields.img_height: 0,
        InputFields.img_width: 0,
    })
  dataset = dataset.padded_batch(batch_size,
                                 padded_shapes=padded_shapes,
                                 padding_values=padding_values,
                                 drop_remainder=True)
  dataset = dataset.prefetch(options.prefetch_buffer_size)
  return dataset


def get_input_fn(options, is_training):
  """Returns a function that generate input examples.

  Args:
    options: An instance of reader_pb2.Reader.
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
