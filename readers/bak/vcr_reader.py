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

  img_format = 'image/format'
  img_encoded = 'image/encoded'
  img_bbox_label = "image/object/bbox/label"
  img_bbox_score = "image/object/bbox/score"
  img_bbox_feature = "image/object/bbox/feature"
  img_bbox_scope = "image/object/bbox/"
  img_bbox_field_keys = ['ymin', 'xmin', 'ymax', 'xmax']

  cls_bert = 'cls_bert'

  question = 'question'
  question_tag = 'question_tag'
  question_bert = 'question_bert'

  answer_choice = 'answer_choice'
  answer_choice_tag = 'answer_choice_tag'
  answer_choice_bert = 'answer_choice_bert'


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
  cls_bert = 'cls_bert'
  question = 'question'
  question_tag = 'question_tag'
  question_len = 'question_len'
  question_bert = 'question_bert'
  answer_choices = 'answer_choices'
  answer_choices_tag = 'answer_choices_tag'
  answer_choices_len = 'answer_choices_len'
  answer_choices_bert = 'answer_choices_bert'
  answer_choices_with_question = 'answer_choices_with_question'
  answer_choices_with_question_tag = 'answer_choices_with_question_tag'
  answer_choices_with_question_len = 'answer_choices_with_question_len'

  # Bert features of both question and answer choices.
  question_bert = 'question_bert'


def _pad_sequences(sequences, pad=PAD):
  """Pads sequences to the max-length.

  Args:
    sequences: A list of 1-D tensor of size num_sequences, each elem in
      the 1-D tensor denotes a sequence.

  Returns:
    padded_sequences: A [num_sequences, max_sequence_len] tensor.
    lengths: A [num_sequences] int tensor.
  """
  lengths = [tf.shape(x)[0] for x in sequences]
  padded_size = tf.reduce_max(lengths)
  padded_sequences = tf.stack([
      tf.pad(x,
             paddings=[[0, padded_size - lengths[i]]],
             mode='CONSTANT',
             constant_values=pad) for i, x in enumerate(sequences)
  ])
  return padded_sequences, lengths


def _pad_feature_sequences(sequences, pad=PAD, feature_dims=768):
  """Pads sequences to the max-length.

  Args:
    sequences: A list of 1-D tensor of size num_sequences, each elem in
      the 1-D tensor denotes a sequence.

  Returns:
    padded_sequences: A [num_sequences, max_sequence_len] tensor.
    lengths: A [num_sequences] int tensor.
  """
  lengths = [tf.shape(x)[0] for x in sequences]
  padded_size = tf.reduce_max(lengths)
  padded_sequences = tf.stack([
      tf.pad(x,
             paddings=[[0, padded_size - lengths[i]], [0, 0]],
             mode='CONSTANT',
             constant_values=pad) for i, x in enumerate(sequences)
  ])
  return padded_sequences, lengths


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
  question_tag = decoded_example[InputFields.question_tag]

  # Answer choices and lengths.
  answer_choices_list = [
      decoded_example.pop(TFExampleFields.answer_choice + '_%i' % i)
      for i in range(1, 1 + NUM_CHOICES)
  ]
  answer_choices_with_question_list = [
      tf.concat([['[CLS]'], question, ['[SEP]'], x, ['[SEP]']], 0)
      for x in answer_choices_list
  ]
  (answer_choices, answer_choices_len) = _pad_sequences(answer_choices_list)
  (answer_choices_with_question, answer_choices_with_question_len
  ) = _pad_sequences(answer_choices_with_question_list)

  # Answer tags.
  answer_choices_tag_list = [
      decoded_example.pop(TFExampleFields.answer_choice_tag + '_%i' % i)
      for i in range(1, 1 + NUM_CHOICES)
  ]
  answer_choices_with_question_tag_list = [
      tf.concat([[-1], question_tag, [-1], x, [-1]], 0)
      for x in answer_choices_tag_list
  ]
  answer_choices_tag, _ = _pad_sequences(answer_choices_tag_list, -1)
  answer_choices_with_question_tag, _ = _pad_sequences(
      answer_choices_with_question_tag_list, -1)

  # Question bert.
  question_bert_list = [
      tf.reshape(decoded_example.pop(TFExampleFields.question_bert + '_%i' % i),
                 [-1, options.bert_feature_dims])
      for i in range(1, 1 + NUM_CHOICES)
  ]
  answer_choice_bert_list = [
      tf.reshape(
          decoded_example.pop(TFExampleFields.answer_choice_bert + '_%i' % i),
          [-1, options.bert_feature_dims]) for i in range(1, 1 + NUM_CHOICES)
  ]
  question_bert, _ = _pad_feature_sequences(question_bert_list, 0,
                                            options.bert_feature_dims)
  answer_choices_bert, _ = _pad_feature_sequences(answer_choice_bert_list,
                                                  options.bert_feature_dims)

  # CLS bert.
  cls_bert_list = [
      decoded_example.pop(TFExampleFields.cls_bert + '_%i' % i)
      for i in range(1, 1 + NUM_CHOICES)
  ]
  cls_bert = tf.stack(cls_bert_list, axis=0)

  decoded_example.update({
      InputFields.num_objects:
          num_objects,
      InputFields.object_features:
          object_features,
      InputFields.question_tag:
          question_tag,
      InputFields.question_bert:
          question_bert,
      InputFields.question_len:
          question_len,
      InputFields.answer_choices:
          answer_choices,
      InputFields.answer_choices_tag:
          answer_choices_tag,
      InputFields.answer_choices_bert:
          answer_choices_bert,
      InputFields.answer_choices_len:
          answer_choices_len,
      InputFields.answer_choices_with_question:
          answer_choices_with_question,
      InputFields.answer_choices_with_question_tag:
          answer_choices_with_question_tag,
      InputFields.answer_choices_with_question_len:
          answer_choices_with_question_len,
      InputFields.cls_bert: cls_bert,
  })

  # Image shape.
  if InputFields.img_data in decoded_example:
    image = decoded_example[InputFields.img_data]
    # image = _resize_image(image)

    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]
    decoded_example.update({
        InputFields.img_data: image,
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
      TFExampleFields.question_tag: tf.io.VarLenFeature(tf.int64),
  }
  for bbox_key in TFExampleFields.img_bbox_field_keys:
    bbox_field = os.path.join(TFExampleFields.img_bbox_scope, bbox_key)
    keys_to_features[bbox_field] = tf.io.VarLenFeature(tf.float32)
  for i in range(1, 1 + NUM_CHOICES):
    keys_to_features.update({
        TFExampleFields.cls_bert + '_%i' % i:
            tf.io.VarLenFeature(tf.float32),
        TFExampleFields.question_bert + '_%i' % i:
            tf.io.VarLenFeature(tf.float32),
        TFExampleFields.answer_choice + '_%i' % i:
            tf.io.VarLenFeature(tf.string),
        TFExampleFields.answer_choice_tag + '_%i' % i:
            tf.io.VarLenFeature(tf.int64),
        TFExampleFields.answer_choice_bert + '_%i' % i:
            tf.io.VarLenFeature(tf.float32)
    })

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
          tfexample_decoder.BoundingBox(
              keys=TFExampleFields.img_bbox_field_keys,
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
      InputFields.question_tag:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.question_tag,
                                   default_value=-1),
      TFExampleFields.img_bbox_feature:
          tfexample_decoder.Tensor(tensor_key=TFExampleFields.img_bbox_feature,
                                   default_value=0),
  }

  for i in range(1, 1 + NUM_CHOICES):
    tensor_key = TFExampleFields.cls_bert + '_%i' % i
    items_to_handlers[tensor_key] = tfexample_decoder.Tensor(
        tensor_key=tensor_key, default_value=0)
    tensor_key = TFExampleFields.question_bert + '_%i' % i
    items_to_handlers[tensor_key] = tfexample_decoder.Tensor(
        tensor_key=tensor_key, default_value=0)
    tensor_key = TFExampleFields.answer_choice + '_%i' % i
    items_to_handlers[tensor_key] = tfexample_decoder.Tensor(
        tensor_key=tensor_key, default_value=PAD)
    tensor_key = TFExampleFields.answer_choice_tag + '_%i' % i
    items_to_handlers[tensor_key] = tfexample_decoder.Tensor(
        tensor_key=tensor_key, default_value=-1)
    tensor_key = TFExampleFields.answer_choice_bert + '_%i' % i
    items_to_handlers[tensor_key] = tfexample_decoder.Tensor(
        tensor_key=tensor_key, default_value=0)
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
      InputFields.cls_bert: [NUM_CHOICES, options.bert_feature_dims],
      InputFields.question: [None],
      InputFields.question_tag: [None],
      InputFields.question_bert: [NUM_CHOICES, None, options.bert_feature_dims],
      InputFields.question_len: [],
      InputFields.answer_choices: [NUM_CHOICES, None],
      InputFields.answer_choices_tag: [NUM_CHOICES, None],
      InputFields.answer_choices_bert: [
          NUM_CHOICES, None, options.bert_feature_dims
      ],
      InputFields.answer_choices_len: [NUM_CHOICES],
      InputFields.answer_choices_with_question: [NUM_CHOICES, None],
      InputFields.answer_choices_with_question_tag: [NUM_CHOICES, None],
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
      InputFields.cls_bert: 0.0,
      InputFields.question: PAD,
      InputFields.question_tag: -1,
      InputFields.question_bert: 0.0,
      InputFields.question_len: 0,
      InputFields.answer_choices: PAD,
      InputFields.answer_choices_tag: -1,
      InputFields.answer_choices_bert: 0.0,
      InputFields.answer_choices_len: 0,
      InputFields.answer_choices_with_question: PAD,
      InputFields.answer_choices_with_question_tag: -1,
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
