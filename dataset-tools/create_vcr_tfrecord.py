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

flags.DEFINE_string('bert_vocab_file',
                    'data/bert/tf1.x/cased_L-12_H-768_A-12/vocab.txt',
                    'Path to the Bert vocabulary file.')

flags.DEFINE_boolean('do_lower_case', False,
                     'To be passed to the bert tokenizer.')

flags.DEFINE_string('annotations_jsonl_file', 'data/vcr1annots/val.jsonl',
                    'Path to the annotations file in jsonl format.')

flags.DEFINE_integer('num_shards', 10,
                     'Number of shards of the output tfrecord files.')

flags.DEFINE_integer('shard_id', 0, 'Shard id of the current process.')

flags.DEFINE_string('output_tfrecord_path', 'output/val.record',
                    'Path to the output tfrecord files.')

flags.DEFINE_string('image_zip_file', '/own_files/yekeren/vcr1images.zip',
                    'Path to the zip file of images.')

flags.DEFINE_string('frcnn_feature_dir',
                    'output/fast_rcnn/inception_resnet_v2_imagenet',
                    'Path to the directory saving FRCNN features.')

flags.DEFINE_string('bert_feature_dir',
                    '/own_files/yekeren/bert/cased_L-12_H-768_A-12',
                    'Path to the directory saving Bert features.')

flags.DEFINE_boolean('only_use_relevant_dets', False,
                     'If true, only use relevant detections.')

flags.DEFINE_boolean('encode_jpeg', True,
                     'If true, add jpeg encoded image to the tf example.')

flags.DEFINE_integer('desired_height', 384, 'Desired height of the images.')

flags.DEFINE_integer('desired_width', 768, 'Desired width of the images.')

FLAGS = flags.FLAGS

# GENDER_NEUTRAL_NAMES = [
#     'Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry',
#     'Jody', 'Kendall', 'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn'
# ]
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


def _fix_tokenization(tokenized_sent, obj_to_type, old_det_to_new_ind,
                      bert_tokenizer, do_lower_case):
  """Converts tokenized annotations into tokenized sentence.

  Args:
    tokenized_sent: Tokenized sentence with detections collapsed to a list.
    obj_to_type: [person, person, pottedplant] indexed by the labels.
    old_det_to_new_ind: A mapping from the old indices to the new indices.

  Returns:
    tokenized_sent: A list of string tokens.
  """
  case_fn = lambda x: x.lower() if do_lower_case else x

  new_tokenization_with_tags = []
  for tok in tokenized_sent:
    if isinstance(tok, list):
      for idx in tok:
        if old_det_to_new_ind is not None:
          idx = old_det_to_new_ind[idx]

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


def get_detections_to_use(obj_to_type, tokens_mixed_with_tags):
  """Gets the detections to use, filtering out objects that are not mentioned.

  Args:
    obj_to_type: A list of object names.
    tokens_mixed_with_tags: Sentences that refer to the object list.

  Returns:
    indices: Indices in the obj_to_type, denoting the mentioned objects.
  """
  detections_to_use = np.zeros(len(obj_to_type), dtype=bool)
  people = np.array([x == 'person' for x in obj_to_type], dtype=bool)

  for sentence in tokens_mixed_with_tags:
    for possibly_det_list in sentence:
      if isinstance(possibly_det_list, list):
        for tag in possibly_det_list:
          assert 0 <= tag < len(obj_to_type)
          detections_to_use[tag] = True
      elif possibly_det_list.lower() in ['everyone', 'everyones']:
        detections_to_use |= people
  if not detections_to_use.any():
    detections_to_use |= people
  detections_to_use = np.where(detections_to_use)[0]

  old_det_to_new_ind = np.zeros(len(obj_to_type), dtype=np.int32) - 1
  old_det_to_new_ind[detections_to_use] = np.arange(detections_to_use.shape[0],
                                                    dtype=np.int32)
  return detections_to_use, old_det_to_new_ind


def _create_tf_example(encoded_jpg, annot, meta, image_and_rcnn_features,
                       bert_features, bert_tokenizer, do_lower_case,
                       encode_jpeg, desired_height, desired_width,
                       only_use_relevant_dets):
  """Creates an example from the annotation.

  Args:
    encoded_jpg: A python string, the encoded jpeg data.
    annot: A python dictionary parsed from the json object.
    meta: A python dictionary containing object information.
    image_and_rcnn_features: A numpy array containing box features.
    bert_features: A numpy array containing bert features.
    bert_tokenizer: A tokenization.FullTokenizer object.
    do_lower_case: If true, convert text to lower case.
    encode_jpeg: If true, encode the jpeg to the tf example.
    desired_height: Desired image height.
    desired_width: Desired image width.

  Returns:
    tf_example: A tf.train.Example proto.
  """

  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[value.encode('utf8')]))

  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

  def _bytes_feature_list(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[x.encode('utf8') for x in value]))

  def _int64_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def _float_feature_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  feature = {}
  for key, value in annot.items():
    if isinstance(value, int):
      feature[key] = _int64_feature(value)
    elif isinstance(value, str):
      feature[key] = _bytes_feature(value)

  # Encode objects and boxes.
  image_height, image_width = meta['height'], meta['width']
  assert (meta['names'] == annot['objects']
         ), 'Meta data does not match the annotation.'

  obj_to_type = annot['objects']
  question = annot['question']
  answer_choices = annot['answer_choices']
  assert NUM_CHOICES == len(answer_choices)

  obj_to_type = np.array(obj_to_type)
  boxes = np.array(meta['boxes'])
  rcnn_features = image_and_rcnn_features[1:]

  old_det_to_new_ind = None
  if only_use_relevant_dets:
    detections_to_use, old_det_to_new_ind = get_detections_to_use(
        obj_to_type, [question] + answer_choices)
    old_det_to_new_ind = [-1 if x < 0 else x + 1 for x in old_det_to_new_ind]

    # Gather elements using the new indices.
    obj_to_type = obj_to_type[detections_to_use]
    boxes = boxes[detections_to_use]
    rcnn_features = rcnn_features[detections_to_use]
  else:
    old_det_to_new_ind = np.arange(len(obj_to_type), dtype=np.int32) + 1

  # Add [0, 0, height, width] as for the full image.
  obj_to_type = np.concatenate([['[IMAGE]'], obj_to_type], 0)
  boxes = np.concatenate([[[0, 0, image_width, image_height, 1]], boxes], 0)
  rcnn_features = np.concatenate([image_and_rcnn_features[:1], rcnn_features],
                                 0)

  xmin, ymin, xmax, ymax, score = [boxes[:, i] for i in range(5)]
  xmin /= image_width
  ymin /= image_height
  xmax /= image_width
  ymax /= image_height
  feature['image/object/bbox/xmin'] = _float_feature_list(xmin.tolist())
  feature['image/object/bbox/ymin'] = _float_feature_list(ymin.tolist())
  feature['image/object/bbox/xmax'] = _float_feature_list(xmax.tolist())
  feature['image/object/bbox/ymax'] = _float_feature_list(ymax.tolist())
  feature['image/object/bbox/score'] = _float_feature_list(score.tolist())
  feature['image/object/bbox/label'] = _bytes_feature_list(obj_to_type.tolist())
  feature['image/object/bbox/feature'] = _float_feature_list(
      rcnn_features.flatten().tolist())

  # Encode jpg data, resize image if specified.
  if encode_jpeg:
    image = PIL.Image.open(io.BytesIO(encoded_jpg))
    assert image.format == 'JPEG'

    width_scale = desired_width / image.width
    height_scale = desired_height / image.height
    image_scale = min(width_scale, height_scale)
    image_height, image_width = (int(image.height * image_scale),
                                 int(image.width * image_scale))

    image = image.resize((image_width, image_height))
    with io.BytesIO() as output:
      image.save(output, format="JPEG")
      encoded_jpg = output.getvalue()

    feature['image/height'] = _int64_feature(image_height)
    feature['image/width'] = _int64_feature(image_width)
    feature['image/format'] = _bytes_feature('jpeg')
    feature['image/encoded'] = tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[encoded_jpg]))

  # Encode question.
  question_tokens, question_tags = _fix_tokenization(question, obj_to_type,
                                                     old_det_to_new_ind,
                                                     bert_tokenizer,
                                                     do_lower_case)
  feature['question'] = _bytes_feature_list(question_tokens)
  feature['question_tag'] = _int64_feature_list(question_tags)

  assert NUM_CHOICES == len(bert_features)
  for idx, tokenized_sent in enumerate(answer_choices):
    # Encode answer choices.
    tokens, tags = _fix_tokenization(tokenized_sent, obj_to_type,
                                     old_det_to_new_ind, bert_tokenizer,
                                     do_lower_case)
    feature['answer_choice_%i' % (idx + 1)] = _bytes_feature_list(tokens)
    feature['answer_choice_tag_%i' % (idx + 1)] = _int64_feature_list(tags)

    # Encode bert embeddings,
    # Loaded embedding form: "[CLS] question [SEP] answer [SEP]".
    bert_feature = bert_features[idx]
    assert bert_feature.shape[0] == 3 + len(question_tokens) + len(tokens)
    cls_bert = bert_feature[0]
    question_bert = bert_feature[1:len(question_tokens) + 1]
    answer_choice_bert = bert_feature[len(question_tokens) + 2:-1]

    feature['cls_bert_%i' % (idx + 1)] = _float_feature_list(
        cls_bert.flatten().tolist())
    feature['question_bert_%i' % (idx + 1)] = _float_feature_list(
        question_bert.flatten().tolist())
    feature['answer_choice_bert_%i' % (idx + 1)] = _float_feature_list(
        answer_choice_bert.flatten().tolist())
    feature['answer_choice_with_question_bert_%i' %
            (idx + 1)] = _float_feature_list(bert_feature.flatten().tolist())

  tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
  return tf_example


def main(_):
  logging.set_verbosity(logging.INFO)

  # Create Bert model.
  bert_tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.bert_vocab_file,
                                              do_lower_case=FLAGS.do_lower_case)

  # Load annotations.
  annots = _load_annotations(FLAGS.annotations_jsonl_file)
  logging.info('Loaded %i annotations.', len(annots))

  shard_id, num_shards = FLAGS.shard_id, FLAGS.num_shards
  assert 0 <= shard_id < num_shards

  writer = tf.io.TFRecordWriter(FLAGS.output_tfrecord_path + '-%05d-of-%05d' %
                                (shard_id, num_shards))

  with zipfile.ZipFile(FLAGS.image_zip_file) as image_zip:
    for idx, annot in enumerate(annots):
      if (idx + 1) % 1000 == 0:
        logging.info('On example %i/%i.', idx + 1, len(annots))

      annot_id = int(annot['annot_id'].split('-')[-1])
      if annot_id % num_shards != shard_id:
        continue

      # Read meta data.
      meta_fn = os.path.join('vcr1images', annot['metadata_fn'])
      try:
        with image_zip.open(meta_fn, 'r') as f:
          meta = json.load(f)
      except Exception as ex:
        logging.warn('Skip %s.', meta_fn)
        continue

      # Read image data.
      encoded_jpg = None
      img_fn = os.path.join('vcr1images', annot['img_fn'])
      if FLAGS.encode_jpeg:
        try:
          with image_zip.open(img_fn, 'r') as f:
            encoded_jpg = f.read()
        except Exception as ex:
          logging.warn('Skip %s.', img_fn)
          continue

      # Read RCNN feature.
      part_id = get_partition_id(annot['annot_id'])
      rcnn_fn = os.path.join(FLAGS.frcnn_feature_dir, '%02d' % part_id,
                             annot['annot_id'] + '.npy')
      if not os.path.isfile(rcnn_fn):
        logging.warn('Skip %s.', rcnn_fn)
        continue
      try:
        with open(rcnn_fn, 'rb') as f:
          rcnn_features = np.load(f)
      except Exception as ex:
        logging.warn('Skip %s.', rcnn_fn)
        raise ValueError('!!!!!!!!!!')

      # Read Bert feature.
      part_id = get_partition_id(annot['annot_id'])
      bert_fn = os.path.join(FLAGS.bert_feature_dir, '%02d' % part_id,
                             annot['annot_id'] + '.npy')
      with open(bert_fn, 'rb') as f:
        bert_features = np.load(f, allow_pickle=True)

      # Create TF example.
      tf_example = _create_tf_example(encoded_jpg, annot, meta, rcnn_features,
                                      bert_features, bert_tokenizer,
                                      FLAGS.do_lower_case, FLAGS.encode_jpeg,
                                      FLAGS.desired_height, FLAGS.desired_width,
                                      FLAGS.only_use_relevant_dets)
      writer.write(tf_example.SerializeToString())

  writer.close()

  logging.info('Done')


if __name__ == '__main__':
  app.run(main)
