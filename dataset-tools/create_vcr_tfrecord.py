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
                    'data/bert/keras/cased_L-12_H-768_A-12/vocab.txt',
                    'Path to the Bert vocabulary file.')

flags.DEFINE_boolean('do_lower_case', False,
                     'To be passed to the bert tokenizer.')

flags.DEFINE_string('annotations_jsonl_file', 'data/vcr1annots/val.jsonl',
                    'Path to the annotations file in jsonl format.')

flags.DEFINE_integer('num_shards', 10,
                     'Number of shards of the output tfrecord files.')

flags.DEFINE_string('output_tfrecord_path', '/own_files/yekeren/VCR2/val.record',
                    'Path to the output tfrecord files.')

flags.DEFINE_string('image_zip_file', 'data/vcr1images.zip',
                    'Path to the zip file of images.')

flags.DEFINE_integer('image_max_size', 600, 'Maximum size of the image.')

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


def _create_tf_example(encoded_jpg,
                       annot,
                       meta,
                       bert_tokenizer,
                       do_lower_case,
                       image_max_size=None):
  """Creates an example from the annotation.

  Args:
    encoded_jpg: A python string, the encoded jpeg data.
    annot: A python dictionary parsed from the json object.
    bert_tokenizer: A tokenization.FullTokenizer object.
    do_lower_case: If true, convert text to lower case.
    image_max_size: If set, resize the larger size to this value.

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

  # Encode jpg data.
  image_height, image_width = meta['height'], meta['width']

  if image_max_size is not None:
    image = PIL.Image.open(io.BytesIO(encoded_jpg))
    assert (image.height == image_height and image.width == image_width and
            image.format == 'JPEG')

    image_scale = image_max_size / max(image_height, image_width)

    image_height, image_width = (int(image_height * image_scale),
                                 int(image_width * image_scale))

    image = image.resize((image_width, image_height))
    with io.BytesIO() as output:
      image.save(output, format="JPEG")
      encoded_jpg = output.getvalue()

  feature['image/height'] = _int64_feature(image_height)
  feature['image/width'] = _int64_feature(image_width)
  feature['image/format'] = _bytes_feature('jpeg')
  feature['image/encoded'] = tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[encoded_jpg]))

  # Encode objects and boxes.
  assert meta['names'] == annot[
      'objects'], 'Meta data does not match the annotation.'
  obj_to_type = annot['objects']

  boxes = np.array(meta['boxes'])
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
  feature['image/object/bbox/label'] = _bytes_feature_list(obj_to_type)

  # Encode question and answer choices.
  question = annot['question']
  answer_choices = annot['answer_choices']
  assert NUM_CHOICES == len(answer_choices)

  def _convert_tokens_to_ids(tokens):
    output = []
    for token in tokens:
      if token in bert_tokenizer.vocab:
        output.append(bert_tokenizer.vocab[token])
      else:
        output.append(bert_tokenizer.vocab["[UNK]"])
    return output

  # Encode question.
  question_tokens, question_tags = _fix_tokenization(question, obj_to_type,
                                                     bert_tokenizer,
                                                     do_lower_case)
  feature['question'] = _bytes_feature_list(question_tokens)
  feature['question_tag'] = _int64_feature_list(question_tags)

  # Encode answer choices.
  for idx, tokenized_sent in enumerate(answer_choices):
    tokens, tags = _fix_tokenization(tokenized_sent, obj_to_type,
                                     bert_tokenizer, do_lower_case)
    feature['answer_choice_%i' % (idx + 1)] = _bytes_feature_list(tokens)
    feature['answer_choice_tag_%i' % (idx + 1)] = _int64_feature_list(tags)

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

  num_shards = FLAGS.num_shards
  output_tfrecord_path = FLAGS.output_tfrecord_path
  writers = [
      tf.io.TFRecordWriter(output_tfrecord_path + '-%05d-of-%05d' %
                           (idx, num_shards)) for idx in range(num_shards)
  ]

  with zipfile.ZipFile(FLAGS.image_zip_file) as image_zip:
    for idx, annot in enumerate(annots):
      if (idx + 1) % 1000 == 0:
        logging.info('On example %i/%i.', idx + 1, len(annots))

      # Read image data.
      img_fn = os.path.join('vcr1images', annot['img_fn'])
      try:
        with image_zip.open(img_fn, 'r') as f:
          encoded_jpg = f.read()
      except Exception as ex:
        logging.warn('Skip %s.', img_fn)
        continue

      # Read meta data.
      meta_fn = os.path.join('vcr1images', annot['metadata_fn'])
      try:
        with image_zip.open(meta_fn, 'r') as f:
          meta = json.load(f)
      except Exception as ex:
        logging.warn('Skip %s.', meta_fn)
        continue

      # Create TF example.
      tf_example = _create_tf_example(encoded_jpg, annot, meta, bert_tokenizer,
                                      FLAGS.do_lower_case, FLAGS.image_max_size)
      annot_id = int(annot['annot_id'].split('-')[-1])
      writers[annot_id % num_shards].write(tf_example.SerializeToString())

  for writer in writers:
    writer.close()

  logging.info('Done')


if __name__ == '__main__':
  app.run(main)
