from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf

import reader
from vcr_reader import InputFields

from google.protobuf import text_format
from protos import reader_pb2

tf.compat.v1.enable_eager_execution()


class VCRReaderTest(tf.test.TestCase):

  def test_get_input_fn(self):
    options_str = r"""
      vcr_reader {
        input_pattern: "/own_files/yekeren/VCR-relevant-only/val.record-*-of-00005"
        shuffle_buffer_size: 10
        interleave_cycle_length: 1
        batch_size: 20
        prefetch_buffer_size: 8000
        frcnn_feature_dims: 1088
        bert_feature_dims: 768
        decode_jpeg: false
      }
    """
    options = text_format.Merge(options_str, reader_pb2.Reader())

    dataset = reader.get_input_fn(options, is_training=False)()
    for elem in dataset.take(1):
      self.assertNotIn(InputFields.img_data, elem)
      self.assertNotIn(InputFields.img_width, elem)
      self.assertNotIn(InputFields.img_height, elem)

      (num_objects, object_bboxes, object_labels, object_scores,
       object_features) = (elem[InputFields.num_objects],
                           elem[InputFields.object_bboxes],
                           elem[InputFields.object_labels],
                           elem[InputFields.object_scores],
                           elem[InputFields.object_features])

      import pdb
      pdb.set_trace()
      for key, value in elem.items():
        logging.info('=' * 64)
        logging.info(key)
        logging.info(value)


if __name__ == '__main__':
  tf.test.main()
