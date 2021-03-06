from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf

import reader
from vcr_text_only_reader import InputFields

from google.protobuf import text_format
from protos import reader_pb2

tf.compat.v1.enable_eager_execution()


class VCRTextOnlyReaderTest(tf.test.TestCase):

  def test_get_input_fn(self):
    options_str = r"""
      vcr_text_only_reader {
        input_pattern: "output/uncased/VCR-text_only/val.record-*-of-00005"
        shuffle_buffer_size: 10
        interleave_cycle_length: 1
        batch_size: 3
        prefetch_buffer_size: 8000
      }
    """
    options = text_format.Merge(options_str, reader_pb2.Reader())

    dataset = reader.get_input_fn(options, is_training=False)()
    for elem in dataset.take(1):
      for key, value in elem.items():
        logging.info('=' * 64)
        logging.info(key)
        logging.info(value)
      import pdb
      pdb.set_trace()
      j = 1


if __name__ == '__main__':
  tf.test.main()
