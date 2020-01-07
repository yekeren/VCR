from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf
from modeling.layers import fast_rcnn

from google.protobuf import text_format
from protos import fast_rcnn_pb2

logging.set_verbosity(logging.DEBUG)


class FastRCNNLayerTest(tf.test.TestCase):

  def test_fast_rcnn_layer(self):
    options_str = r"""
      feature_extractor {
        type: 'faster_rcnn_inception_v2'
        first_stage_features_stride: 16
      }
      initial_crop_size: 14
      maxpool_kernel_size: 2
      maxpool_stride: 2
      dropout_keep_prob: 0.5
      dropout_on_feature_map: false
      checkpoint_path: 'data/detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt'
    """
    options = text_format.Merge(options_str, fast_rcnn_pb2.FastRCNN())

    test_layer = fast_rcnn.FastRCNNLayer(options)

    inputs = tf.random.uniform(shape=[5, 320, 320, 3], maxval=255)
    proposals = tf.constant([[[0, 0, 1, 1]]] * 5, dtype=tf.float32)
    output = test_layer(inputs, proposals)
    self.assertAllEqual(output.shape, [5, 1, 1024])


if __name__ == '__main__':
  tf.test.main()
