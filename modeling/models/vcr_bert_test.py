from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np
import tensorflow as tf
from modeling.layers import token_to_id
from official.nlp.bert_modeling import BertConfig
from official.nlp.bert_models import _get_transformer_encoder as get_transformer_encoder
from official.nlp.modeling.networks import bert_pretrainer
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


def load_vocab(filename):
  with open(filename, 'r', encoding='utf8') as f:
    return [x.strip('\n') for x in f]


class VCRBertTest(tf.test.TestCase):

  def test_masked_lm(self):
    example_sentence = [
        'alice', 'became', '[MASK]', 'after', 'felt', 'left', 'out', 'by',
        'her', 'friends.'
    ]
    num_token_predictions = 1
    lm_mask = [2]

    bert_unk_token_id = 100
    bert_dir = 'data/bert/keras/cased_L-12_H-768_A-12'
    bert_vocab_file = "{}/vocab.txt".format(bert_dir)
    bert_config_file = "{}/bert_config.json".format(bert_dir)
    bert_checkpoint_file = "{}/bert_model.ckpt".format(bert_dir)
    num_classes = 2
    sequence_length = len(example_sentence)
    vocab = load_vocab(bert_vocab_file)

    token_to_id_layer = token_to_id.TokenToIdLayer(bert_vocab_file,
                                                   bert_unk_token_id)

    bert_config = BertConfig.from_json_file(bert_config_file)
    transformer_encoder = get_transformer_encoder(bert_config, sequence_length)

    pretrainer_model = bert_pretrainer.BertPretrainer(
        network=transformer_encoder,
        num_classes=num_classes,
        num_token_predictions=num_token_predictions,
        output='predictions')

    checkpoint = tf.train.Checkpoint(model=transformer_encoder)
    status = checkpoint.restore(bert_checkpoint_file)

    with tf.compat.v1.Session() as sess:
      status.initialize_or_restore(sess)
      values = sess.run(transformer_encoder.trainable_variables)
      print(values[-1])
      j = 1

    # word_ids = token_to_id_layer(
    #     tf.constant([example_sentence], dtype=tf.string))
    # mask = tf.constant([[1] * len(example_sentence)], dtype=tf.int32)
    # type_ids = tf.constant([[0] * len(example_sentence)], dtype=tf.int32)
    # lm_mask = tf.constant([lm_mask], dtype=tf.int32)
    # results = pretrainer_model([word_ids, mask, type_ids, lm_mask],
    #                            training=False)
    # results = results[0].numpy().flatten()

    # indices = np.argsort(results)[::-1]
    # logging.info(' '.join(example_sentence))
    # for i in indices[:20]:
    #   logging.info('%s %.4lf', vocab[i], results[i])


if __name__ == '__main__':
  tf.test.main()
