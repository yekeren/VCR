from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protos import model_pb2
from vcr.bert import VCRBert
from vcr.vbert import VCRVBert
from vcr.bilstm_glove import VCRBiLSTMGloVe

MODELS = {
    model_pb2.VCRBert.ext: VCRBert,
    model_pb2.VCRVBert.ext: VCRVBert,
    model_pb2.VCRBiLSTMGloVe.ext: VCRBiLSTMGloVe,
}


def build(options, is_training):
  """Builds a model based on the options.

  Args:
    options: A model_pb2.Model instance.

  Returns:
    A model instance.

  Raises:
    ValueError: If the model proto is invalid or cannot find a registered entry.
  """
  if not isinstance(options, model_pb2.Model):
    raise ValueError('The options has to be an instance of model_pb2.Model.')

  for extension, model_proto in options.ListFields():
    if extension in MODELS:
      return MODELS[extension](model_proto, is_training)

  raise ValueError('Invalid model config!')
