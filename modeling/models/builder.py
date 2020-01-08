from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protos import model_pb2
from modeling.models.vcr_bert import VCRBert
from modeling.models.vcr_vbert import VCRVBert
from modeling.models.vcr_bilstm_fuse import VCRBiLSTMFuse
from modeling.models.vcr_bilstm_concat import VCRBiLSTMConcat
from modeling.models.vcr_bilstm_glove import VCRBiLSTMGloVe
from modeling.models.vcr_bilstm_bert import VCRBiLSTMBert

MODELS = {
    model_pb2.VCRBiLSTMFuse.ext: VCRBiLSTMFuse,
    model_pb2.VCRBiLSTMConcat.ext: VCRBiLSTMConcat,
    model_pb2.VCRBert.ext: VCRBert,
    model_pb2.VCRVBert.ext: VCRVBert,
    model_pb2.VCRBiLSTMGloVe.ext: VCRBiLSTMGloVe,
    model_pb2.VCRBiLSTMBert.ext: VCRBiLSTMBert,
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
