from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protos import model_pb2
from vcr.bert import VCRBert
from vcr.bilstm_glove import VCRBiLSTMGloVe
from vcr.vibilstm_glove import VCRViBiLSTMGloVe
from vcr.r2c import VCRR2C
from vcr.r2c_grounding import VCRR2CGrounding
from vcr.r2c_glove import R2CGlove
from vcr.r2c_bert import R2CBert
from vcr.r2c_vil_bert import R2CVilBert

MODELS = {
    model_pb2.VCRBert.ext: VCRBert,
    model_pb2.VCRBiLSTMGloVe.ext: VCRBiLSTMGloVe,
    model_pb2.VCRViBiLSTMGloVe.ext: VCRViBiLSTMGloVe,
    model_pb2.VCRR2C.ext: VCRR2C,
    model_pb2.VCRR2CGrounding.ext: VCRR2CGrounding,
    model_pb2.R2CGlove.ext: R2CGlove,
    model_pb2.R2CBert.ext: R2CBert,
    model_pb2.R2CVilBert.ext: R2CVilBert,
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