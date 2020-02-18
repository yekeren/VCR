from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protos import model_pb2
from vcr.r2c_frozen import VCRR2CFrozen
from vcr.b2t2_frozen import VCRB2T2Frozen
from vcr.bert_next_sentence import VCRBertNextSentence
from vcr.bert_next_sentence_adversarial import VCRBertNextSentenceAdversarial

MODELS = {
    model_pb2.VCRR2CFrozen.ext: VCRR2CFrozen,
    model_pb2.VCRB2T2Frozen.ext: VCRB2T2Frozen,
    model_pb2.VCRBertNextSentence.ext: VCRBertNextSentence,
    model_pb2.VCRBertNextSentenceAdversarial.ext: VCRBertNextSentenceAdversarial,
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
