from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protos import model_pb2
# from vcr.r2c_frozen import VCRR2CFrozen
# from vcr.r2c_glove import VCRR2CGloVe
from vcr.r2c_glove_adv import VCRR2CGloVeAdv
from vcr.r2c_bert_adv import VCRR2CBertAdv
# from vcr.r2c_glove_adv2 import VCRR2CGloVeAdv2
from vcr.b2t2_frozen import VCRB2T2Frozen
from vcr.bilstm_next_sentence import VCRBiLSTMNextSentence
# from vcr.compatible_qa import VCRCompatibleQA

from vcr.bert_text_only import BertTextOnly
from vcr.bert_b2t2 import BertB2T2
from vcr.bert_b2t2_kb import BertB2T2Kb
from vcr.finetune_cc import FinetuneCC

MODELS = {
    #     model_pb2.VCRR2CFrozen.ext: VCRR2CFrozen,
    #     model_pb2.VCRR2CGloVe.ext: VCRR2CGloVe,
    model_pb2.VCRR2CGloVeAdv.ext:
        VCRR2CGloVeAdv,
    model_pb2.VCRR2CBertAdv.ext:
        VCRR2CBertAdv,
    #    model_pb2.VCRR2CGloVeAdv2.ext: VCRR2CGloVeAdv2,
    model_pb2.VCRB2T2Frozen.ext:
        VCRB2T2Frozen,
    model_pb2.VCRBiLSTMNextSentence.ext:
        VCRBiLSTMNextSentence,
    model_pb2.BertTextOnly.ext:
        BertTextOnly,
    model_pb2.BertB2T2.ext:
        BertB2T2,
    model_pb2.BertB2T2Kb.ext:
        BertB2T2Kb,
    #    model_pb2.VCRCompatibleQA.ext: VCRCompatibleQA,
    model_pb2.FinetuneCC.ext:
        FinetuneCC,
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
