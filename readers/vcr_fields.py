from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

PAD = '[PAD]'
NUM_CHOICES = 4


class TFExampleFields(object):
  """Fields in the tf.train.Example."""
  img_id = 'img_id'
  annot_id = 'annot_id'
  answer_label = 'answer_label'
  rationale_label = 'rationale_label'

  img_bbox_label = "image/object/bbox/label"
  img_bbox_score = "image/object/bbox/score"
  img_bbox_feature = "image/object/bbox/feature"
  img_bbox_scope = "image/object/bbox/"
  img_bbox_field_keys = ['ymin', 'xmin', 'ymax', 'xmax']

  question = 'question'
  question_tag = 'question_tag'
  answer_choice = 'answer_choice'
  answer_choice_tag = 'answer_choice_tag'
  rationale_choice = 'rationale_choice'
  rationale_choice_tag = 'rationale_choice_tag'


class InputFields(object):
  """Names of the input tensors."""
  # Meta information.
  img_id = 'img_id'
  annot_id = 'annot_id'
  answer_label = 'answer_label'
  rationale_label = 'rationale_label'

  # Objects.
  num_objects = 'num_objects'
  object_bboxes = 'object_bboxes'
  object_labels = 'object_labels'
  object_scores = 'object_scores'
  object_features = 'object_features'

  # Question.
  question = 'question'
  question_tag = 'question_tag'
  question_len = 'question_len'

  # Answer choices.
  answer_choices = 'answer_choices'
  answer_choices_tag = 'answer_choices_tag'
  answer_choices_len = 'answer_choices_len'
  answer_choices_with_question = 'answer_choices_with_question'
  answer_choices_with_question_tag = 'answer_choices_with_question_tag'
  answer_choices_with_question_len = 'answer_choices_with_question_len'

  # answer_choices_with_question_wordnet = 'answer_choices_with_question_wordnet'
  # answer_choices_with_question_wordnet_len = 'answer_choices_with_question_wordnet_len'

  # Rationale choices.
  rationale_choices = 'rationale_choices'
  rationale_choices_tag = 'rationale_choices_tag'
  rationale_choices_len = 'rationale_choices_len'
  rationale_choices_with_question = 'rationale_choices_with_question'
  rationale_choices_with_question_tag = 'rationale_choices_with_question_tag'
  rationale_choices_with_question_len = 'rationale_choices_with_question_len'

  # rationale_choices_with_question_wordnet = 'rationale_choices_with_question_wordnet'
  # rationale_choices_with_question_wordnet_len = 'rationale_choices_with_question_wordnet_len'
