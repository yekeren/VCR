# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/learning_rate_schedule.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='protos/learning_rate_schedule.proto',
  package='',
  syntax='proto2',
  serialized_pb=b'\n#protos/learning_rate_schedule.proto\"\x9d\x01\n\x14LearningRateSchedule\x12;\n\x18piecewise_constant_decay\x18\x01 \x01(\x0b\x32\x17.PiecewiseConstantDecayH\x00\x12.\n\x11\x65xponential_decay\x18\x02 \x01(\x0b\x32\x11.ExponentialDecayH\x00\x42\x18\n\x16learning_rate_schedule\"<\n\x16PiecewiseConstantDecay\x12\x12\n\nboundaries\x18\x01 \x03(\x05\x12\x0e\n\x06values\x18\x02 \x03(\x02\"t\n\x10\x45xponentialDecay\x12\x1d\n\x15initial_learning_rate\x18\x01 \x01(\x02\x12\x13\n\x0b\x64\x65\x63\x61y_steps\x18\x02 \x01(\x05\x12\x12\n\ndecay_rate\x18\x03 \x01(\x02\x12\x18\n\tstaircase\x18\x04 \x01(\x08:\x05\x66\x61lse'
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_LEARNINGRATESCHEDULE = _descriptor.Descriptor(
  name='LearningRateSchedule',
  full_name='LearningRateSchedule',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='piecewise_constant_decay', full_name='LearningRateSchedule.piecewise_constant_decay', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='exponential_decay', full_name='LearningRateSchedule.exponential_decay', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='learning_rate_schedule', full_name='LearningRateSchedule.learning_rate_schedule',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=40,
  serialized_end=197,
)


_PIECEWISECONSTANTDECAY = _descriptor.Descriptor(
  name='PiecewiseConstantDecay',
  full_name='PiecewiseConstantDecay',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='boundaries', full_name='PiecewiseConstantDecay.boundaries', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='values', full_name='PiecewiseConstantDecay.values', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=199,
  serialized_end=259,
)


_EXPONENTIALDECAY = _descriptor.Descriptor(
  name='ExponentialDecay',
  full_name='ExponentialDecay',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='initial_learning_rate', full_name='ExponentialDecay.initial_learning_rate', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='decay_steps', full_name='ExponentialDecay.decay_steps', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='decay_rate', full_name='ExponentialDecay.decay_rate', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='staircase', full_name='ExponentialDecay.staircase', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=261,
  serialized_end=377,
)

_LEARNINGRATESCHEDULE.fields_by_name['piecewise_constant_decay'].message_type = _PIECEWISECONSTANTDECAY
_LEARNINGRATESCHEDULE.fields_by_name['exponential_decay'].message_type = _EXPONENTIALDECAY
_LEARNINGRATESCHEDULE.oneofs_by_name['learning_rate_schedule'].fields.append(
  _LEARNINGRATESCHEDULE.fields_by_name['piecewise_constant_decay'])
_LEARNINGRATESCHEDULE.fields_by_name['piecewise_constant_decay'].containing_oneof = _LEARNINGRATESCHEDULE.oneofs_by_name['learning_rate_schedule']
_LEARNINGRATESCHEDULE.oneofs_by_name['learning_rate_schedule'].fields.append(
  _LEARNINGRATESCHEDULE.fields_by_name['exponential_decay'])
_LEARNINGRATESCHEDULE.fields_by_name['exponential_decay'].containing_oneof = _LEARNINGRATESCHEDULE.oneofs_by_name['learning_rate_schedule']
DESCRIPTOR.message_types_by_name['LearningRateSchedule'] = _LEARNINGRATESCHEDULE
DESCRIPTOR.message_types_by_name['PiecewiseConstantDecay'] = _PIECEWISECONSTANTDECAY
DESCRIPTOR.message_types_by_name['ExponentialDecay'] = _EXPONENTIALDECAY

LearningRateSchedule = _reflection.GeneratedProtocolMessageType('LearningRateSchedule', (_message.Message,), dict(
  DESCRIPTOR = _LEARNINGRATESCHEDULE,
  __module__ = 'protos.learning_rate_schedule_pb2'
  # @@protoc_insertion_point(class_scope:LearningRateSchedule)
  ))
_sym_db.RegisterMessage(LearningRateSchedule)

PiecewiseConstantDecay = _reflection.GeneratedProtocolMessageType('PiecewiseConstantDecay', (_message.Message,), dict(
  DESCRIPTOR = _PIECEWISECONSTANTDECAY,
  __module__ = 'protos.learning_rate_schedule_pb2'
  # @@protoc_insertion_point(class_scope:PiecewiseConstantDecay)
  ))
_sym_db.RegisterMessage(PiecewiseConstantDecay)

ExponentialDecay = _reflection.GeneratedProtocolMessageType('ExponentialDecay', (_message.Message,), dict(
  DESCRIPTOR = _EXPONENTIALDECAY,
  __module__ = 'protos.learning_rate_schedule_pb2'
  # @@protoc_insertion_point(class_scope:ExponentialDecay)
  ))
_sym_db.RegisterMessage(ExponentialDecay)


# @@protoc_insertion_point(module_scope)