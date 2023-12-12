# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/grappler/costs/op_performance_data.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tensorflow.core.framework import tensor_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__pb2
from tensorflow.core.framework import tensor_shape_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2
from tensorflow.core.framework import types_pb2 as tensorflow_dot_core_dot_framework_dot_types__pb2
from tensorflow.core.framework import attr_value_pb2 as tensorflow_dot_core_dot_framework_dot_attr__value__pb2
from tensorflow.core.protobuf import device_properties_pb2 as tensorflow_dot_core_dot_protobuf_dot_device__properties__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n8tensorflow/core/grappler/costs/op_performance_data.proto\x12\ntensorflow\x1a&tensorflow/core/framework/tensor.proto\x1a,tensorflow/core/framework/tensor_shape.proto\x1a%tensorflow/core/framework/types.proto\x1a*tensorflow/core/framework/attr_value.proto\x1a\x30tensorflow/core/protobuf/device_properties.proto\"+\n\x0bSessionInfo\x12\x1c\n\x14intra_op_parallelism\x18\x01 \x01(\x03\"\xdb\x03\n\x06OpInfo\x12\n\n\x02op\x18\x01 \x01(\t\x12*\n\x04\x61ttr\x18\x02 \x03(\x0b\x32\x1c.tensorflow.OpInfo.AttrEntry\x12\x33\n\x06inputs\x18\x03 \x03(\x0b\x32#.tensorflow.OpInfo.TensorProperties\x12\x34\n\x07outputs\x18\x05 \x03(\x0b\x32#.tensorflow.OpInfo.TensorProperties\x12,\n\x06\x64\x65vice\x18\x04 \x01(\x0b\x32\x1c.tensorflow.DeviceProperties\x12-\n\x0csession_info\x18\x06 \x01(\x0b\x32\x17.tensorflow.SessionInfo\x1a\x42\n\tAttrEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.tensorflow.AttrValue:\x02\x38\x01\x1a\x8c\x01\n\x10TensorProperties\x12#\n\x05\x64type\x18\x01 \x01(\x0e\x32\x14.tensorflow.DataType\x12+\n\x05shape\x18\x02 \x01(\x0b\x32\x1c.tensorflow.TensorShapeProto\x12&\n\x05value\x18\x03 \x01(\x0b\x32\x17.tensorflow.TensorProto\"/\n\x12NormalDistribution\x12\n\n\x02mu\x18\x01 \x01(\x01\x12\r\n\x05sigma\x18\x02 \x01(\x01\"2\n\x15LogNormalDistribution\x12\n\n\x02mu\x18\x01 \x01(\x01\x12\r\n\x05sigma\x18\x02 \x01(\x01\"\xf3\x04\n\rOpPerformance\x12\x1e\n\x02op\x18\x01 \x01(\x0b\x32\x12.tensorflow.OpInfo\x12\x31\n\x0csession_info\x18\x0c \x01(\x0b\x32\x17.tensorflow.SessionInfoB\x02\x18\x01\x12\x0c\n\x04node\x18\x05 \x01(\t\x12\x1d\n\x15temporary_memory_size\x18\x02 \x01(\x03\x12\x14\n\x0c\x63ompute_cost\x18\x03 \x01(\x03\x12\x14\n\x0c\x63ompute_time\x18\x06 \x01(\x03\x12\x13\n\x0bmemory_time\x18\x07 \x01(\x03\x12\x1a\n\x12\x63ompute_efficiency\x18\x04 \x01(\x01\x12\x19\n\x11memory_efficiency\x18\x08 \x01(\x01\x12?\n\x15\x65xecution_time_normal\x18\n \x01(\x0b\x32\x1e.tensorflow.NormalDistributionH\x00\x12\x46\n\x19\x65xecution_time_log_normal\x18\x0b \x01(\x0b\x32!.tensorflow.LogNormalDistributionH\x00\x12\x35\n\top_memory\x18\t \x01(\x0b\x32\".tensorflow.OpPerformance.OpMemory\x1a\x97\x01\n\x08OpMemory\x12\x15\n\routput_memory\x18\x01 \x03(\x03\x12\x13\n\x0btemp_memory\x18\x02 \x01(\x03\x12\x19\n\x11persistent_memory\x18\x04 \x01(\x03\x12\x1e\n\x12\x64\x65vice_temp_memory\x18\x03 \x01(\x03\x42\x02\x18\x01\x12$\n\x18\x64\x65vice_persistent_memory\x18\x05 \x01(\x03\x42\x02\x18\x01\x42\x10\n\x0e\x65xecution_time\"F\n\x11OpPerformanceList\x12\x31\n\x0eop_performance\x18\x01 \x03(\x0b\x32\x19.tensorflow.OpPerformanceB\x03\xf8\x01\x01\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorflow.core.grappler.costs.op_performance_data_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\370\001\001'
  _OPINFO_ATTRENTRY._options = None
  _OPINFO_ATTRENTRY._serialized_options = b'8\001'
  _OPPERFORMANCE_OPMEMORY.fields_by_name['device_temp_memory']._options = None
  _OPPERFORMANCE_OPMEMORY.fields_by_name['device_temp_memory']._serialized_options = b'\030\001'
  _OPPERFORMANCE_OPMEMORY.fields_by_name['device_persistent_memory']._options = None
  _OPPERFORMANCE_OPMEMORY.fields_by_name['device_persistent_memory']._serialized_options = b'\030\001'
  _OPPERFORMANCE.fields_by_name['session_info']._options = None
  _OPPERFORMANCE.fields_by_name['session_info']._serialized_options = b'\030\001'
  _SESSIONINFO._serialized_start=291
  _SESSIONINFO._serialized_end=334
  _OPINFO._serialized_start=337
  _OPINFO._serialized_end=812
  _OPINFO_ATTRENTRY._serialized_start=603
  _OPINFO_ATTRENTRY._serialized_end=669
  _OPINFO_TENSORPROPERTIES._serialized_start=672
  _OPINFO_TENSORPROPERTIES._serialized_end=812
  _NORMALDISTRIBUTION._serialized_start=814
  _NORMALDISTRIBUTION._serialized_end=861
  _LOGNORMALDISTRIBUTION._serialized_start=863
  _LOGNORMALDISTRIBUTION._serialized_end=913
  _OPPERFORMANCE._serialized_start=916
  _OPPERFORMANCE._serialized_end=1543
  _OPPERFORMANCE_OPMEMORY._serialized_start=1374
  _OPPERFORMANCE_OPMEMORY._serialized_end=1525
  _OPPERFORMANCELIST._serialized_start=1545
  _OPPERFORMANCELIST._serialized_end=1615
# @@protoc_insertion_point(module_scope)
