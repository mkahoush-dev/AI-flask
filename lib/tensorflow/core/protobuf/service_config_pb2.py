# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/protobuf/service_config.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tensorflow.core.protobuf import data_service_pb2 as tensorflow_dot_core_dot_protobuf_dot_data__service__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n-tensorflow/core/protobuf/service_config.proto\x12\x1ctensorflow.data.experimental\x1a+tensorflow/core/protobuf/data_service.proto\"\xf3\x02\n\x10\x44ispatcherConfig\x12\x0c\n\x04port\x18\x01 \x01(\x03\x12\x10\n\x08protocol\x18\x02 \x01(\t\x12\x10\n\x08work_dir\x18\x03 \x01(\t\x12\x1b\n\x13\x66\x61ult_tolerant_mode\x18\x04 \x01(\x08\x12\x18\n\x10worker_addresses\x18\x07 \x03(\t\x12\x38\n\x0f\x64\x65ployment_mode\x18\t \x01(\x0e\x32\x1f.tensorflow.data.DeploymentMode\x12 \n\x18job_gc_check_interval_ms\x18\x05 \x01(\x03\x12\x19\n\x11job_gc_timeout_ms\x18\x06 \x01(\x03\x12 \n\x18gc_dynamic_sharding_jobs\x18\x0b \x01(\x08\x12\x19\n\x11\x63lient_timeout_ms\x18\x08 \x01(\x03\x12\x19\n\x11worker_timeout_ms\x18\n \x01(\x03\x12\'\n\x1fworker_max_concurrent_snapshots\x18\x0c \x01(\x03\"\xe5\x02\n\x0cWorkerConfig\x12\x0c\n\x04port\x18\x01 \x01(\x03\x12\x10\n\x08protocol\x18\x02 \x01(\t\x12\x1a\n\x12\x64ispatcher_address\x18\x03 \x01(\t\x12\x16\n\x0eworker_address\x18\x04 \x01(\t\x12\x13\n\x0bworker_tags\x18\n \x03(\t\x12\x1d\n\x15heartbeat_interval_ms\x18\x05 \x01(\x03\x12\x1d\n\x15\x64ispatcher_timeout_ms\x18\x06 \x01(\x03\x12\x1e\n\x16\x64\x61ta_transfer_protocol\x18\x07 \x01(\t\x12\x1d\n\x15\x64\x61ta_transfer_address\x18\x08 \x01(\t\x12&\n\x1e\x63ross_trainer_cache_size_bytes\x18\x0b \x01(\x03\x12%\n\x1dsnapshot_max_chunk_size_bytes\x18\x0c \x01(\x03\x12 \n\x18shutdown_quiet_period_ms\x18\t \x01(\x03\x42WZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_protob\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorflow.core.protobuf.service_config_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'ZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto'
  _DISPATCHERCONFIG._serialized_start=125
  _DISPATCHERCONFIG._serialized_end=496
  _WORKERCONFIG._serialized_start=499
  _WORKERCONFIG._serialized_end=856
# @@protoc_insertion_point(module_scope)
