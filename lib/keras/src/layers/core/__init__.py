# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Core Keras layers."""

from keras.src.layers.core.activation import Activation
from keras.src.layers.core.dense import Dense
from keras.src.layers.core.einsum_dense import EinsumDense
from keras.src.layers.core.embedding import Embedding
from keras.src.layers.core.identity import Identity
from keras.src.layers.core.lambda_layer import Lambda
from keras.src.layers.core.masking import Masking

# Required by third_party/py/tensorflow_gnn/keras/keras_tensors.py
from keras.src.layers.core.tf_op_layer import ClassMethod
from keras.src.layers.core.tf_op_layer import InstanceMethod
from keras.src.layers.core.tf_op_layer import InstanceProperty
from keras.src.layers.core.tf_op_layer import SlicingOpLambda
from keras.src.layers.core.tf_op_layer import TFOpLambda
from keras.src.layers.core.tf_op_layer import _delegate_method
from keras.src.layers.core.tf_op_layer import _delegate_property

# Regularization layers imported for backwards namespace compatibility
from keras.src.layers.regularization.activity_regularization import (
    ActivityRegularization,
)
from keras.src.layers.regularization.dropout import Dropout
from keras.src.layers.regularization.spatial_dropout1d import SpatialDropout1D
from keras.src.layers.regularization.spatial_dropout2d import SpatialDropout2D
from keras.src.layers.regularization.spatial_dropout3d import SpatialDropout3D

# Reshaping layers imported for backwards namespace compatibility
from keras.src.layers.reshaping.flatten import Flatten
from keras.src.layers.reshaping.permute import Permute
from keras.src.layers.reshaping.repeat_vector import RepeatVector
from keras.src.layers.reshaping.reshape import Reshape

