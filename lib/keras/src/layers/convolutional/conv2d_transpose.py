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
"""Keras 2D transposed convolution layer (sometimes called deconvolution)."""


import tensorflow.compat.v2 as tf

from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.dtensor import utils
from keras.src.engine.input_spec import InputSpec
from keras.src.layers.convolutional.conv2d import Conv2D
from keras.src.utils import conv_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export(
    "keras.layers.Conv2DTranspose", "keras.layers.Convolution2DTranspose"
)
class Conv2DTranspose(Conv2D):
    """Transposed convolution layer (sometimes called Deconvolution).

    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers or `None`, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.

    Args:
      filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding with zeros
        evenly to the left/right or up/down of the input such that output has
        the same height/width dimension as the input.
      output_padding: An integer or tuple/list of 2 integers,
        specifying the amount of padding along the height and width
        of the output tensor.
        Can be a single integer to specify the same value for all
        spatial dimensions.
        The amount of output padding along a given dimension must be
        lower than the stride along that same dimension.
        If set to `None` (default), the output shape is inferred.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch_size, channels, height, width)`.
        When unspecified, uses `image_data_format` value found in your Keras
        config file at `~/.keras/keras.json` (if exists) else 'channels_last'.
        Defaults to "channels_last".
      dilation_rate: an integer, specifying the dilation rate for all spatial
        dimensions for dilated convolution. Specifying different dilation rates
        for different dimensions is not supported.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any stride value != 1.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (see `keras.activations`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix
        (see `keras.initializers`). Defaults to 'glorot_uniform'.
      bias_initializer: Initializer for the bias vector
        (see `keras.initializers`). Defaults to 'zeros'.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix (see `keras.regularizers`).
      bias_regularizer: Regularizer function applied to the bias vector
        (see `keras.regularizers`).
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation") (see `keras.regularizers`).
      kernel_constraint: Constraint function applied to the kernel matrix
        (see `keras.constraints`).
      bias_constraint: Constraint function applied to the bias vector
        (see `keras.constraints`).

    Input shape:
      4D tensor with shape:
      `(batch_size, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(batch_size, rows, cols, channels)` if data_format='channels_last'.

    Output shape:
      4D tensor with shape:
      `(batch_size, filters, new_rows, new_cols)` if
      data_format='channels_first'
      or 4D tensor with shape:
      `(batch_size, new_rows, new_cols, filters)` if
      data_format='channels_last'.  `rows` and `cols` values might have changed
      due to padding.
      If `output_padding` is specified:
      ```
      new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] +
      output_padding[0])
      new_cols = ((cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] +
      output_padding[1])
      ```

    Returns:
      A tensor of rank 4 representing
      `activation(conv2dtranspose(inputs, kernel) + bias)`.

    Raises:
      ValueError: if `padding` is "causal".
      ValueError: when both `strides` > 1 and `dilation_rate` > 1.

    References:
      - [A guide to convolution arithmetic for deep
        learning](https://arxiv.org/abs/1603.07285v1)
      - [Deconvolutional
        Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
    """

    @utils.allow_initializer_layout
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        output_padding=None,
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs,
        )

        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, 2, "output_padding", allow_zero=True
            )
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError(
                        "Strides must be greater than output padding. "
                        f"Received strides={self.strides}, "
                        f"output_padding={self.output_padding}."
                    )

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                "Inputs should have rank 4. "
                f"Received input_shape={input_shape}."
            )
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "to `Conv2DTranspose` should be defined. "
                f"The input_shape received is {input_shape}, "
                f"where axis {channel_axis} (0-based) "
                "is the channel dimension, which found to be `None`."
            )
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == "channels_first":
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        # Use the constant height and weight when possible.
        # TODO(scottzhu): Extract this into a utility function that can be
        # applied to all convolutional layers, which currently lost the static
        # shape information due to tf.shape().
        height, width = None, None
        if inputs.shape.rank is not None:
            dims = inputs.shape.as_list()
            height = dims[h_axis]
            width = dims[w_axis]
        height = height if height is not None else inputs_shape[h_axis]
        width = width if width is not None else inputs_shape[w_axis]

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_output_length(
            height,
            kernel_h,
            padding=self.padding,
            output_padding=out_pad_h,
            stride=stride_h,
            dilation=self.dilation_rate[0],
        )
        out_width = conv_utils.deconv_output_length(
            width,
            kernel_w,
            padding=self.padding,
            output_padding=out_pad_w,
            stride=stride_w,
            dilation=self.dilation_rate[1],
        )
        if self.data_format == "channels_first":
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        output_shape_tensor = tf.stack(output_shape)
        outputs = backend.conv2d_transpose(
            inputs,
            self.kernel,
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        if not tf.executing_eagerly() and inputs.shape.rank:
            # Infer the static output shape:
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs,
                self.bias,
                data_format=conv_utils.convert_data_format(
                    self.data_format, ndim=4
                ),
            )

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        output_shape = list(input_shape)
        if self.data_format == "channels_first":
            c_axis, h_axis, w_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 3, 1, 2

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        output_shape[c_axis] = self.filters
        output_shape[h_axis] = conv_utils.deconv_output_length(
            output_shape[h_axis],
            kernel_h,
            padding=self.padding,
            output_padding=out_pad_h,
            stride=stride_h,
            dilation=self.dilation_rate[0],
        )
        output_shape[w_axis] = conv_utils.deconv_output_length(
            output_shape[w_axis],
            kernel_w,
            padding=self.padding,
            output_padding=out_pad_w,
            stride=stride_w,
            dilation=self.dilation_rate[1],
        )
        return tf.TensorShape(output_shape)

    def get_config(self):
        config = super().get_config()
        config["output_padding"] = self.output_padding
        return config


# Alias

Convolution2DTranspose = Conv2DTranspose

