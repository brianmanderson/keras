from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.layers.reshaping.extract_patches2d import (
    _extract_compute_output_shape,
)
from keras.src.saving.object_registration import register_keras_serializable


@keras_export("keras.layers.ExtractPatches3D")
@register_keras_serializable(package="keras")
class ExtractPatches3D(Layer):
    """Layer wrapper for `keras.ops.image.extract_patches_3d`.

    Identical semantics to the op with a length-3 `size`. See
    `ExtractPatches2D` for the full description; this variant handles
    5D volume inputs and 3D patch sizes. Uses the explicit
    `extract_patches_3d` entry point so that `int`-typed `size` is
    correctly interpreted as 3D rather than dispatched to the 2D path.

    Args:
        size: int or tuple `(pD, pH, pW)`.
        strides: int or tuple. Defaults to `size`.
        padding: `"valid"` or `"same"`.
        data_format: `"channels_last"` or `"channels_first"`.
    """

    def __init__(
        self,
        size,
        strides=None,
        padding="valid",
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(size, int):
            size = (size, size, size)
        if len(size) != 3:
            raise ValueError(
                f"`size` must be int or tuple of length 3; got {size}"
            )
        if padding not in ("same", "valid"):
            raise ValueError(
                f"`padding` must be 'same' or 'valid'; got {padding}"
            )
        self.size = tuple(size)
        self.strides = strides
        self.padding = padding
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, volumes):
        return ops.image.extract_patches_3d(
            volumes,
            size=self.size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )

    def compute_output_shape(self, input_shape):
        strides = self.strides if self.strides is not None else self.size
        if isinstance(strides, int):
            strides = (strides, strides, strides)
        return _extract_compute_output_shape(
            input_shape, self.size, strides, self.padding, self.data_format,
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "size": self.size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
        }
        return {**base_config, **config}
