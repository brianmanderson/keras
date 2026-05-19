from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving.object_registration import register_keras_serializable


def _compute_grid(input_size, kernel, stride, padding):
    """Single-axis grid dim for a conv with given args; matches extract_patches."""
    if input_size is None:
        return None
    if padding == "valid":
        return max(0, (input_size - kernel) // stride + 1)
    return (input_size + stride - 1) // stride  # same


def _extract_compute_output_shape(
    input_shape, size, strides, padding, data_format,
):
    """Shared compute_output_shape for ExtractPatches{2,3}D."""
    n_spatial = len(size)
    rank = len(input_shape)
    if rank not in (n_spatial + 1, n_spatial + 2):
        raise ValueError(
            f"ExtractPatches expected rank {n_spatial + 1} or "
            f"{n_spatial + 2} (with batch); got input_shape={input_shape}"
        )
    batch = input_shape[0] if rank == n_spatial + 2 else None
    if data_format == "channels_last":
        channels = input_shape[-1]
        spatial = input_shape[-(n_spatial + 1):-1]
    else:
        if rank == n_spatial + 2:
            channels = input_shape[1]
            spatial = input_shape[2:]
        else:
            channels = input_shape[0]
            spatial = input_shape[1:]
    flat = None
    if channels is not None:
        flat = channels
        for s in size:
            flat *= s
    grid = tuple(_compute_grid(d, k, s, padding)
                 for d, k, s in zip(spatial, size, strides))
    if data_format == "channels_last":
        spatial_out = grid + (flat,)
    else:
        spatial_out = (flat,) + grid
    if rank == n_spatial + 2:
        return (batch,) + spatial_out
    return spatial_out


@keras_export("keras.layers.ExtractPatches2D")
@register_keras_serializable(package="keras")
class ExtractPatches2D(Layer):
    """Layer wrapper for `keras.ops.image.extract_patches` (2D).

    Identical semantics to the op; provided so users can compose
    extract -> reconstruct as symmetric Layer pairs in Functional or
    Sequential models without a `Lambda` layer (which avoids
    closure-deserialization gotchas at save/load time).

    Example:

    >>> import numpy as np
    >>> from keras.layers import ExtractPatches2D
    >>> image = np.random.random((2, 16, 16, 3)).astype("float32")
    >>> layer = ExtractPatches2D(size=(4, 4), padding="valid")
    >>> patches = layer(image)
    >>> tuple(patches.shape)
    (2, 4, 4, 48)

    Args:
        size: int or tuple `(pH, pW)`. Patch size.
        strides: int or tuple. Defaults to `size` (non-overlapping).
        padding: `"valid"` or `"same"`.
        data_format: `"channels_last"` or `"channels_first"`.

    Input shape:
        4D `(batch, H, W, C)` for channels_last,
        4D `(batch, C, H, W)` for channels_first.

    Output shape:
        4D `(batch, gH, gW, pH*pW*C)` for channels_last,
        4D `(batch, pH*pW*C, gH, gW)` for channels_first.
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
            size = (size, size)
        if len(size) != 2:
            raise ValueError(
                f"`size` must be int or tuple of length 2; got {size}"
            )
        if padding not in ("same", "valid"):
            raise ValueError(
                f"`padding` must be 'same' or 'valid'; got {padding}"
            )
        self.size = tuple(size)
        self.strides = strides
        self.padding = padding
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return ops.image.extract_patches(
            images,
            size=self.size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )

    def compute_output_shape(self, input_shape):
        strides = self.strides if self.strides is not None else self.size
        if isinstance(strides, int):
            strides = (strides, strides)
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
