from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving.object_registration import register_keras_serializable


@keras_export("keras.layers.ReconstructPatches2D")
@register_keras_serializable(package="keras")
class ReconstructPatches2D(Layer):
    """Reconstructs 4D image(s) from 2D patches.

    Inverse of `keras.ops.image.extract_patches` (with length-2 `size`).
    See `ReconstructPatches3D` for the full feature set — this layer is
    the 2D analogue with identical semantics and arguments adapted to 4D
    image inputs and length-2 `size`/`output_size`.

    Example:

    >>> import numpy as np
    >>> import keras
    >>> image = np.random.random((1, 20, 20, 3)).astype("float32")
    >>> patches = keras.ops.image.extract_patches(image, (5, 5))
    >>> recon = keras.layers.ReconstructPatches2D(
    ...     size=(5, 5), output_size=(20, 20)
    ... )(patches)
    >>> recon.shape
    (1, 20, 20, 3)

    Args:
        size: int or tuple `(pH, pW)`.
        output_size: Optional tuple `(H, W)`. Required for `padding="same"`.
        strides: int or tuple. Defaults to `size`.
        padding: `"valid"` or `"same"`.
        data_format: `"channels_last"` or `"channels_first"`.
        reduction: `"mean"` (default) or `"sum"`.
        dilation_rate: int or tuple.

    Input shape:
        3D `(gH, gW, pH*pW*C)` or 4D `(batch, gH, gW, pH*pW*C)`
        (channels_last); `(batch, pH*pW*C, gH, gW)` (channels_first batched).
        Or a list `[patches, reference]`.

    Output shape:
        3D `(H, W, C)` or 4D `(batch, H, W, C)`; channels_first puts
        `C` first.
    """

    def __init__(
        self,
        size,
        output_size=None,
        strides=None,
        padding="valid",
        data_format=None,
        reduction="mean",
        dilation_rate=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(size, int):
            size = (size, size)
        if len(size) != 2:
            raise ValueError(
                f"`size` must be an int or a tuple of length 2. "
                f"Received: size={size}"
            )
        if output_size is None:
            pass
        elif len(output_size) != 2:
            raise ValueError(
                f"`output_size` must be a tuple of length 2 (H, W). "
                f"Received: output_size={output_size}"
            )
        if padding not in ("same", "valid"):
            raise ValueError(
                f"`padding` must be 'same' or 'valid'. "
                f"Received: padding={padding}"
            )
        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"`reduction` must be 'mean' or 'sum'. "
                f"Received: reduction={reduction}"
            )
        from keras.src.ops.image import _normalize_strides
        _normalize_strides(strides, size, "ReconstructPatches2D")
        self.size = tuple(size)
        self.output_size = tuple(output_size) if output_size is not None else None
        self.strides = strides
        self.padding = padding
        self.data_format = backend.standardize_data_format(data_format)
        self.reduction = reduction
        self.dilation_rate = dilation_rate

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 2:
                raise ValueError(
                    "ReconstructPatches2D called with a list expects exactly "
                    f"[patches, reference], got list of length {len(inputs)}."
                )
            patches, reference = inputs
            ref_shape = ops.shape(reference)
            if self.data_format == "channels_last":
                output_size = (ref_shape[1], ref_shape[2])
            else:
                output_size = (ref_shape[2], ref_shape[3])
        else:
            patches = inputs
            output_size = self.output_size
        from keras.src.ops.image import reconstruct_patches
        return reconstruct_patches(
            patches,
            size=self.size,
            output_size=output_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            reduction=self.reduction,
            dilation_rate=self.dilation_rate,
        )

    def compute_output_shape(self, input_shape):
        patch_volume = self.size[0] * self.size[1]
        if isinstance(input_shape, list) and len(input_shape) == 2:
            patches_shape, ref_shape = input_shape
            if self.data_format == "channels_last":
                output_size = tuple(ref_shape[1:3])
            else:
                output_size = tuple(ref_shape[2:4])
            input_shape = patches_shape
        elif self.output_size is not None:
            output_size = self.output_size
        else:
            output_size = self._infer_output_size_from_shape(input_shape)
        if self.data_format == "channels_last":
            flat = input_shape[-1]
            channels = None if flat is None else flat // patch_volume
            if len(input_shape) == 4:
                return (input_shape[0],) + tuple(output_size) + (channels,)
            elif len(input_shape) == 3:
                return tuple(output_size) + (channels,)
        else:
            if len(input_shape) == 4:
                flat = input_shape[1]
                channels = None if flat is None else flat // patch_volume
                return (input_shape[0], channels) + tuple(output_size)
            elif len(input_shape) == 3:
                flat = input_shape[0]
                channels = None if flat is None else flat // patch_volume
                return (channels,) + tuple(output_size)
        raise ValueError(
            f"Unexpected patches rank for ReconstructPatches2D: "
            f"{len(input_shape)}"
        )

    def _infer_output_size_from_shape(self, input_shape):
        if self.data_format == "channels_last":
            grid = input_shape[-3:-1] if len(input_shape) == 4 else input_shape[:-1]
        else:
            grid = input_shape[-2:] if len(input_shape) >= 3 else input_shape[1:]
        if any(g is None for g in grid):
            return (None, None)
        strides = self.strides if self.strides is not None else self.size
        if isinstance(strides, int):
            strides = (strides, strides)
        return tuple(
            (g - 1) * s + k for g, s, k in zip(grid, strides, self.size)
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "size": self.size,
            "output_size": self.output_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "reduction": self.reduction,
            "dilation_rate": self.dilation_rate,
        }
        return {**base_config, **config}
