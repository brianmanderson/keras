from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving.object_registration import register_keras_serializable


@keras_export("keras.layers.ReconstructPatches3D")
@register_keras_serializable(package="keras")
class ReconstructPatches3D(Layer):
    """Reconstructs 5D volume(s) from 3D patches.

    Inverse of `keras.ops.image.extract_patches_3d`. Supports non-overlapping
    (`strides == size`, fast reshape path) and overlapping (`strides < size`,
    conv-transpose path) reconstruction; `channels_last` and `channels_first`
    data formats; `"valid"` and `"same"` padding; optional `dilation_rate`;
    and either of two `reduction` modes for overlapping patches (`"mean"`
    recovers the original input, `"sum"` matches `torch.nn.Fold` semantics).

    The layer also supports a dual-input call signature
    `layer([patches, reference])`, in which the reconstruction's output
    spatial shape is derived from `reference`'s shape at call time. This
    enables drop-in use in variable-input models declared with
    `Input(shape=[None, None, None, C])`.

    Example:

    >>> import numpy as np
    >>> import keras
    >>> volume = np.random.random((1, 10, 10, 10, 3)).astype("float32")
    >>> patches = keras.ops.image.extract_patches_3d(volume, (5, 5, 5))
    >>> recon = keras.layers.ReconstructPatches3D(
    ...     size=(5, 5, 5), output_size=(10, 10, 10)
    ... )(patches)
    >>> recon.shape
    (1, 10, 10, 10, 3)

    Args:
        size: int or tuple `(pD, pH, pW)`, matching the size used for
            extraction.
        output_size: Optional tuple `(D, H, W)` — the original spatial
            shape before extraction. Required for `padding="same"`. For
            `padding="valid"` it is inferred from `patches` shape if
            omitted. Can also be omitted when the layer is called with
            `[patches, reference]`, in which case it is derived from
            `reference` at call time.
        strides: int or tuple. Must satisfy `strides <= size` on every
            axis. Defaults to `size` (non-overlapping).
        padding: `"valid"` or `"same"`, matching the extraction.
        data_format: `"channels_last"` (default) or `"channels_first"`.
        reduction: `"mean"` (default) or `"sum"`. Combining strategy for
            overlapping patches; ignored when patches do not overlap.
        dilation_rate: int or tuple. Dilation of the patch kernel,
            matching the rate used at extraction. Meaningful only with
            `strides == 1` (overlap path).

    Input shape:
        4D tensor `(gD, gH, gW, pD*pH*pW*C)` or
        5D tensor `(batch, gD, gH, gW, pD*pH*pW*C)` (channels_last);
        `(batch, pD*pH*pW*C, gD, gH, gW)` (channels_first batched).
        Alternatively a list `[patches, reference]` of two tensors,
        where `reference`'s spatial dims are used as output_size.

    Output shape:
        4D tensor `(D, H, W, C)` or
        5D tensor `(batch, D, H, W, C)`; `(batch, C, D, H, W)` for
        channels_first.
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
            size = (size, size, size)
        if len(size) != 3:
            raise ValueError(
                f"`size` must be an int or a tuple of length 3. "
                f"Received: size={size}"
            )
        if output_size is None:
            # Legal in two modes:
            # 1. padding='valid' — inferred from patches.shape at call time
            # 2. dual-input call() — inferred from reference at call time
            pass
        elif len(output_size) != 3:
            raise ValueError(
                f"`output_size` must be a tuple of length 3 (D, H, W). "
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
        # Eagerly validate strides (rejects gapped strides at construct time).
        from keras.src.ops.image import _normalize_strides
        _normalize_strides(strides, size, "ReconstructPatches3D")
        self.size = tuple(size)
        self.output_size = tuple(output_size) if output_size is not None else None
        self.strides = strides
        self.padding = padding
        self.data_format = backend.standardize_data_format(data_format)
        self.reduction = reduction
        self.dilation_rate = dilation_rate

    def call(self, inputs):
        # Dual-input mode: [patches, reference]. Derive output_size dynamically.
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 2:
                raise ValueError(
                    "ReconstructPatches3D called with a list expects exactly "
                    f"[patches, reference], got list of length {len(inputs)}."
                )
            patches, reference = inputs
            ref_shape = ops.shape(reference)
            if self.data_format == "channels_last":
                output_size = (ref_shape[1], ref_shape[2], ref_shape[3])
            else:
                output_size = (ref_shape[2], ref_shape[3], ref_shape[4])
        else:
            patches = inputs
            output_size = self.output_size
        from keras.src.ops.image import reconstruct_patches_3d
        return reconstruct_patches_3d(
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
        patch_volume = self.size[0] * self.size[1] * self.size[2]
        # Dual-input: input_shape is [patches_shape, reference_shape].
        if isinstance(input_shape, list) and len(input_shape) == 2:
            patches_shape, ref_shape = input_shape
            if self.data_format == "channels_last":
                output_size = tuple(ref_shape[1:4])
            else:
                output_size = tuple(ref_shape[2:5])
            input_shape = patches_shape
        elif self.output_size is not None:
            output_size = self.output_size
        else:
            output_size = self._infer_output_size_from_shape(input_shape)
        if self.data_format == "channels_last":
            flat = input_shape[-1]
            channels = None if flat is None else flat // patch_volume
            if len(input_shape) == 5:
                return (input_shape[0],) + tuple(output_size) + (channels,)
            elif len(input_shape) == 4:
                return tuple(output_size) + (channels,)
        else:
            if len(input_shape) == 5:
                flat = input_shape[1]
                channels = None if flat is None else flat // patch_volume
                return (input_shape[0], channels) + tuple(output_size)
            elif len(input_shape) == 4:
                flat = input_shape[0]
                channels = None if flat is None else flat // patch_volume
                return (channels,) + tuple(output_size)
        raise ValueError(
            f"Unexpected patches rank for ReconstructPatches3D: "
            f"{len(input_shape)}"
        )

    def _infer_output_size_from_shape(self, input_shape):
        if self.data_format == "channels_last":
            grid = input_shape[-4:-1] if len(input_shape) == 5 else input_shape[:-1]
        else:
            grid = input_shape[-3:] if len(input_shape) >= 4 else input_shape[1:]
        if any(g is None for g in grid):
            return (None, None, None)
        strides = self.strides if self.strides is not None else self.size
        if isinstance(strides, int):
            strides = (strides, strides, strides)
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
