from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer


@keras_export("keras.layers.ReconstructPatches3D")
class ReconstructPatches3D(Layer):
    """Reconstructs 5D volume(s) from non-overlapping 3D patches.

    Inverse of `keras.ops.image.extract_patches_3d` for the non-overlapping
    case (`strides == size`).

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
        size: Patch size as int or tuple `(patch_depth, patch_height,
            patch_width)`, matching the `size` used for extraction.
        output_size: Tuple `(D, H, W)` — the original spatial shape before
            extraction. May be omitted (`None`) for `padding="valid"`, in
            which case it is inferred from the patch grid; required for
            `padding="same"`, where the original size cannot be recovered
            from the patches alone.
        strides: Currently must equal `size` (non-overlapping). Defaults to
            `size`.
        padding: One of `"valid"` or `"same"`, matching the extraction.
        data_format: A string, one of `"channels_last"` (default) or
            `"channels_first"`. The ordering of the dimensions in the inputs.

    Input shape:
        5D tensor. With `data_format="channels_last"`,
        `(batch_size, gD, gH, gW, pD*pH*pW*C)`; with `"channels_first"`,
        `(batch_size, pD*pH*pW*C, gD, gH, gW)`.

    Output shape:
        5D tensor. With `data_format="channels_last"`,
        `(batch_size, D, H, W, C)`; with `"channels_first"`,
        `(batch_size, C, D, H, W)`.
    """

    def __init__(
        self,
        size,
        output_size=None,
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
                f"`size` must be an int or a tuple of length 3. "
                f"Received: size={size}"
            )
        if output_size is not None and len(output_size) != 3:
            raise ValueError(
                f"`output_size` must be a tuple of length 3 (D, H, W). "
                f"Received: output_size={output_size}"
            )
        if padding not in ("same", "valid"):
            raise ValueError(
                f"`padding` must be 'same' or 'valid'. "
                f"Received: padding={padding}"
            )
        if output_size is None and padding == "same":
            raise ValueError(
                "`output_size` is required when `padding='same'`: the "
                "original size cannot be inferred from patches alone."
            )
        self.size = tuple(size)
        self.output_size = (
            tuple(output_size) if output_size is not None else None
        )
        self.strides = strides
        self.padding = padding
        self.data_format = backend.standardize_data_format(data_format)
        self.input_spec = InputSpec(ndim=5)

    def call(self, patches):
        return ops.image.reconstruct_patches_3d(
            patches,
            size=self.size,
            output_size=self.output_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )

    def compute_output_shape(self, input_shape):
        # `InputSpec(ndim=5)` means we always see a 5D (batched) input.
        patch_volume = self.size[0] * self.size[1] * self.size[2]
        output_size = self.output_size
        if output_size is None:
            # Auto-infer (padding="valid"): grid spacing is the stride.
            grid = (
                input_shape[1:4]
                if self.data_format == "channels_last"
                else input_shape[2:5]
            )
            strides = self.strides if self.strides is not None else self.size
            if isinstance(strides, int):
                strides = (strides, strides, strides)
            output_size = tuple(
                None if g is None else (g - 1) * s + k
                for g, s, k in zip(grid, strides, self.size)
            )
        if self.data_format == "channels_last":
            flat = input_shape[-1]
            channels = None if flat is None else flat // patch_volume
            return (input_shape[0],) + output_size + (channels,)
        flat = input_shape[1]
        channels = None if flat is None else flat // patch_volume
        return (input_shape[0], channels) + output_size

    def get_config(self):
        config = {
            "size": self.size,
            "output_size": self.output_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return {**base_config, **config}
