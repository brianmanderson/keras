import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import testing


def _gradient_image(H, W, C=1, batch=1):
    h = np.linspace(0, 1, H)
    w = np.linspace(0, 1, W)
    c = np.linspace(0, 1, C) if C > 1 else np.array([0.5])
    Hg, Wg, Cg = np.meshgrid(h, w, c, indexing="ij")
    img = ((2 * Hg + 3 * Wg + 4 * Cg) / 9.0).astype("float32")
    return np.broadcast_to(img[None, ...], (batch, H, W, C)).copy()


class ReconstructPatches2DTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        # Pin channels_last so tests that don't pass `data_format` explicitly
        # are independent of the backend's default image_data_format (the
        # torch CI config defaults to channels_first).
        self._original_data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")

    def tearDown(self):
        super().tearDown()
        backend.set_image_data_format(self._original_data_format)

    @parameterized.parameters(
        # (H, W, C, size, padding, data_format)
        (64, 64, 1, (8, 8), "valid", "channels_last"),
        (64, 64, 3, (8, 8), "valid", "channels_last"),
        (32, 48, 1, (4, 8), "valid", "channels_last"),
        (59, 55, 3, (8, 8), "same", "channels_last"),
        (33, 41, 2, (5, 7), "same", "channels_last"),
        (7, 11, 1, (3, 5), "same", "channels_last"),
        (64, 64, 3, (8, 8), "valid", "channels_first"),
        (32, 48, 1, (4, 8), "valid", "channels_first"),
        (59, 55, 3, (8, 8), "same", "channels_first"),
        (33, 41, 2, (5, 7), "same", "channels_first"),
        (7, 11, 1, (3, 5), "same", "channels_first"),
    )
    def test_extract_then_reconstruct_roundtrip(
        self, H, W, C, size, padding, data_format
    ):
        x = _gradient_image(H, W, C, batch=2)
        x_t = ops.convert_to_tensor(x)
        if data_format == "channels_first":
            x_t = ops.transpose(x_t, (0, 3, 1, 2))
        patches = ops.image.extract_patches(
            x_t, size=size, padding=padding, data_format=data_format
        )
        layer = layers.ReconstructPatches2D(
            size=size,
            output_size=(H, W),
            padding=padding,
            data_format=data_format,
        )
        recon = layer(patches)
        self.assertEqual(tuple(recon.shape), tuple(x_t.shape))
        self.assertAllClose(recon, x_t, atol=1e-6)

    @parameterized.parameters("channels_last", "channels_first")
    def test_dynamic_spatial_dim(self, data_format):
        size = (4, 4)
        flat = size[0] * size[1] * 3
        if data_format == "channels_last":
            input_layer = layers.Input(batch_shape=(1, None, None, flat))
            expected = (1, 16, 16, 3)
        else:
            input_layer = layers.Input(batch_shape=(1, flat, None, None))
            expected = (1, 3, 16, 16)
        recon = layers.ReconstructPatches2D(
            size=size,
            output_size=(16, 16),
            padding="valid",
            data_format=data_format,
        )(input_layer)
        self.assertEqual(recon.shape, expected)

    @parameterized.parameters("channels_last", "channels_first")
    def test_output_size_autoinfer_valid(self, data_format):
        # output_size omitted -> inferred from the patch grid for valid.
        x = _gradient_image(64, 64, 3, batch=2)
        x_t = ops.convert_to_tensor(x)
        if data_format == "channels_first":
            x_t = ops.transpose(x_t, (0, 3, 1, 2))
        patches = ops.image.extract_patches(
            x_t, size=(8, 8), padding="valid", data_format=data_format
        )
        layer = layers.ReconstructPatches2D(
            size=(8, 8), padding="valid", data_format=data_format
        )
        recon = layer(patches)
        self.assertEqual(tuple(recon.shape), tuple(x_t.shape))
        self.assertAllClose(recon, x_t, atol=1e-6)

    def test_autoinfer_compute_output_shape(self):
        size = (4, 4)
        flat = size[0] * size[1] * 3
        inp = layers.Input(batch_shape=(2, 5, 5, flat))
        out = layers.ReconstructPatches2D(size=size, padding="valid")(inp)
        self.assertEqual(out.shape, (2, 20, 20, 3))

    def test_autoinfer_compute_output_shape_cf_int_strides(self):
        # channels_first auto-infer + int `strides` normalization.
        size = (4, 4)
        flat = size[0] * size[1] * 3
        inp = layers.Input(batch_shape=(2, flat, 5, 5))
        out = layers.ReconstructPatches2D(
            size=size,
            strides=4,
            padding="valid",
            data_format="channels_first",
        )(inp)
        self.assertEqual(out.shape, (2, 3, 20, 20))

    def test_output_size_required_for_same(self):
        with self.assertRaisesRegex(ValueError, "required when"):
            layers.ReconstructPatches2D(size=(8, 8), padding="same")

    def test_get_config(self):
        layer = layers.ReconstructPatches2D(
            size=(3, 4),
            output_size=(12, 16),
            padding="valid",
            data_format="channels_first",
        )
        config = layer.get_config()
        restored = layers.ReconstructPatches2D.from_config(config)
        self.assertEqual(restored.size, (3, 4))
        self.assertEqual(restored.output_size, (12, 16))
        self.assertEqual(restored.padding, "valid")
        self.assertEqual(restored.data_format, "channels_first")

    def test_invalid_size(self):
        with self.assertRaisesRegex(ValueError, "length 2"):
            layers.ReconstructPatches2D(size=(2, 3, 4), output_size=(10, 15))

    def test_int_size(self):
        # int size is normalized to (size, size).
        layer = layers.ReconstructPatches2D(size=4, output_size=(16, 16))
        self.assertEqual(layer.size, (4, 4))

    def test_invalid_output_size(self):
        with self.assertRaisesRegex(ValueError, "length 2"):
            layers.ReconstructPatches2D(size=(2, 2), output_size=(8, 8, 8))

    def test_invalid_padding(self):
        with self.assertRaisesRegex(ValueError, "'same' or 'valid'"):
            layers.ReconstructPatches2D(
                size=(2, 2),
                output_size=(8, 8),
                padding="reflect",
            )
