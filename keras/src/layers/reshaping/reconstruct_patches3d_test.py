import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import testing


def _gradient_volume(D, H, W, C=1, batch=1):
    d = np.linspace(0, 1, D)
    h = np.linspace(0, 1, H)
    w = np.linspace(0, 1, W)
    c = np.linspace(0, 1, C) if C > 1 else np.array([0.5])
    Dg, Hg, Wg, Cg = np.meshgrid(d, h, w, c, indexing="ij")
    vol = ((Dg + 2 * Hg + 3 * Wg + 4 * Cg) / 10.0).astype("float32")
    return np.broadcast_to(vol[None, ...], (batch, D, H, W, C)).copy()


class ReconstructPatches3DTest(testing.TestCase):
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
        # (D, H, W, C, size, padding, data_format)
        (32, 64, 64, 1, (16, 32, 32), "valid", "channels_last"),
        (24, 48, 48, 1, (8, 16, 16), "valid", "channels_last"),
        (16, 16, 16, 2, (2, 4, 8), "valid", "channels_last"),
        (25, 59, 55, 2, (16, 32, 32), "same", "channels_last"),
        (17, 33, 41, 3, (4, 8, 8), "same", "channels_last"),
        (5, 7, 11, 1, (3, 5, 7), "same", "channels_last"),
        # channels_first: a representative subset (transpose-sandwich around
        # the same channels_last core), keeping the heavy cases out.
        (16, 16, 16, 2, (2, 4, 8), "valid", "channels_first"),
        (17, 33, 41, 3, (4, 8, 8), "same", "channels_first"),
        (5, 7, 11, 1, (3, 5, 7), "same", "channels_first"),
    )
    def test_extract_then_reconstruct_roundtrip(
        self, D, H, W, C, size, padding, data_format
    ):
        x = _gradient_volume(D, H, W, C, batch=2)
        x_t = ops.convert_to_tensor(x)
        if data_format == "channels_first":
            x_t = ops.transpose(x_t, (0, 4, 1, 2, 3))
        patches = ops.image.extract_patches(
            x_t, size=size, padding=padding, data_format=data_format
        )
        layer = layers.ReconstructPatches3D(
            size=size,
            output_size=(D, H, W),
            padding=padding,
            data_format=data_format,
        )
        recon = layer(patches)
        self.assertEqual(tuple(recon.shape), tuple(x_t.shape))
        self.assertAllClose(recon, x_t, atol=1e-6)

    @parameterized.parameters("channels_last", "channels_first")
    def test_dynamic_spatial_dim(self, data_format):
        # patches: batch known, grid axes None, flat dim known.
        size = (4, 4, 4)
        flat = size[0] * size[1] * size[2] * 3  # C=3
        if data_format == "channels_last":
            input_layer = layers.Input(batch_shape=(1, None, None, None, flat))
            expected = (1, 16, 16, 16, 3)
        else:
            input_layer = layers.Input(batch_shape=(1, flat, None, None, None))
            expected = (1, 3, 16, 16, 16)
        recon = layers.ReconstructPatches3D(
            size=size,
            output_size=(16, 16, 16),
            padding="valid",
            data_format=data_format,
        )(input_layer)
        self.assertEqual(recon.shape, expected)

    def test_get_config(self):
        layer = layers.ReconstructPatches3D(
            size=(2, 3, 4),
            output_size=(10, 15, 20),
            padding="same",
            data_format="channels_first",
        )
        config = layer.get_config()
        restored = layers.ReconstructPatches3D.from_config(config)
        self.assertEqual(restored.size, (2, 3, 4))
        self.assertEqual(restored.output_size, (10, 15, 20))
        self.assertEqual(restored.padding, "same")
        self.assertEqual(restored.data_format, "channels_first")

    def test_invalid_size(self):
        with self.assertRaisesRegex(ValueError, "length 3"):
            layers.ReconstructPatches3D(size=(2, 3), output_size=(10, 15, 20))

    def test_int_size(self):
        # int size is normalized to (size, size, size).
        layer = layers.ReconstructPatches3D(size=4, output_size=(16, 16, 16))
        self.assertEqual(layer.size, (4, 4, 4))

    def test_invalid_output_size(self):
        with self.assertRaisesRegex(ValueError, "length 3"):
            layers.ReconstructPatches3D(size=(2, 2, 2), output_size=(8, 8))

    def test_invalid_padding(self):
        with self.assertRaisesRegex(ValueError, "'same' or 'valid'"):
            layers.ReconstructPatches3D(
                size=(2, 2, 2),
                output_size=(8, 8, 8),
                padding="reflect",
            )

    def test_strides_overlap_not_implemented(self):
        x = _gradient_volume(16, 16, 16, 1, batch=1)
        patches = ops.image.extract_patches(
            ops.convert_to_tensor(x),
            size=(4, 4, 4),
            padding="valid",
        )
        with self.assertRaisesRegex(NotImplementedError, "non-overlapping"):
            layers.ReconstructPatches3D(
                size=(4, 4, 4),
                output_size=(16, 16, 16),
                strides=(2, 2, 2),
                padding="valid",
            )(patches)
