import pytest
import tensorflow as tf
import sys
sys.path.insert(0, '')
from app.codingAssistant import CodingAssistant

from tensorflow.keras import layers, models


class Testvgg16Model:
    @pytest.fixture(scope="class")
    def setup(model_config):
        # This setup method will be run once for all tests in this class
        assistant = CodingAssistant()
        responses = [32, 10, None, 3]  # Example: image size, num_classes, dataset_path, channels
        filters = 64
        dropout_rate = 0.5
        learning_rate = 1e-3
        model = assistant.build_vgg(responses, filters=filters, dropout_rate=dropout_rate, learning_rate=learning_rate)
        return model

    def test_instance(self, setup):
        model = setup
        assert isinstance(model, tf.keras.models.Model), "The method did not return a Keras model instance."

    def test_output_shape(self, setup):
        model = setup
        assert model.output_shape == (None, 10), "The output shape does not match the expected number of classes."

    def test_dropout_layers(self, setup, dropout_rate=0.5):
        model = setup
        dropout_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)]
        assert len(dropout_layers) > 0, "No dropout layers found."
        assert all(layer.rate == dropout_rate for layer in dropout_layers), "Dropout rates do not match the expected value."
