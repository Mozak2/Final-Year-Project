import pytest
import tensorflow as tf
from app.codingAssistant import CodingAssistant  # Adjust the import based on your project structure

@pytest.fixture
def le_net_setup():
    # Example setup with arbitrary values
    responses = [32, 10, None, 3]  # [img_size, num_classes, dataset_path, channels]
    learning_rate = 1e-3
    assistant = CodingAssistant()
    model = assistant.build_leNet(responses, learning_rate)
    return model, responses

def test_le_net_instance(le_net_setup):
    model, _ = le_net_setup
    assert isinstance(model, tf.keras.Model), "The method did not return a Keras Model instance."

def test_le_net_input_shape(le_net_setup):
    model, responses = le_net_setup
    expected_shape = (None, responses[0], responses[0], responses[3])
    assert model.input_shape == expected_shape, f"Input shape {model.input_shape} does not match expected {expected_shape}."

def test_le_net_output_shape(le_net_setup):
    model, responses = le_net_setup
    assert model.output_shape == (None, responses[1]), f"Output shape {model.output_shape} does not match expected number of classes {responses[1]}."

def test_le_net_layers(le_net_setup):
    model, _ = le_net_setup
    # Check for specific layer types that should be present in the LeNet architecture
    layer_types = [type(layer) for layer in model.layers]
    expected_layers = [tf.keras.layers.Conv2D, tf.keras.layers.AveragePooling2D, tf.keras.layers.Flatten, tf.keras.layers.Dense]
    for expected_layer in expected_layers:
        assert expected_layer in layer_types, f"Expected layer {expected_layer} not found in model."