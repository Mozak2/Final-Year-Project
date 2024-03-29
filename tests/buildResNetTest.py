import pytest
import tensorflow as tf
from app.codingAssistant import CodingAssistant
@pytest.fixture
def res_net_setup():
    assistant = CodingAssistant()
    responses = [64, 10, None, 3]  # Adjust based on your typical input
    model = assistant.build_resNet(responses, learning_rate=1e-3)
    return model, responses

def test_res_net_model_structure(res_net_setup):
    model, _ = res_net_setup
    assert isinstance(model, tf.keras.Model), "build_resNet should return a Keras Model instance."

def test_res_net_input_output_shape(res_net_setup):
    model, responses = res_net_setup
    input_shape = (None, responses[0], responses[0], responses[3])
    output_shape = (None, responses[1])
    assert model.input_shape == input_shape, f"Input shape should be {input_shape}."
    assert model.output_shape == output_shape, f"Output shape should be {output_shape}."

def test_res_net_residual_blocks(res_net_setup):
    model, _ = res_net_setup
    # Check for the presence of Conv2D and Add operations indicating residual connections
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    add_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Add)]
    assert len(conv_layers) > 0, "Should have Conv2D layers."
    assert len(add_layers) > 0, "Should have Add layers for residual connections."