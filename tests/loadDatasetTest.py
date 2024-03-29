import pytest
import tensorflow as tf
from app.codingAssistant import CodingAssistant
import os
import shutil
import pytest
from unittest.mock import patch, MagicMock
from tensorflow.keras.preprocessing import image
import numpy as np

# Fixture to setup a minimal dataset
@pytest.fixture(scope="module")
def minimal_dataset(tmpdir_factory):
    base_dir = tmpdir_factory.mktemp("data")
    class_dirs = ['class1', 'class2']  # Adjust based on your needs

    # Create directory structure and dummy images for each class in train/test/val
    for dtype in ['train', 'test', 'validation']:
        for class_dir in class_dirs:
            os.makedirs(os.path.join(base_dir, dtype, class_dir))
            # Create a dummy image file
            img_array = np.random.rand(32,32,3) * 255
            img = image.array_to_img(img_array)
            img.save(os.path.join(base_dir, dtype, class_dir, "dummy.jpg"))

    return str(base_dir)

# Actual test using the minimal dataset
def test_load_my_dataset(minimal_dataset):
    assistant = CodingAssistant()
    responses = [32, 2, minimal_dataset, 3]  # Example responses

    train_dataset, test_dataset, val_dataset = assistant.load_my_dataset(responses)

    # Verify dataset types
    assert isinstance(train_dataset, tf.data.Dataset), "Train dataset not loaded correctly."
    assert isinstance(test_dataset, tf.data.Dataset), "Test dataset not loaded correctly."
    assert isinstance(val_dataset, tf.data.Dataset), "Validation dataset not loaded correctly."

    # Example additional checks: count the batches in one of the datasets
    # Assuming at least one image per class per dataset part, and a batch size of 32
    assert len(list(train_dataset)) > 0, "Train dataset should contain at least one batch."
    # Similarly for test_dataset and val_dataset if needed

# Example test for the choice that leads to Optuna optimization

@pytest.fixture
def setup_environment():
    assistant = CodingAssistant()
    responses = [32, 10, "path/to/dataset", 3]  # Example responses
    n_trials = 5

    # Simulate train and validation datasets
    train_dataset = MagicMock()
    val_dataset = MagicMock()

    return assistant, responses, n_trials, train_dataset, val_dataset


@patch('codingAssistant.CodingAssistant.choose_cnn_architecture')
def test_setup_random_search(mock_choose_cnn_architecture, setup_environment):
    assistant = CodingAssistant()
    assistant, responses, n_trials, train_dataset, val_dataset = setup_environment

    # Mock the choose_cnn_architecture method to return a model with mocked fit and evaluate methods
    mock_model = MagicMock()
    mock_model.fit.return_value = None  # Assuming fit doesn't need to return anything specific
    mock_model.evaluate.return_value = (0.5, np.random.rand())  # Random accuracy for each call
    mock_choose_cnn_architecture.return_value = mock_model

    best_hyperparameters = assistant.setup_random_search(responses, n_trials, train_dataset, val_dataset)

    # Verify that choose_cnn_architecture was called n_trials times
    assert mock_choose_cnn_architecture.call_count == n_trials, "choose_cnn_architecture not called the expected number of times."

    # Verify keys in best_hyperparameters
    assert set(best_hyperparameters.keys()) == {'filters', 'dropout_rate',
                                                'learning_rate'}, "Missing expected hyperparameters."

    # Optionally, verify that best_hyperparameters values are from the expected options
    assert best_hyperparameters['filters'] in [16, 32, 64, 128], "Filters not selected from the expected options."
    assert 0.0 <= best_hyperparameters['dropout_rate'] <= 0.5, "Dropout rate not within the expected range."
    assert best_hyperparameters['learning_rate'] in [1e-5, 1e-4, 1e-3,
                                                     1e-2], "Learning rate not selected from the expected options."
