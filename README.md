# Cassava Leaf Disease Classification

This project implements a deep learning solution for classifying Cassava leaf diseases using PyTorch and the ConvNext architecture. The model is trained to identify different types of diseases affecting Cassava plants from leaf images.

## Project Overview

The project uses a ConvNext-based classification model to identify 5 different classes of Cassava leaf diseases. The implementation includes data augmentation, learning rate adjustment, and early stopping mechanisms to optimize model performance.

## Dataset

The project uses the [Cassava Leaf Disease Classification dataset](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data) from Kaggle, which contains:
- Images of Cassava leaves
- A CSV file containing image names and their corresponding labels
- 5 different disease classes:
  1. Cassava Bacterial Blight (CBB)
  2. Cassava Brown Streak Disease (CBSD)
  3. Cassava Green Mottle (CGM)
  4. Cassava Mosaic Disease (CMD)
  5. Healthy

The dataset is used to train a model that can accurately classify these different conditions affecting Cassava plants.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- pandas
- PIL (Python Imaging Library)
- tqdm

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dhaval-zala/CassavaLeafDisease
cd CassavaLeafDisease
```

2. Install the required packages:
```bash
pip install torch torchvision pandas pillow tqdm
```

## Project Structure

```
CassavaLeafDisease/
├── train.py              # Main training script
├── ClassModels.py        # Model architecture definitions
├── dataset/             # Dataset directory
│   └── CassavaLeafDisease-data/
│       ├── images/      # Image files
│       └── label.csv    # Labels file
└── README.md
```

## Configuration

The main configuration parameters in `train.py` include:

- `batch_size`: 64
- `learning_rate`: 0.0001
- `n_epochs`: 20
- `num_classes`: 5
- `image_size`: (96, 96)
- `train_split_ratio`: 0.8

## Model Architecture

The project uses a ConvNext-based classification model with the following features:
- Pre-trained weights
- Configurable number of layers
- Option to freeze layers during training

## Training Process

The training process includes:

1. **Data Preparation**:
   - Image resizing
   - Data augmentation (random horizontal flips and rotations)
   - Train-test split (80-20)

2. **Training Features**:
   - Learning rate adjustment based on performance
   - Early stopping mechanism
   - Best model weight preservation
   - Progress tracking and user interaction

3. **Evaluation**:
   - Regular evaluation on test set
   - Accuracy tracking
   - Loss monitoring

## Usage

To train the model:

```bash
python train.py
```

The training process will:
1. Load and preprocess the dataset
2. Initialize the model
3. Train for the specified number of epochs
4. Save the best model weights
5. Provide training progress and evaluation metrics

## Training Features

- **Learning Rate Adjustment**: Automatically adjusts learning rate when performance plateaus
- **Early Stopping**: Stops training if no improvement is seen after specified patience
- **Best Model Preservation**: Saves the best performing model weights
- **Interactive Training**: Allows user to decide whether to continue training after specified epochs

## Learning Rate Adjuster Parameters

The `LearningRateAdjuster` class implements an adaptive learning rate mechanism with the following parameters:

- `patience` (int): Number of epochs to wait before adjusting learning rate if no improvement (default: 3)
- `stop_patience` (int): Maximum number of learning rate adjustments before stopping training (default: 5)
- `threshold` (float): Accuracy threshold that triggers learning rate adjustment (default: 97.0)
- `factor` (float): Factor by which learning rate is multiplied during adjustment (default: 0.5)
- `dwell` (bool): Whether to restore best weights after learning rate adjustment (default: True)
- `model_name` (str): Name of the model for logging purposes (default: "ConvNext")
- `batches` (int): Total number of batches in training set
- `epochs` (int): Total number of epochs for training
- `ask_epoch` (int): Epoch number after which user will be prompted to continue training (default: 10)

The adjuster monitors training progress and:
1. Tracks the best model weights
2. Adjusts learning rate when performance plateaus
3. Implements early stopping if no improvement is seen
4. Allows user interaction during training
5. Restores best weights when needed

## Performance Monitoring

The training process provides:
- Training accuracy per epoch
- Test accuracy per epoch
- Loss values
- Learning rate adjustments
- Early stopping notifications

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Add your license information here]

## Acknowledgments

- The Cassava Leaf Disease dataset
- PyTorch team for the deep learning framework
- ConvNext architecture developers 
