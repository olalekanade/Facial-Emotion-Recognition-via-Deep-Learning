# Facial Emotion Recognition via Deep Learning

This project implements a real-time facial emotion recognition system using a pre-trained Convolutional Neural Network (CNN). The system detects faces from live video input and predicts the emotions displayed, based on seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Project Structure

- **Model Training.ipynb**: Contains the code used to train the CNN model and the evaluation metrics. This file does not need to be run again, as the trained model is already saved.
- **best_model.keras**: The pre-trained model file that is used for making live predictions.
- **Prediction.py**: The Python script that uses OpenCV to run real-time emotion recognition with the webcam.

## Setup Instructions

### Installation
1. Install the necessary dependencies by running the following commands:
   ```bash
   pip install numpy
   pip install opencv-python
   pip install tensorflow
   ```

### Running the Real-Time Emotion Recognition
To start the facial emotion recognition system, ensure your webcam is connected, then execute the following command in your terminal:
```bash
python Prediction.py
```

The system will access your webcam, detect your face, and display the predicted emotion in real-time.

## Features

- **Real-Time Emotion Detection**: Detects and classifies emotions in real-time from webcam input.
- **Facial Emotion Categories**: Recognizes seven basic emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
- **Model Architecture**: Uses a Convolutional Neural Network based on ResNet50, trained using the FER-2013 dataset.
