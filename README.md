# sustAGE Project - Time Series Forecasting with LSTM and GRU Models

![sustage](https://github.com/gpaps/TS_forecasting/assets/29929836/280d8248-045a-40cb-a037-e1c7877c8078)

### Overview

__This repository contains code developed for the European Horizon project called sustAGE. The code focuses on time series forecasting using LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) models. The goal is to predict future values based on historical data.__

__Code Description__

__The code is written in Python and utilizes various libraries, including TensorFlow, NumPy, and Matplotlib. Here's a brief overview of the code and its functionality:__

    Data Preparation: The code starts by loading data from a CSV file (userData_HR60.csv) containing time series data. It extracts heart rate data and timestamps from the file, preparing them for further processing.

    Data Preprocessing: The heart rate data is scaled using Min-Max scaling, which maps the values to a range between 0 and 1. This preprocessing step is crucial for optimizing the training of the neural network.

    Sequence Splitting: The split_sequence function is used to split the data into input sequences (X) and corresponding target values (y). It also ensures that the data is consistent in terms of timestamps and removes any inconsistent sequences.

    Model Building: The code defines a neural network model using TensorFlow's Keras API. It creates multiple LSTM layers with dropout regularization to prevent overfitting. The model is configured for training using the Mean Absolute Error (MAE) loss function and the Adam optimizer.

    Model Training: The model is trained on the prepared data with a specified number of epochs. Callbacks are used to monitor training progress, including early stopping, model checkpointing, and tensorboard for visualization.

    Model Evaluation: After training, the model is used to make predictions on the training data. The predictions are plotted alongside the ground truth values to visualize the model's performance.

    Multiple Predictions: The code allows for multiple predictions by changing the pred_distance variable. This enables forecasting at different time horizons.

### Usage

__To use this code for your own time series forecasting tasks, follow these steps:__

    Ensure you have the required Python libraries installed, including TensorFlow, NumPy, and Matplotlib.

    Replace the data source (userData_HR60.csv) with your own time series data in CSV format. Ensure that the data file contains a timestamp and the values to be forecasted.

    Adjust the n_steps and pred_distance variables according to your specific forecasting needs. n_steps defines the window length of each sequence, and pred_distance determines how far into the future the predictions will be made.

    Customize the model architecture in the build_model function if necessary.

    Run the code, and it will train the model on your data and provide predictions.

__Contact__
_If you have any questions or need further assistance with this code or the sustAGE project, please feel free to contact George (gpaps@ics.forth.gr)._
