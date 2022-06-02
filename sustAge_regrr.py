import os

import datetime
import numpy as np
import tensorflow as tf
import tensorflow.python
from matplotlib import pyplot as plt
from pandas import pandas as pd, Series

tensorflow.python.enable_eager_execution()
print("eager execution result", tensorflow.python.executing_eagerly())
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, GRU

# PWD for results
# Create a directory for logs files
my_dir = os.path.join("/media/gpaps/ilapla/sustAGE_logs/")

input_data = pd.read_csv('/home/gpaps/Framework/Lstm/deploy_data/CRF_HR-HRV/userData_HR60.csv', sep=';', skiprows=0)

# start, pastlength = 0, 25
# hr_data = input_data.iloc[start-pastlength:, 2].values ; hr_data = np.array(hr_data, dtype=float)
hr_data = input_data.iloc[:, 6].values
hr_data = np.array(hr_data, dtype=float)
hr_data = hr_data.reshape(hr_data.shape[0], 1)

timestamp = input_data.iloc[:, 3].values // 1000


def split_sequence(sequence, n_steps, timesseq, pred_distance):
    X, y, badlist, t_norm = list(), list(), list(), list()

    # print(len(sequence)-pred_distantce-n_steps)
    for i in range(len(sequence) - pred_distance - n_steps):
        # grab the i-th and find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break

        # gather input and output parts of the pattern+
        if end_ix + pred_distance < len(sequence):
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix + pred_distance]

        seq_t, seq_ttt = timesseq[i:end_ix], timesseq[i:end_ix] - timesseq[i]
        totdif, count, groupok = 0, 0, 1
        for k in range(len(seq_t) - 1):
            count = count + 1
            dif = (seq_t[k + 1] - seq_t[k])
            totdif = totdif + dif
            if dif > 300:  # checks time-consistency between values
                groupok = 0
        if groupok == 1 or totdif / count > 120:  # 120 secs
            X.append(seq_x), y.append(seq_y), t_norm.append(seq_ttt)
        else:
            badlist.append(seq_x)

    return np.array(X), np.array(y), np.array(t_norm)


# scaling values is good for optimizing the training of the network
scaler = MinMaxScaler(feature_range=(0, 1))
X = Series(hr_data[:, 0]).values
X = X.reshape((len(X), 1))
scaler.fit(X)
Xtr = scaler.transform(X)

# define the window-length of each sequence or choose a number of time-steps, <== n_steps
# create list of how far prediction's will be, (depends on type of input-data)
n_steps, pred_distance = 25, [10]  # [10, 20, 30]


# define model
def build_model():
    global model, X_train, y, y_out
    for i in pred_distance:
        for k in range(2):  # changing the value, runs the train_sess for the given pred_distance k*times
            time0 = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            Xi, y, T_norm = split_sequence(Xtr, n_steps, np.ones_like(timestamp), i)
            Xi = Xi.reshape(Xi.shape[0], Xi.shape[1])
            X = np.column_stack((Xi,))  # prev change [:, -1]
            y = y.reshape(1, -1)
            y_train = np.column_stack(y)

            n_features = 1
            input_shape = (n_steps, n_features)
            """ reshape from a 2D-input[samples, time-steps]  --to--->3D-input  [samples, time-steps, features] """
            X_train = X.reshape((X.shape[0], X.shape[1], n_features))

            # model
            model = Sequential()
            model.add(LSTM(units=16, return_sequences=True,
                           input_shape=input_shape))  # ret a sequen. of vectors of dim f{UNTITS}
            model.add(Dropout(0.1))
            model.add(LSTM(units=10, return_sequences=True))  # ret a seque. of vectors of dimension UNITS
            model.add(Dropout(0.1))
            model.add(LSTM(units=16, return_sequences=False))
            model.add(Dropout(0.1))
            model.add((Dense(1)))

            # configures model training
            tf.keras.optimizers.Adam(learning_rate=0.0001)
            model.compile(optimizer='adam', loss="mae", metrics=['mae'])
            print(model.summary())

            # Callbacks are used for observation and debug purposes were we store,
            # EarlyStopping Stop training when a monitored metric has stopped improving.
            # ModelCheckpoint creates h5 for every training epoch, potentially user will restart train from the desired
            # Tensorboard has many usages on of those is to observe if the model fits while train occurs,
            # CSVLogger store's in csv-format type training information,
            my_callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=20, min_delta=0.0001),  # restore_best_weights=True),

                tf.keras.callbacks.ModelCheckpoint(filepath=my_dir + 'logs/fit/' + time0 + 'lstmodel1.h5',
                                                   monitor='mae',
                                                   # 'model.{epoch:02d}-{val_loss:.2f}.h5' # for iter # ommit this block
                                                   verbose=2, save_best_only=True, mode='min'),

                tf.keras.callbacks.TensorBoard(log_dir=my_dir + 'logs/fit/' + time0, histogram_freq=1, ),

                tf.keras.callbacks.CSVLogger(filename=(my_dir + 'train_logs/' + str(time0) + '_' + f'{i}' + '.csv'),
                                             separator=',', append=False),
            ]

            # fit model - start's the training session
            history = model.fit(X_train,
                                y_train,
                                epochs=400,
                                validation_split=0.20,  # the last 20% of our data(series) is for prediction
                                verbose=2,
                                # callbacks=[my_callbacks]  # uncomment for callbacks, see above for explanation
                                )

            # predict - model predicts based on train data
            y_out = model.predict(X_train, verbose=1)
            print(f"y_out.shape{y_out.shape}")

            # plot train with History
            plt.semilogy(history.history['loss'], label='Train Loss')
            plt.semilogy(history.history['val_loss'], label='Val Loss/test')
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.legend(['train', 'validation'], loc='upper right')
            plt.savefig(my_dir + 'pred_img/' + time0 + '_pred_ahead' + str(i) + '__loss_val_loss.png')
            plt.show()
            # plots input(y_trained) / output(y_out)
            plt.plot(y.reshape(-1, 1), 'r'), plt.plot(y_out, 'b'), plt.title('y_train \n y_out'),
            plt.legend(['Groundtruth', 'Predicted'], loc='upper right')
            plt.savefig(my_dir + 'pred_img/' + time0 + '_pred_ahead' + str(i) + '_prediction.png')
            plt.show()

            # save model's h5 for using it later to predict!
            # model.save('/home/_chooseYour_Directory/h5_dump/lstm_petra/test13.h5')

    return model, X_train, y, y_out


model, X, y, y_o = build_model()
