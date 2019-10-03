import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)


def train():
    print('Starting analysis...')

    # Read data from csv files
    print('Data read...', end='')
    # load the dataset
    dataframe = pd.read_csv('sffd.csv', usecols=[3], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    print(' OK')

    # fix random seed for reproducibility
    np.random.seed(7)

    epochs = [1, 5, 10, 20, 30]
    look_back = [1, 2, 7, 14, 21, 30, 60, 90]
    # epochs = [1, 2]
    # look_back = [1, 2]

    t0 = time.time()
    for ep in epochs:
        for lb in look_back:
            print('\nEpochs {} and look_back {}'.format(ep, lb))
            s_train(ep, lb, dataset)

    print('\nComputation completed in: {}'.format(round(time.time() - t0, 2)))


def s_train(epochs, look_back, dataset):
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print(len(train), len(test))

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    t0 = time.time()
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = LossHistory()
    model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=0, callbacks=[history])
    t1 = time.time()

    with open('stat/loss-{}e{}lb.txt'.format(epochs, look_back), 'w') as fh:
        print(history.losses, file=fh)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    # plot baseline and predictions
    plt.figure(figsize=(16, 12))
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.title('LSTM {} epochs - {} look_back'.format(epochs, look_back))
    plt.savefig('plot/LSTM-{}e{}lb.png'.format(epochs, look_back), dpi=300)
    plt.show()

    with open('stat/info.txt', 'a') as f:
        print(epochs, look_back, '{}'.format(round(trainScore, 2)), '{}'.format(round(testScore, 2)), '{}'.format(round((t1 - t0), 2)), 'model-{}e{}lb.h5'.format(epochs, look_back), file=f, sep='\t')

    # save model and architecture to single file
    model.save("models/model-{}e{}lb.h5".format(epochs, look_back))


def load():
    # load model
    model = load_model('model.h5')
    # summarize model.
    model.summary()
    print('model loaded')


def main():
    train()
    # load()


if __name__ == "__main__":
    main()
