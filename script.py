# %% [markdown]
# **GRASP-AND-LIFT EEG Detection Project**
#
# This project aims to compare different machine learning models in terms of viability for detecting events from EEG signals.

# %%
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.models import Sequential
import math
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne

# %%
SAMPLING_FREQUENCY = 500


def loadSubjectSeries(subject: int, series: int) -> mne.io.RawArray:
    pandata = pd.read_csv(f"data/train/subj{subject}_series{series}_data.csv")
    pandata = pandata.drop(["id"], axis=1)
    channel_names = list(pandata.keys())
    info = mne.create_info(
        ch_names=channel_names, ch_types=["eeg" for _ in range(len(channel_names))],
        sfreq=SAMPLING_FREQUENCY)

    data = mne.io.RawArray(pandata.transpose().to_numpy(), info)
    data.set_montage('easycap-M1')
    return data, channel_names


def icaTransform(data: mne.io.RawArray) -> mne.preprocessing.ICA:
    ica = mne.preprocessing.ICA(random_state=42)
    ica.fit(data)
    return ica


def searchIcaTemplates(modelIca, templateIndices, sampleIca):
    componentsToRemove = []
    for template in templateIndices:
        _, labels = mne.preprocessing.corrmap([modelIca, sampleIca], template=(0, template), plot=False)
        componentsToRemove.append(labels)
    return componentsToRemove


def applyIca(ica: mne.preprocessing.ICA, componentsToRemove, data: mne.io.RawArray) -> mne.io.RawArray:
    ica.exclude = componentsToRemove
    return ica.apply(data)


def bandFilter(data: mne.io.RawArray) -> mne.io.RawArray:
    return data.filter(13, 50)


# %%
# data, keys = loadSubjectSeries(1, 1)
# data = bandFilter(data)
# modelIca = icaTransform(data)

# # %%
# data.plot_sensors(show_names=True)

# # %%
# modelIca.plot_components()

# # %%
# components = [6, 20, 21]


def prepareAndFilterData(subject: int, series: int) -> np.ndarray:
    data, keys = loadSubjectSeries(subject, series)
    # data = bandFilter(data)
    # ica = icaTransform(data)
    # toExclude = searchIcaTemplates(modelIca, components, ica)
    # data = applyIca(ica, toExclude, data)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data.get_data().T)
    return pd.DataFrame(scaled, columns=keys)


# %%
train_set_labels = pd.read_csv("data/train/subj1_series1_events.csv")
train_set_labels.head()
labels = train_set_labels.columns.drop('id')
labelNames = labels.values

# %%
labels = train_set_labels.columns.drop('id')
labelNames = labels.values

# %%
train_set_signals = pd.read_csv("data/train/subj1_series1_data.csv")

train_set_complete = pd.concat([train_set_signals, train_set_labels], axis=1)
train_set_complete.insert(0, "order", range(0, len(train_set_complete)))

# %%


def highlight(indices, ax, color):
    i = 0
    while i < len(indices):
        ax.axvspan(indices[i]-0.5, indices[i]+0.5, facecolor=color, edgecolor='none', alpha=.4)
        i += 1


# %%
def vizualize_predictions(signals, predictions, expected, labelName, limit=2000):
    # 0-31
    signalIndex = 10

    # Relevant only for multilabel predictions, else is always 0
    labelIndex = 0

    signals = pd.DataFrame(data=np.array(signals))
    axis = signals[signals.columns[signalIndex]].iloc[0:limit].plot(figsize=(20, 4))

    expected = pd.DataFrame(data=expected)
    predictions = pd.DataFrame(data=np.around(predictions))

    expectedCropped = expected.iloc[0:limit, ]
    predictionsCropped = predictions.iloc[0:limit, ]

    highlight(expectedCropped[expectedCropped.iloc[:, labelIndex] == 1].index, axis, "red")
    highlight(predictionsCropped[predictionsCropped.iloc[:, labelIndex] == 1].index, axis, "black")

    red_patch = mpatches.Patch(color='red', label='Expected event')
    black_patch = mpatches.Patch(color='black', label='Predicted event')
    plt.legend(handles=[red_patch, black_patch])

    plt.title(labelName)
    plt.show()

# %% [markdown]
# Helper methods for loading data
#
# Features are standartized by removing the mean and scaling to unit variance
#
# Further preprocessing can be done in *prepare_signals* function


# %%


def prepareLabels(subject, series):
    data = pd.read_csv(f"data/train/subj{subject}_series{series}_events.csv")
    data = data.drop("id", axis=1)
    return data


def loadDataReady(subject, series):
    return prepareAndFilterData(subject, series), prepareLabels(subject, series)


# %% [markdown]
# Helper for printing success rates for given predictions and expected values

# %%
def transform(data):
    results = []
    lastOne = None
    lastEnd = -100
    MIN_GAP = 100
    MIN_WIDTH = 0

    for i, dato in enumerate(data):
        if dato == 1 and lastOne == None:
            lastOne = i

        elif dato == 0 and lastOne != None:
            if i-lastEnd < MIN_GAP:
                tmp = results[-1]
                results.pop()
                results.append((tmp[0], i))
            else:
                results.append((i, lastOne))

            lastOne = None
            lastEnd = i

    return results


def printSucc(predictions, expected, dataLabel):
    # success counters
    succ = 1
    onesTotal = 1
    onesSucc = 1

    predictions = np.round(predictions)

    ALLOWED_DEVIATION = 100

    tp = transform(predictions)
    te = transform(expected)

    succ = 0

    for p in tp:
        for ie, e in enumerate(te):
            if e[0] - ALLOWED_DEVIATION < p[0] < e[1] + ALLOWED_DEVIATION:
                succ += 1
                te.pop(ie)
                break

    print(f"Successfull {succ}, un {len(tp)-succ}")

    return succ/len(tp)


# %% [markdown]
# **RECURRENT NEURAL NETWORK**
#
# RNN with LSTM, dropout and activation layers
#
# * Adam optimizer
# * Binary crossentropy loss
# * Sigmoid activation layer

# %% [markdown]
# Transform 2D dataset to 3D for LSTM layer - add floating window of *look_back* length

# %%
def create_sequences(dataset, labels, look_back=1):
    dataX = []
    dataY = labels[look_back//2:-look_back//2]
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:(i+look_back), ])
    return np.array(dataX), np.array(dataY)


# %% [markdown]
# Tests given rnn model on predicting label or more labels from testing data.
#
# Provides visualisations and success rates.
#
#
#
#

# %%
DOWNSAMPLING = 20
LOOK_BACK = 50
BATCH_SIZE = 512
EPOCHS = 30

# %%


def evaluate_rnn(subject, model, label, draw):
    # Last (8th) series is used as testing data
    test_signals, test_labels = loadDataReady(subject=subject, series=8)

    # Creating sequences for lstm layer
    X_test_signals, X_test_labels = create_sequences(
        test_signals.values[::DOWNSAMPLING],
        test_labels.values[::DOWNSAMPLING],
        look_back=LOOK_BACK
    )
    # Selecting only desired labels
    X_test_labels = X_test_labels[:, label]

    # Last few data points that do not fit batch size are omitted
    croppedSize = math.floor(len(X_test_signals)/BATCH_SIZE)*BATCH_SIZE

    # Prediction for testing data
    predictions = model.predict(X_test_signals[0:croppedSize], batch_size=BATCH_SIZE)
    expected = X_test_labels[0:croppedSize]

    # Selecting only desired labels
    labelsPredicted = len(predictions[0])
    predictions = predictions[:, 0:1]

    # Success rate printing
    totalPercent = printSucc(predictions, expected, dataLabel="Testing")

    # Vizualization
    if (draw):
        vizualize_predictions(
            test_signals.values[::DOWNSAMPLING][LOOK_BACK//2:croppedSize+LOOK_BACK//2:],
            predictions,
            expected,
            labelName=labelNames[label],
            limit=10000
        )

    return totalPercent

# %% [markdown]
# Training of a single model for a single or more subjects.
#
# Uses series 1-7 (leaves series 8 for testing)


# %%

def train_rnn(subjects, labelToTrain, model, callbacks):
    # For specified subjects
    for subject in subjects:
        # For series 1-7
        for j in range(1, 8):    # TODO change
            signals, labels = loadDataReady(subject=subject, series=j)
            # Create sequences
            X_train_signals, X_train_labels = create_sequences(
                signals.iloc[::DOWNSAMPLING].values,
                labels.iloc[::DOWNSAMPLING].values,
                look_back=LOOK_BACK
            )

            X_train_labels = X_train_labels[:, labelToTrain]
            croppedSize = math.floor(len(X_train_signals)/BATCH_SIZE)*BATCH_SIZE
            # Train model on relevant label (calling fit repeatedly in keras doesnt reset the model)
            model.fit(
                X_train_signals[0:croppedSize],
                X_train_labels[0:croppedSize],
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                verbose=0,
                callbacks=callbacks
            )

            model.reset_states()

    print("FITTING DONE")
    return model

# %% [markdown]
# Training of model for given subjects and evaluating it.
#
# Evaluating each label success rate separarely and averaging them. Random behaviour has 50% success rate.
#
# For given subject, we use first 7 series as training data and 8'th series as test data.

# %%


def rnn_validation(model, trainSeparateLabels, callbacks, draw=False, subjectToEvaluateOn=1, label=0):
    print(labelNames[label], "label model evaluation")

    total = evaluate_rnn(subjectToEvaluateOn, model, label, draw=draw)

    print("SUMMARY")
    print("TOTAL :", total)

# %% [markdown]
# RNN configuration


# %%

callbacks = [EarlyStopping(monitor="accuracy", verbose=0, patience=10, restore_best_weights=True)]

# %% [markdown]
# Basic model, training all labels at once for single subject

# %% [markdown]
# Stacked model, training all labels separately for single subject

# %%
LABEL = 0

# %%
model = Sequential()
model.add(LSTM(50, batch_input_shape=(BATCH_SIZE, LOOK_BACK, 32),
          return_sequences=True, stateful=False, dropout=0.5, activation="softsign"))
model.add(LSTM(50, return_sequences=False, stateful=False, dropout=0.5, activation="softsign"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %%
# Separate models for training all labels separately
model = train_rnn(
    subjects=[1],
    labelToTrain=LABEL,
    model=model,
    callbacks=callbacks
)

# %%
rnn_validation(
    model=model,
    trainSeparateLabels=True,
    draw=True,
    subjectToEvaluateOn=10,
    callbacks=callbacks,
    label=LABEL
)

# %%
