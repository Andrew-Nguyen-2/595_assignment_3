from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

if __name__ == '__main__':
    # get all data from dataset
    data = pd.read_csv("iris/iris.data")
    # split into samples, labels
    X = data.iloc[:, 0:4].values
    Y = data.iloc[:, 4].values
    # convert classes from strings to integer value
    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)
    # do one hot encoding
    Y = pd.get_dummies(Y).values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)

    model = Sequential()
    model.add(Dense(25, input_shape=(4,), activation='gelu'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=150)
    y_pred = model.predict(X_test)
    predictions = np.argmax(y_pred, axis=1)

    Y_test_one_label = []
    for i in range(len(Y_test)):
        Y_test_one_label.append(np.where(Y_test[i] == 1)[0][0])

    precision = precision_score(Y_test_one_label, predictions, average=None)
    recall = recall_score(Y_test_one_label, predictions, average=None)
    f1 = f1_score(Y_test_one_label, predictions, average=None)
    print("precision:", precision, "recall:", recall, "f1:", f1)
