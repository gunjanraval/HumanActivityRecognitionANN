import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as plt
import keras
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('FirstDraft3.csv')
X = dataset.iloc[:, 0:559].values
y = dataset.iloc[:, 560].values
y=y-1
y_train = keras.utils.to_categorical(y)

X = dataset.iloc[:, 0:559]
y = dataset.iloc[:, 560]


XT = dataset.iloc[:, 0:559].values
yT = dataset.iloc[:, 560].values
yT=yT-1
y_test = keras.utils.to_categorical(yT)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(X)
x_test = sc.fit_transform(XT)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=559))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)



y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.values.argmax(axis=1), y_pred.values.argmax(axis=1))
