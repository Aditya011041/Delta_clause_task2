import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()

# plt.imshow(X_train[2])

X_train = X_train/255
X_test = X_test/255

model = Sequential()

model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=50,validation_split=0.2)

model.save('mnist_model.h5')


y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)

accuracy_score(y_test,y_pred)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

model.predict(X_test[1].reshape(1,28,28)).argmax(axis=1)
