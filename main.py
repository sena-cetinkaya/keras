# import "numpy" and other "keras" libraries.
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# We downloaded  the "pima-indians-diabetes.csv" dataset.
data = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Getting the input (X) and output (y) values.
x = data[:,0:8]
y = data[:,8]

# Creating a 3-layer keras model.
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile the keras model.
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Fit the keras model on the dataset.
model.fit(x, y, epochs=150, batch_size=10)

# Learning the accuracy of the keras model.
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))


