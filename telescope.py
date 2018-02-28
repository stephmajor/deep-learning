from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout
from keras import optimizers
from keras import regularizers
import numpy

# replace all values x with their z score (by column)
def standardize_by_col(mat):
    avg = numpy.mean(mat, axis = 0)
    std = numpy.std(mat, axis = 0)
    final = (mat - avg) / std
    return final

def class_converter(v):
    if v == b'g':
        return 0.0
    else:
        return 1.0

# fix random seed for reproducibility
numpy.random.seed(13)

# load the spam base data
dataset = numpy.genfromtxt("MagicTelescope.csv", skip_header = 2, delimiter = ",", converters = { 11 : class_converter })
standardized = standardize_by_col(dataset[:,0:11])

# split into input (X) and output (Y) variables
X_train = standardized[0:18000,1:11]
Y_train = dataset[0:18000,11]

X_test = standardized[18000:,1:11]
Y_test = dataset[18000:,11]

print("training shape =", X_train.shape)
print("testing shape =", X_test.shape)

# setup the shape of the input, based on the first training example
X_input = Input(shape = X_train[0].shape)

# first dense layer
X = Dense(128, activation = 'relu', name='dense1')(X_input)
X = Dropout(0.1, name='dropout1')(X)

# second dense layer
X = Dense(64, activation = 'relu', name='dense2')(X)
X = Dropout(0.1, name='dropout2')(X)

# third dense layer
X = Dense(32, activation = 'relu', name='dense3')(X)

# final sigmoid for binary classification
X = Dense(1, activation = 'sigmoid', name='classifier')(X)

# construct the model with prescribed layers
model = Model(inputs = X_input, outputs = X, name='telescope')

# compile and then train (fit) our model
model.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(x = X_train, y = Y_train, epochs = 100, batch_size = 128)

# evaluate the model
print()
scores = model.evaluate(X_train, Y_train)
print ("Train Loss = " + str(scores[0]))
print ("Train Accuracy = " + str(scores[1]))

print()
scores = model.evaluate(X_test, Y_test)
print ("Test Loss = " + str(scores[0]))
print ("Test Accuracy = " + str(scores[1]))
