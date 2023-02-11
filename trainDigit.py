# training the model
import keras
from keras.datasets import mnist # 60000 0-9
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K

# the data, splitting the train from the test sets
(train_x, train_y) ,(test_x, test_y) = mnist.load_data()

print(train_x.shape, train_y.shape)

train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
input_shape = (28,28,1)

# convert class vectors to binary class matrices 
train_y = keras.utils.to_categorical(train_y, 10) # 10 because the output are from 0 to 9
test_y = keras.utils.to_categorical(test_y, 10)

train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_x/=255
test_x/=255
print('train_x shape:', train_x.shape)
print(train_x.shape[0], 'train samples')
print(test_x.shape[0], 'test samples')

batchSize = 128
numClasses = 10
epochs = 3 # the best one that makes accurate data i got yet

model = Sequential() # calling the constructer sequential to make sequential object
model.add(Conv2D(32, kernel_size=(5,5), activation='relu',input_shape=input_shape)) # remember input_shape = (28,28,1)
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) # to make variance and avoid overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(numClasses, activation='softmax'))

model.compile(loss=keras.losses.CategoricalCrossentropy(),optimizer=RMSprop(),metrics=['accuracy']) # when i tried ada delta optimizer the accuracy was wayyyy down then i tried the RMSprop optimizer and it is great!
hist = model.fit(train_x, train_y, batch_size=batchSize, epochs=epochs, verbose=1, validation_data=(test_x,test_y))
print('The model has successfully trained!')

score = model.evaluate(test_x, test_y, verbose=0)
print('Total loss:', score[0])
print('Test accuracy:', score[1])

model.save('mnist.h5')
print('Saving the model as mnist.h5')