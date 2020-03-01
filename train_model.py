import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import keras
from keras.utils import to_categorical

from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
x_data = np.load('D:/samples.npy')
y_data = np.load('D:/labels.npy')
x_test = np.load('D:/test_samples.npy')
y_test = np.load('D:/test_labels.npy')

# x_validate, x_further, y_validate, y_further = train_test_split(x_test, y_test, test_size = 0.5)
earlystop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.01,
    patience=3,
    verbose=1,
    mode='auto'
)

reduce = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1, 
    mode='auto'
)

callbacks = [reduce, earlystop]
# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        zoom_range=0.2,
        width_shift_range=0.15,
        height_shift_range=0.15,
        rotation_range=17,
        fill_mode='constant',
        cval=0)

test_datagen = ImageDataGenerator(zoom_range=0.2, fill_mode='constant', cval=0)

train_datagen.fit(x_data, augment=True)
test_datagen.fit(x_test, augment=True)

train_batch_size=64
test_batch_size=64

train_generator = train_datagen.flow(x_data, y_data, batch_size=train_batch_size)

validation_generator = test_datagen.flow(x_test, y_test, batch_size=test_batch_size)

model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding = 'same', activation='relu', input_shape=(128, 192, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(96, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(16, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(lr=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

history = model.fit(train_generator, epochs=15, steps_per_epoch=len(x_data)//train_batch_size, verbose=1, validation_data=validation_generator, validation_steps=len(x_test)//test_batch_size, callbacks=callbacks)

# Loss Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)
plt.show()

# Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['acc'], 'r', linewidth=3.0)
plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
plt.show()

model.save('hand_rec_model_crossentropy.h5')

model2 = models.Sequential()
model2.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding = 'same', activation='relu', input_shape=(128, 192, 1)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(96, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Flatten())
model2.add(layers.Dense(512, activation='relu'))
model2.add(layers.Dropout(0.3))
model2.add(layers.Dense(16, activation='softmax'))
# Tried:  RMSProp with default hyperparameters 
#         Adam with default hyperparameters
#         Adam with lr=0.00146

model2.compile(keras.optimizers.RMSprop(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model2.summary())

history2 = model2.fit(train_generator, epochs=15, steps_per_epoch=len(x_data)//train_batch_size, verbose=1, validation_data=validation_generator, validation_steps=len(x_test)//test_batch_size, callbacks=callbacks)

# Loss Curves
plt.figure(figsize=[8, 6])
plt.plot(history2.history['loss'], 'r', linewidth=3.0)
plt.plot(history2.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)
plt.show()

# Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(history2.history['acc'], 'r', linewidth=3.0)
plt.plot(history2.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
plt.show()

model2.save('hand_rec_model_crossentropy2.h5')