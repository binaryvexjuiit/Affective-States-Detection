
from tensorflow import keras
import tensorflow as tf # Imports tensorflow
import tensorflow_addons as tfa


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D,LSTM
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tensorflow.keras.metrics import Recall,Precision,AUC,TruePositives,TrueNegatives,FalseNegatives,FalsePositives
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import cv2
from glob import glob



img_shape = (48, 48, 3)
train_dir="MMAFEDB/train"
test_dir="MMAFEDB/test"
Name = "CNN-LSTM"
train_image = []
lebel = []
rel_dirname = os.path.dirname(__file__)
    
for dirname in os.listdir(os.path.join(rel_dirname, train_dir)):
    i=0
    for filename in glob(os.path.join(rel_dirname, train_dir+'/'+dirname+'/*.jpg')):
             img = image.load_img(os.path.join(rel_dirname, filename),target_size=img_shape)
             img = image.img_to_array(img)
             img = img/255.0
             train_image.append(img)
             lebel.append(dirname)
             i=i+1
             if i == 4500:
                 break
                 
X_train = np.array(train_image)
lebel = np.array(lebel)
y_train=to_categorical(lebel)
#y_train= np.delete(y_train,0,1)
print(X_train.shape)
print(y_train.shape)

test_image = []
label = []
for dirname in os.listdir(os.path.join(rel_dirname, test_dir)):
    i=0
    for filename in glob(os.path.join(rel_dirname, test_dir+'/'+dirname+'/*.jpg')):
             img = image.load_img(os.path.join(rel_dirname, filename),target_size=img_shape)
             img = image.img_to_array(img)
             img = img/255.0
             test_image.append(img)
             label.append(dirname)
             if i == 700:
                 break

X_test = np.array(test_image)
label = np.array(label)
y_test = to_categorical(label)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(48,48,3)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(filters=32, kernel_size=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

model.add(Conv2D(filters=48, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(Conv2D(filters=48, kernel_size=(2, 2), activation='relu', padding='valid'))
model.add(Conv2D(filters=48, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(filters=48, kernel_size=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(filters=64, kernel_size=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))
model.add(layers.Reshape(target_shape=(16,64)))
#model.add((LSTM(96,dropout=0.1,return_sequences=True)))
#model.add((LSTM(64,dropout=0.1,return_sequences=True)))

model.add(layers.GlobalAveragePooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(7, activation='softmax'))
print(model.summary())
#plot_model(model, to_file=Name+'.png',show_shapes= True , show_layer_names=True)
model.compile(optimizer= keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True), 
              metrics=['acc',Recall(),Precision(),AUC(),
                       TruePositives(),TrueNegatives(),FalseNegatives(),FalsePositives()])
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=64)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower left')
plt.savefig(Name+'acc.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(Name+'loss.png')
plt.show()



model.save(Name+'.h5')

pd.DataFrame.from_dict(history.history).to_csv(Name+'.csv',index=False)