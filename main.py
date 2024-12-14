import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers,regularizers
from keras import Sequential
from keras.models import Model
from keras.applications import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set a memory limit (e.g., 4096 MB = 4 GB)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)


base_dir = 'PlantVillage'


img_size = 224
batch_size = 16

#data gen

data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    
)

#train Generator

train_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(img_size,img_size),
    batch_size=batch_size,
    subset="training",
    class_mode='categorical'
)


#validation Generator
validation_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(img_size,img_size),
    batch_size=batch_size,
    subset="validation",
    class_mode='categorical'
)

#model Definition

model = Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(img_size,img_size,3)))
model.add(layers.MaxPool2D(2,2))

model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.MaxPool2D(2,2))

model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D(2,2))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPool2D(2,2))


model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu')) 
model.add(layers.Dense(train_generator.num_classes, activation='softmax',kernel_regularizer=regularizers.l2(0.001)))

model.summary()


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
model.summary()

history = model.fit(train_generator, 
                    validation_data=validation_generator, 
                    epochs=5, 
                )
# model eval

print("Evaluation model...")
val_loss, val_accuracy = model.evaluate(validation_generator, steps= validation_generator.samples // batch_size)
print(f"validation Accuracy: {val_accuracy*100:.2f}%")
history_dict = history.history

model.save('Plant Disease Prediction v7.h5')
with open('training_history v7.json', 'w') as f:
    json.dump(history_dict, f)

# plot training & validation accuracy val
with open('training_history v6.json', 'r') as f:
    loaded_history = json.load(f)
plt.plot(loaded_history['accuracy'])
plt.plot(loaded_history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

#  plot training & validation loss value

plt.plot(loaded_history['loss'])
plt.plot(loaded_history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


# model = Model.load_model('Plant Disease Prediction v6.h5')


def load_and_process_img(img_path,target_size=(224,224)) :
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array,axis=0)
    img_array = img_array.astype('float32')/255.

    return img_array

def predict_image_class(model, img_path, class_indices) :
    preprocessed_img = load_and_process_img(img_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name

class_indices = {v:k for k ,v in train_generator.class_indices.items()}
print(class_indices)


json.dump(class_indices, open('class_indices.json','w'))

image_path = "plantvillage dataset\color\Corn_(maize)___Common_rust_\RS_Rust 1563.JPG"
image = mpimg.imread(image_path)
plt.imshow(image)
plt.show()



print(predict_image_class(model,image_path,class_indices))