import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models, regularizers
from keras.src.callbacks import EarlyStopping

# Cargar los datasets
train_dataset = keras.utils.image_dataset_from_directory(
    'data',
    color_mode='grayscale',  # Asegurarse de que las imágenes sean cargadas en blanco y negro
    validation_split=0.3,
    batch_size=32,
    labels='inferred',
    label_mode='binary',
    shuffle=True,
    subset='training',
    seed=123
)

validation_dataset = keras.utils.image_dataset_from_directory(
    'data',
    color_mode='grayscale',  # Asegurarse de que las imágenes sean cargadas en blanco y negro
    validation_split=0.3,
    batch_size=32,
    labels='inferred',
    label_mode='binary',
    shuffle=True,
    subset='validation',
    seed=123
)

# Definir el aumento de datos
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

model = models.Sequential([
    layers.Input(shape=(256, 256, 1)),
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(train_dataset, validation_data=validation_dataset, epochs=100, callbacks=[early_stopping])

model.save('model_latest_latest.keras')

# Visualizar la precisión y la pérdida
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
