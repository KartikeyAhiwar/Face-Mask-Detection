# train_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Parameters
BATCH_SIZE = 32
EPOCHS = 20
IMG_SIZE = (224, 224)
DATASET_DIR = "dataset"

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15, 
                                   width_shift_range=0.2, height_shift_range=0.2, 
                                   shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Load MobileNetV2 and build model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freezing base model layers
for layer in base_model.layers:
    layer.trainable = False

head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten()(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(1, activation="sigmoid")(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

# Compile model
opt = Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(train_generator, epochs=EPOCHS)

# Save model
model.save("model/mask_detector.model")
