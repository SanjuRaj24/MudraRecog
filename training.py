import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import json


img_size = 224
batch_size = 32

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "dataset/training", target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    "dataset/validation", target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical"
)

# Class Indices
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

print("Saved class indices to class_indices.json")

# Load Pretrained MobileNetV2 (Exclude Top Layers)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  

# Custom Classification Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Efficient feature extraction
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)  # Reduce overfitting
output_layer = Dense(len(train_generator.class_indices), activation="softmax")(x)

# Define Model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Add Early Stopping
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Train Model (First Phase - Frozen Base Model)
print("\n Training Model with Frozen Layers")
model.fit(train_generator, validation_data=val_generator, epochs=15, callbacks=[early_stopping])

# Save Initial Model
model.save("mudra_mobilenetv2_initial.keras")
print("Initial training complete. Model saved as mudra_mobilenetv2_initial.keras")


# Unfreezing the last few layes for fine tuning
print("\n Unfreezing last 10 layers and fine-tuning...")
base_model.trainable = True  # Unfreeze all layers

for layer in base_model.layers[:-10]:  # Keep early layers frozen
    layer.trainable = False

# compile with lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# Train Again for Fine-Tuning
model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[early_stopping])

# Save the Fine-Tuned Model
model.save("mudra_mobilenetv2_finetuned.keras")
print("Fine-tuning complete. Model saved as mudra_mobilenetv2_finetuned.keras")
