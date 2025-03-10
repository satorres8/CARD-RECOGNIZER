import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# Ajusta según tus rutas
train_dir = "data/train"
test_dir = "data/test"

# Parámetros básicos
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 11  # Empieza con pocas épocas y luego ajusta

# 1. Cargar dataset usando image_dataset_from_directory
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',  # Devuelve etiquetas one-hot
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# 2. Aplicar preprocess_input de MobileNetV2
#    (Eliminamos la capa Rescaling y en su lugar usamos el mapeo a preprocess_input)
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

# 3. Definir un modelo base (MobileNetV2 pre-entrenado en ImageNet)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,  # Quitamos la parte de clasificación de ImageNet
    weights='imagenet'
)

# Congelamos las capas del modelo base para no reentrenarlas inicialmente
base_model.trainable = False

# 4. Crear la parte final que hará la clasificación de las cartas
#    Ajusta el número de clases a tu dataset real
NUM_CLASSES = 4  # Cambia a 54 si tienes 54 clases

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Entrenar el modelo
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds
)

# 6. Evaluar en test
loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc:.2f}")

# 7. Guardar el modelo
model.save("card_recognizer.keras")
