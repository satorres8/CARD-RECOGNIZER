import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Ajusta la lista de clases a tu dataset (mismo orden que tenías en train/test)
class_names = ["A_clubs", "A_diamonds", "A_hearts", "A_spades"]

# Carga el modelo (asegúrate de que el archivo exista en el mismo directorio o ajusta la ruta)
model_path = "card_recognizer.keras"
model = tf.keras.models.load_model(model_path)

def load_and_preprocess_image(image_path, img_size=(256, 256)):
    """Carga y preprocesa la imagen para MobileNetV2."""
    img = Image.open(image_path).convert('RGB')  # Asegurarse de que esté en RGB
    img = img.resize(img_size)                   # Redimensiona a 256x256
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Si usaste preprocess_input de MobileNetV2 en el entrenamiento:
    img_array = preprocess_input(img_array)
    return img_array

def predict_image():
    """Abre el explorador de archivos, selecciona imagen y predice la clase."""
    # Abre el cuadro de diálogo para seleccionar archivo
    image_path = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.gif")]
    )
    if not image_path:
        return  # Usuario canceló la selección

    # Carga y preprocesa la imagen para la predicción
    input_data = load_and_preprocess_image(image_path)

    # Hacer la predicción
    predictions = model.predict(input_data)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # Muestra el resultado en la etiqueta de texto
    result_text.set(f"Imagen: {os.path.basename(image_path)}\n"
                    f"Predicción: {predicted_class}\n"
                    f"Confianza: {confidence:.2f}%")

    # Mostrar la imagen seleccionada en la interfaz
    # (la redimensionamos para que encaje mejor en la GUI, por ejemplo a 200x200)
    pil_image = Image.open(image_path).convert('RGB')
    pil_image = pil_image.resize((200, 200))
    tk_image = ImageTk.PhotoImage(pil_image)

    # Actualizamos el label de la imagen
    label_image.config(image=tk_image)
    # Necesario para evitar que la imagen sea recolectada por el garbage collector
    label_image.image = tk_image

# Interfaz gráfica con Tkinter
root = tk.Tk()
root.title("Detector de Ases")

# Frame principal
frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

# Botón para seleccionar la imagen y predecir
btn_select_image = tk.Button(frame, text="Seleccionar imagen", command=predict_image)
btn_select_image.pack(pady=10)

# Label para mostrar la imagen
label_image = tk.Label(frame)
label_image.pack(pady=10)

# Label para mostrar el resultado (texto)
result_text = tk.StringVar()
label_result = tk.Label(frame, textvariable=result_text, width=40, height=5, anchor='center')
label_result.pack(pady=10)

root.mainloop()
