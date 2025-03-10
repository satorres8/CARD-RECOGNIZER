import os
import random
import shutil

# Subcarpetas (clases) que existen dentro de train y test
subcarpetas = ["A_clubs", "A_diamonds", "A_hearts", "A_spades"]

# Cantidad de imágenes a mover de cada subcarpeta
NUM_IMAGENES = 62

for carpeta in subcarpetas:
    # Rutas de origen y destino (asumiendo que ejecutas este script desde 'data')
    origen = os.path.join("train", carpeta)
    destino = os.path.join("test", carpeta)

    # Lista de todos los ficheros (imágenes) en la subcarpeta origen
    ficheros = os.listdir(origen)
    
    # Asegúrate de no intentar tomar más imágenes de las que hay
    if len(ficheros) < NUM_IMAGENES:
        print(f"¡Atención! La carpeta {origen} no tiene suficientes imágenes.")
        continue

    # Escoger aleatoriamente 62 ficheros
    elegidos = random.sample(ficheros, NUM_IMAGENES)

    # Moverlos a la carpeta destino
    for fichero in elegidos:
        ruta_origen = os.path.join(origen, fichero)
        ruta_destino = os.path.join(destino, fichero)
        shutil.move(ruta_origen, ruta_destino)

    print(f"Se han movido {NUM_IMAGENES} imágenes de {origen} a {destino}.")

print("¡Proceso completado!")
