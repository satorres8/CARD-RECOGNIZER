import os
from PIL import Image, ImageEnhance

def augment_brightness_in_directory(root_dir, brightness_factors):
    """
    Recorre todas las subcarpetas de `root_dir` (train) y para cada imagen:
    - Aplica los factores de brillo indicados (por ejemplo, 0.75 y 1.25).
    - Guarda las nuevas imágenes con un sufijo en el nombre (p.ej. '_bright0.75').
    """
    for subdir, dirs, files in os.walk(root_dir):
        for filename in files:
            # Ignora archivos que no sean imágenes (p.ej. .txt, .py, etc.)
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            filepath = os.path.join(subdir, filename)
            try:
                with Image.open(filepath) as img:
                    for factor in brightness_factors:
                        enhancer = ImageEnhance.Brightness(img)
                        img_enhanced = enhancer.enhance(factor)
                        
                        base_name, ext = os.path.splitext(filename)
                        new_filename = f"{base_name}_bright{factor}{ext}"
                        new_filepath = os.path.join(subdir, new_filename)
                        
                        img_enhanced.save(new_filepath)
                        print(f"Guardada imagen con factor {factor}: {new_filepath}")
            except Exception as e:
                print(f"Error al procesar {filepath}: {e}")

def main():
    # Directorio donde se encuentran las imágenes de entrenamiento
    train_dir = 'data/train'
    
    # Factores de brillo (0.75 = -25%, 1.25 = +25%)
    brightness_factors = [0.75, 1.25]
    
    # Aplica Data Augmentation SOLO en el directorio de train
    print("Aplicando brillo en carpeta 'train'...")
    augment_brightness_in_directory(train_dir, brightness_factors)

if __name__ == "__main__":
    main()
