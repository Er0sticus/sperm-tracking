### RED NEURONAL MIXTA UTILIZANDO VGG16 EN LA IDENTIFICACIÓN DE CELULAS GAMETOS MASCULINOS
Este script procesa un video para detectar, rastrear y registrar la posición de objetos en movimiento, utilizando técnicas de visión por computadora con OpenCV. Los resultados se almacenan en un archivo local.

### importar librerias
```python
    import cv2
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance
	```
    

Este bloque importa las librerías necesarias:

**cv2:** Para procesamiento de video e imágenes.
**numpy**: Para cálculos matriciales.
**pandas**: Para organizar y exportar datos a Excel.
**scipy.spatial.distance**: Para calcular distancias entre puntos (centroides).
    
### 2. Configuración de rutas y parámetros iniciales

```python
video_path = 'D:/UDG/Modular/MINIV.mp4'
excel_path = 'D:/UDG/Modular/Base.Datos.xlsx'
data = []
cap = cv2.VideoCapture(video_path)
mog2 = cv2.createBackgroundSubtractorMOG2(history=4000, varThreshold=150, detectShadows=False)
```
Se definen las rutas del video y del archivo Excel donde se guardarán los datos.
MOG2 se utiliza para detectar movimiento al separar el fondo del primer plano.

### 3. Configuración de preprocesamiento
```python
alpha = 0.5
beta = 0
min_contrast = 0
max_contrast = 255
brightness_reduction = 0.1
gamma_value = 0.8
min_area = 100
max_distance = 30
max_age = 5
```
Parámetros para ajustar brillo, contraste y exposición.
min_area: Filtra objetos demasiado pequeños.
max_distance y max_age: Parámetros para rastrear objetos en movimiento.

### 4. Función para corrección de gamma4. Función para corrección de gamma

```python
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
```
Esta función corrige la gamma para ajustar el brillo de una imagen.

### 5. Inicialización de variables de seguimiento
```python
centroid_dict = {}
next_id = 0
frame_num = 0

```
centroid_dict: Almacena los centroides detectados y sus IDs.
next_id: Genera IDs únicos para los objetos.
frame_num: Contador para rastrear el número de cuadros procesados.
### 6. Procesamiento del video cuadro por cuadro
```python
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1
```
El bucle lee y procesa cada cuadro hasta que el video termina.
### 7. Preprocesamiento del cuadro
```python
frame_resized = cv2.resize(frame, (700, 700))
gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
```
Redimensiona el cuadro para estandarizar el tamaño.
Convierte el cuadro a escala de grises.

### 8. Aplicación del filtro Sobel
```python
sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
sobel_gamma_corrected = adjust_gamma(sobel_contrasted, gamma=gamma_value)
```
Calcula gradientes con Sobel para detectar bordes.
Ajusta el brillo y el contraste con la corrección gamma.
### 9. Detección de movimiento con MOG2
```python
fg_mask = mog2.apply(sobel_gamma_corrected)
fg_mask = cv2.GaussianBlur(fg_mask, (7, 7), 0)
_, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

```
Detecta regiones en movimiento mediante segmentación de primer plano.
Suaviza la máscara para mejorar la calidad de los contornos.

### 10. Detección de contornos y cálculo de centroides
```python
contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_area:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        current_centroids.append((cX, cY))
```
Filtra contornos por tamaño mínimo.
Calcula el centroide de cada contorno válido.

### 11. Rastreo de objetos y asignación de IDs

```python
for centroid in current_centroids:
    closest_id = None
    min_dist = max_distance + 1
    for existing_centroid, (obj_id, age) in centroid_dict.items():
        dist = distance.euclidean(centroid, existing_centroid)
        if dist < min_dist and dist <= max_distance:
            min_dist = dist
            closest_id = obj_id
    updated_centroid_dict[centroid] = (next_id, 0) if closest_id is None else (closest_id, 0)
```
Asocia cada centroide actual con el ID más cercano.
Si no encuentra coincidencia, asigna un nuevo ID.

### 12. Visualización y almacenamiento de datos
```python
cv2.putText(frame_resized, f"ID: {obj_id}", (cX - 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
data.append([obj_id, cX, cY, frame_num])
```
Dibuja el ID en cada centroide detectado.
Guarda los datos de cada cuadro en una lista para exportarlos.

### 13. Exportación a Excel
```python
df = pd.DataFrame(data, columns=['ID', 'Centroide_X', 'Centroide_Y', 'Frame_Num'])
df.to_excel(excel_path, index=False)
```
Convierte los datos en un DataFrame de pandas.
Exporta los datos a un archivo Excel.
### 14. Liberación de recursos
```python
cap.release()
cv2.destroyAllWindows()

```

#  Implementacion de la VGG16

### 1. Montar Google Drive
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```
Este bloque conecta Google Drive al entorno de Google Colab, lo que permite acceder a los datos almacenados en él. Debido a que se tubo que hacer una clsificacion manual de que es considerado un esperma util y que no.El argumento force_remount=True asegura que el montaje sea forzado en caso de que ya esté montado.

### 2. Importación de Librerías
```python
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```
Se importan librerías necesarias para el procesamiento de imágenes, creación y entrenamiento del modelo, optimización y evaluación de métricas de clasificación.

### 3. Configuración de Constantes

```python
IMG_SIZE = 224
BATCH_SIZE = 32
TRAIN_DIR = '/content/drive/MyDrive/dataset/train'
VALIDATION_DIR = '/content/drive/MyDrive/dataset/validation'
```
Se definen las constantes:

IMG_SIZE: Tamaño de las imágenes de entrada (224x224 píxeles).
BATCH_SIZE: Número de imágenes procesadas por lote.
TRAIN_DIR y VALIDATION_DIR: Directorios que contienen los conjuntos de datos de entrenamiento y validación.

### 4. Generadores de Imágenes y Aumento de Datos

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
```
train_datagen: Realiza aumentos en las imágenes de entrenamiento, como rotación, desplazamiento, zoom y volteo horizontal, para aumentar la diversidad del conjunto de datos y mejorar la generalización del modelo.
validation_datagen: Solo escala las imágenes del conjunto de validación al rango [0, 1].

### 5. Preparar Generadores
```python
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    seed=42
)
```
flow_from_directory: Carga las imágenes desde los directorios especificados, las ajusta al tamaño requerido y las organiza en lotes.
class_mode='binary': Configura la tarea como una clasificación binaria.

### 6. Construir el Modelo
```python
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=x)
```
VGG16: Se utiliza como base una red VGG16 preentrenada con pesos de ImageNet.
Congelación de capas: Las capas de la base no se entrenan para preservar el conocimiento adquirido.
Capas adicionales:
Flatten: Aplana las características extraídas por VGG16.
Dense: Añade capas densas para realizar la clasificación.

### 7. Compilar el Modelo
```python
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
```
Optimizador: Se usa Adam con una tasa de aprendizaje de 0.001.
Pérdida: Se usa binary_crossentropy para tareas de clasificación binaria.
Métrica: Se evalúa la precisión del modelo.

### 8. Entrenar el Modelo

```python
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)
```
fit: Entrena el modelo usando los datos generados.
steps_per_epoch: Número de lotes procesados en cada época.
epochs=10: Número de iteraciones completas sobre los datos.

### 9. Evaluar el Modelo
```python
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation loss: {loss:.3f}')
print(f'Validation accuracy: {accuracy:.3f}')
```
Evalúa el desempeño del modelo en el conjunto de validación y muestra la pérdida y precisión.
### 10. Realizar Predicciones y Generar Reportes
```python
predictions = model.predict(validation_generator)
predicted_classes = (predictions > 0.5).astype('int32')

print('Classification Report:')
print(classification_report(validation_generator.classes, predicted_classes))
print('Confusion Matrix:')
print(confusion_matrix(validation_generator.classes, predicted_classes))

```
Predicciones: Se generan predicciones sobre el conjunto de validación.
Reportes:
classification_report: Muestra métricas como precisión, sensibilidad y F1-score.
confusion_matrix: Resumen de las predicciones correctas e incorrectas.

![](e)




