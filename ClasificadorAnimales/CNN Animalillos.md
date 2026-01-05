# Red Neuronal Convolucional (CNN) para Clasificación de Animales por Imágenes

## Descripción General

En este proyecto se entrena una red neuronal para reconocer y clasificar imágenes según categorías. En este caso, el proyecto está configurado para reconocer diferentes tipos de animales en fotos.

---

## 1. Importar Librerías Necesarias

```python
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
```

**¿Qué es cada librería?**

- **NumPy**: Sirve para trabajar con números y matrices
- **os y re**: Para buscar y leer archivos del disco
- **matplotlib**: Para mostrar gráficos e imágenes
- **sklearn**: Para dividir datos y medir resultados
- **TensorFlow/Keras**: La librería principal para crear redes neuronales

---

## 2. Cargar Imágenes del Disco

```python
from skimage.transform import resize
dirname = os.path.join(os.getcwd(),'dataset')
imgpath = dirname + os.sep 

images = []
directories = []
dircount = []
```

**¿Qué hace este código?**

1. Busca en una carpeta llamada "dataset"
2. Lee todas las imágenes que encuentre (jpg, png, bmp, etc.)
3. Redimensiona cada imagen a 64x64 píxeles (para que todas tengan el mismo tamaño)
4. Convierte imágenes de 4 canales (RGBA) a 3 canales (RGB)
5. Convierte imágenes en blanco y negro a color
6. Guarda todas las imágenes en una lista

**Importante:** Las imágenes deben estar organizadas en subcarpetas, donde cada subcarpeta tiene imágenes de una categoría diferente.

---

## 3. Crear Etiquetas

```python
labels=[]
indice=0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice=indice+1
```

**¿Qué hace?**

Crea etiquetas numéricas para cada imagen:
- Imágenes en carpeta 1 → etiqueta 0
- Imágenes en carpeta 2 → etiqueta 1
- Imágenes en carpeta 3 → etiqueta 2
- Y así sucesivamente...

**Nombres de categorías:**

```python
for directorio in directories:
    name = directorio.split(os.sep)
    deportes.append(name[len(name)-1])
```

Guarda los nombres de las carpetas como nombres de categorías.

---

## 4. Preparar los Datos

### 4.1 Convertir a Arrays de NumPy

```python
y = np.array(labels)
X = np.array(images, dtype=np.uint8)
```

Convierte las listas a matrices NumPy para que TensorFlow pueda usarlas.

### 4.2 Dividir en Conjuntos de Entrenamiento y Prueba

```python
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2)
```

**¿Qué significa?**

- `train_X`: 80% de las imágenes para entrenar
- `test_X`: 20% de las imágenes para probar
- `train_Y`: Etiquetas de entrenamiento
- `test_Y`: Etiquetas de prueba

### 4.3 Preprocesar Imágenes

```python
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = preprocess_input(train_X)
test_X = preprocess_input(test_X)
```

**¿Por qué hacemos esto?**

- Convierte los píxeles de números entre 0-255 a números entre -1 y 1
- Esto ayuda a la red neuronal a aprender mejor
- Es una normalización estándar en redes de imágenes

### 4.4 Convertir Etiquetas a "One-Hot Encoding"

```python
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)
```

**Ejemplo:**
- Antes: `3` (significa categoría 3)
- Después: `[0, 0, 0, 1, 0]` (1 en la posición 3)

La red necesita este formato para entender mejor.

### 4.5 Dividir Entrenamiento en Validación

```python
train_X, valid_X, train_label, valid_label = train_test_split(
    train_X, train_Y_one_hot, test_size=0.2, random_state=13
)
```

Divide el conjunto de entrenamiento:
- 80% para entrenar la red
- 20% para validar mientras entrena

---

## 5. Crear la Red Neuronal (CNN)

```python
sport_model = Sequential()

# Bloque 1 (32 filtros) 
sport_model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)))
sport_model.add(BatchNormalization())
sport_model.add(MaxPooling2D(pool_size=(2, 2)))
sport_model.add(Dropout(0.25))
```

**¿Qué es cada parte?**

1. **Conv2D (Convolución)**: Busca patrones en las imágenes
   - Primer número (32): Cantidad de filtros para detectar patrones
   - (3, 3): Tamaño de la ventana

2. **BatchNormalization**: Normaliza los datos entre capas para entrenar mejor

3. **MaxPooling2D**: Reduce el tamaño de las imágenes sin perder información importante
   - Pool de 2x2: Divide el tamaño por 2

4. **Dropout(0.25)**: Apaga el 25% de las neuronas aleatoriamente para evitar memorizar datos

**La red tiene 4 bloques:**
- Bloque 1: 32 filtros
- Bloque 2: 64 filtros
- Bloque 3: 128 filtros
- Bloque 4: 256 filtros

**Capa final (Clasificación):**

```python
sport_model.add(Flatten())
sport_model.add(Dense(256, activation='relu'))
sport_model.add(Dropout(0.5))
sport_model.add(Dense(nClasses, activation='softmax'))
```

- **Flatten**: Convierte la matriz 2D a una línea
- **Dense(256)**: Capa completamente conectada con 256 neuronas
- **Dense(nClasses)**: Capa final con número de categorías
- **softmax**: Convierte salidas a probabilidades (0-1)

---

## 6. Compilar la Red

```python
sport_model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR),
    metrics=['accuracy']
)
```

**¿Qué significa?**

- **loss**: Función para medir qué tan mal predice la red
- **optimizer**: Algoritmo que mejora la red (Adam con velocidad de aprendizaje = 0.001)
- **metrics**: Medida a mostrar (precisión = porcentaje de aciertos)

---

## 7. Entrenar la Red

```python
INIT_LR = 1e-3  # Velocidad de aprendizaje (0.001)
epochs = 50     # Entrenar 50 veces con todos los datos
batch_size = 32 # Procesar 32 imágenes a la vez

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

sport_train = sport_model.fit(
    datagen.flow(train_X, train_label, batch_size=batch_size),
    steps_per_epoch=len(train_X) // batch_size,
    epochs=epochs,
    validation_data=(valid_X, valid_label),
    callbacks=[reduce_lr]
)
```

**¿Qué es esto?**

1. **epochs**: La red ve todos los datos 50 veces
2. **batch_size**: Procesa 32 imágenes juntas para calcular cambios en la red
3. **ImageDataGenerator**: Aumenta datos rotando, moviendo y volteando imágenes
4. **validation_data**: Valida con datos no vistos durante el entrenamiento
5. **reduce_lr**: Si se estanca, reduce la velocidad de aprendizaje

---

## 8. Guardar la Red

```python
sport_model.save("animales_2.h5")
```

Guarda la red entrenada para usarla después sin volver a entrenar.

---

## 9. Evaluar la Red

```python
test_eval = sport_model.evaluate(test_X, test_Y_one_hot, verbose=1)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
```

Prueba la red con imágenes que nunca vio:
- **Loss**: Qué tan segura está la red (menor es mejor)
- **Accuracy**: Porcentaje de predicciones correctas

---

## 10. Visualizar Resultados

### Gráficos de Entrenamiento

```python
accuracy = sport_train.history['accuracy']
val_accuracy = sport_train.history['val_accuracy']
loss = sport_train.history['loss']
val_loss = sport_train.history['val_loss']

plt.plot(epochs, accuracy, label='Training accuracy')
plt.plot(epochs, val_accuracy, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()
```

Muestra gráficos de cómo mejoró la red durante el entrenamiento.

### Predicciones Correctas

```python
predicted_classes = sport_model.predict(test_X)
correct = np.where(predicted_classes==test_Y)[0]

for i, correct in enumerate(correct[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[correct], cmap='gray')
    plt.title("{} (Correcto)".format(deportes[predicted_classes[correct]]))
```

Muestra las primeras 9 imágenes que la red clasificó correctamente.

### Predicciones Incorrectas

```python
incorrect = np.where(predicted_classes!=test_Y)[0]

for i, incorrect in enumerate(incorrect[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect], cmap='gray')
    plt.title("Predijo: {}, Real: {}".format(
        deportes[predicted_classes[incorrect]],
        deportes[test_Y[incorrect]]
    ))
```

Muestra las primeras 9 imágenes que clasificó mal (para aprender dónde falla).

### Reporte de Resultados

```python
print(classification_report(test_Y, predicted_classes, target_names=deportes))
```

Muestra un reporte detallado de precisión para cada categoría.

---

## 11. Probar con Imágenes Nuevas

```python
filenames = ['tortuga 16.jpg', 'tortuga 17.jpg']

for filepath in filenames:
    image = plt.imread(filepath)
    image_resized = resize(image, (64, 64))
    X = np.array([image_resized], dtype=np.uint8)
    X = X.astype('float32')
    X = preprocess_input(X)
    
    prediction = sport_model.predict(X)
    print(filepath, deportes[prediction.tolist().index(max(prediction))])
```

Carga imágenes nuevas, las preprocesa igual que las de entrenamiento, y predice la categoría.

---

## Flujo Completo Resumido

1. **Cargar imágenes** → redimensionar a 64x64
2. **Crear etiquetas** → asignar un número a cada categoría
3. **Dividir datos** → 80% entrenamiento, 20% prueba
4. **Preprocesar** → normalizar píxeles de 0-255 a -1 a 1
5. **Convertir etiquetas** → formato one-hot encoding
6. **Crear red neuronal** → 4 bloques convolucionales + capas finales
7. **Compilar red** → configurar función de pérdida y optimizador
8. **Entrenar red** → 50 épocas con validación
9. **Guardar modelo** → para uso futuro
10. **Evaluar** → medir precisión en datos de prueba
11. **Visualizar** → gráficos y ejemplos
12. **Probar con nuevas imágenes** → usar el modelo entrenado

---

## Notas Importantes

- **Calidad de datos**: Imágenes claras y bien etiquetadas = mejor red
- **Cantidad de datos**: Más imágenes por categoría = mejor entrenamiento
- **Epochs**: 50 es un buen número, si pierde precisión aumenta
- **Batch size**: 32 es estándar, más bajo = más precisión pero más lento
- **Dropout**: Previene memorizar (sobreajuste), importante en redes profundas
- **Augmentation**: Girar y mover imágenes aumenta la cantidad de datos
- **Validación**: Es importante validar mientras entrena para detectar problemas
