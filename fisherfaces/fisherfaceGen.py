import cv2 as cv 
import numpy as np 
import os

dataSet = 'C:/IA-Alcaraz/fisherfaces/fotos'
faces  = os.listdir(dataSet)
print(faces)

labels = []
facesData = []
label = 0 
for face in faces:
    facePath = os.path.join(dataSet, face)
    for faceName in os.listdir(facePath):
        # 1. Comprobamos si el archivo es una imagen (puedes añadir más extensiones)
        if faceName.endswith('.jpg') or faceName.endswith('.png'):
            labels.append(label)
            
            # Creamos la ruta completa de la imagen
            image_path = os.path.join(facePath, faceName)
            
            # Leemos la imagen
            image = cv.imread(image_path, 0)
            
            # 2. Verificamos que la imagen se haya cargado correctamente
            if image is not None:
                facesData.append(image)
            else:
                print(f"No se pudo leer la imagen: {image_path}")

    label = label + 1
#print(np.count_nonzero(np.array(labels)==0)) 
faceRecognizer = cv.face.FisherFaceRecognizer_create()
faceRecognizer.train(facesData, np.array(labels))
faceRecognizer.write('FisherFace.xml')
