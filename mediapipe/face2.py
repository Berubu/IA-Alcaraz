import cv2
from deepface import DeepFace

# Captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Espejo para mayor naturalidad

    try:
        # Usamos DeepFace para analizar el frame
        # 1. 'actions' le dice que solo busque emociones.
        # 2. 'enforce_detection=False' evita que el programa se cierre si no detecta una cara.
        results = DeepFace.analyze(frame, 
                                   actions=['emotion'], 
                                   enforce_detection=False,
                                   detector_backend='opencv') # Usamos un detector rápido

        # DeepFace puede detectar múltiples caras, 'results' es una lista
        for result in results:
            # Extraer el rectángulo (bounding box) de la cara
            x = result['region']['x']
            y = result['region']['y']
            w = result['region']['w']
            h = result['region']['h']

            # Extraer la emoción dominante (la que tuvo mayor puntaje)
            # DeepFace traduce las emociones al español si el sistema está en español.
            emocion = result['dominant_emotion']
            
            # Dibujar el rectángulo
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Escribir la emoción detectada
            cv2.putText(frame, 
                        emocion, 
                        (x, y - 10), # Posición (un poco arriba de la caja)
                        cv2.FONT_HERSHEY_SIMPLEX, # Tipo de letra
                        0.9, # Tamaño de letra
                        (0, 255, 0), # Color (verde)
                        2) # Grosor

    except Exception as e:
        # A veces, si no hay cara, puede fallar; 'pass' lo ignora y sigue.
        pass

    # Mostrar la imagen
    cv2.imshow('Detector de Emociones (con DeepFace)', frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()