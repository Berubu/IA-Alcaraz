import cv2 as cv
import numpy as np

img = cv.imread('C:\\IA-Alcaraz\\Imagen\\figura.png', 1)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Definir rangos de color en HSV
# Rojo (dos rangos)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Verde
lower_green = np.array([40, 100, 100])
upper_green = np.array([80, 255, 255])

# Azul
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])

# Amarillo
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

# Variable para seguir el color actual
current_color = 0
color_names = ['Rojo', 'Verde', 'Azul', 'Amarillo']

print("Presiona 'n' para cambiar al siguiente color")
print("Presiona 'q' para salir")

while True:
    # Crear máscara según el color actual
    if current_color == 0:  # Rojo
        mask1 = cv.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
    elif current_color == 1:  # Verde
        mask = cv.inRange(hsv, lower_green, upper_green)
    elif current_color == 2:  # Azul
        mask = cv.inRange(hsv, lower_blue, upper_blue)
    else:  # Amarillo
        mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    # Aplicar máscara
    result = cv.bitwise_and(img, img, mask=mask)

    # Mostrar imágenes
    cv.imshow('Imagen Original', img)
    cv.imshow(f'Detección de {color_names[current_color]}', result)
    cv.imshow(f'Máscara de {color_names[current_color]}', mask)

    # Esperar tecla
    key = cv.waitKey(1) & 0xFF
    
    # 'n' para siguiente color, 'q' para salir
    if key == ord('n'):
        cv.destroyWindow(f'Detección de {color_names[current_color]}')
        cv.destroyWindow(f'Máscara de {color_names[current_color]}')
        current_color = (current_color + 1) % 4
        print(f"Mostrando color: {color_names[current_color]}")
    elif key == ord('q'):
        break

cv.destroyAllWindows()