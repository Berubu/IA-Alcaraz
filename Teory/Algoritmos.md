# Arboles de decisión #

¿Que es un arbol de decision?
Un árbol de decisión es un modelo de aprendizaje automático que representa decisiones y sus posibles consecuencias en forma de estructura de árbol. Cada nodo interno corresponde a una pregunta o condición sobre los datos, las ramas representan las respuestas posibles y las hojas indican el resultado final o la clasificación. Se utiliza para tomar decisiones, clasificar datos y resolver problemas de predicción.

# Cómo funciona

Los árboles de decisión funcionan siguiendo estos pasos fundamentales:

1. **Selección de características**: 
   - El algoritmo elige la característica más importante para dividir los datos
   - Usa medidas como Ganancia de Información o Índice Gini

2. **División de nodos**:
   - Cada nodo interno representa una prueba sobre una característica
   - Los datos se dividen según las respuestas a estas pruebas
   - Las divisiones buscan crear grupos más "puros" de datos

3. **Criterios de parada**:
   - Profundidad máxima alcanzada
   - Número mínimo de muestras por hoja
   - Pureza del nodo alcanzada

4. **Predicción**:
   - Para clasificación: la clase mayoritaria en la hoja
   - Para regresión: el valor promedio en la hoja

### Ejemplo simple:
```
                ¿Edad > 30?
               /           \
              /             \
           No                Sí
          /                    \
    ¿Salario > 50k?        ¿Tiene casa?
      /          \          /          \
     No          Sí        No          Sí
    [No]        [Sí]      [No]        [Sí]
```

## Índice Gini

El Índice Gini es una medida de impureza utilizada en árboles de decisión que:

1. **Definición**: Mide qué tan frecuentemente un elemento aleatorio sería clasificado incorrectamente.

2. **Fórmula**: Gini = 1 - Σ(pi)²
   - Donde pi es la probabilidad de un elemento pertenecer a la clase i

3. **Características**:
   - Varía entre 0 (pureza total) y 1 (máxima impureza)
   - Un valor de 0 significa que todos los casos pertenecen a una sola clase
   - Un valor de 0.5 indica una distribución igual entre clases

4. **Ejemplo**:
   Para un nodo con:
   - 70% clase A
   - 30% clase B
   Gini = 1 - (0.7² + 0.3²) = 1 - (0.49 + 0.09) = 0.42





   **Datasheet 
   deben tener ejemplos de mas colores pertenecientes al objeto a evaluar **