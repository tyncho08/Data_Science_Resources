# Reporte Ejercicio 1

## Modelos utilizados
1. __Modelo:__
Se probaron 200 modelos en total, sin embargo la mayoría daban resultados muy parecidos, por ello se optó sólo por poner el modelo con mejores resultados, el mismo está conformado por:
La capa de Embedding, la capa de Lambda para tomar el promedio, una capa oculta de 300 neuronas, con activavión __ReLU__ y regularización __L2__, una capa de Dropout,una capa oculta de 300 neuronas, con activavión __ReLU__ y regularización __L2__, una capa de Dropout, una capa oculta de 250 neuronas, con activavión __ReLU__ y regularización __L2__, una capa de Dropout, una capa oculta de 120 neuronas, con activavión __ReLU__ y regularización __L2__, una capa de Dropout, una capa oculta de 80 neuronas, con activavión __ReLU__ y regularización __L2__, una capa de Dropout, una capa oculta de 40 neuronas, con activavión __ReLU__ y regularización __L2__ y finalmente una capa de Salida de 2 neuronas y activación sigmoid.

## Decisiones del Modelo
1. __Modelo:__ Los modelos probados fueron muy variados desde muy básicos hasta muy complejos, sin embargo los resultados fueron muy pobres lo que nos hace sospechar que el embedding no es la mejor opción para el dataset provisto, otro problema puede ser el tamaño del conjunto de datos, el cual es muy chico.
   * __Capas:__ Embedding, Lambda, 11 ocultas (300, 300, 250, 120, 80, 40 y 5 de dropout) y 1 de salida (2).
   * __Activación:__ ReLU y salida sigmoid.
   * __Regularización:__ L2.
   * __Dropout:__ Si.
   * __Función de costo:__ binary_crossentropy.
   * __Optimizador:__ Adam.
   * __Metrica:__ Accuracy.

## Proceso de Entrenamiento
1. __División de Train y Test:__ 75% para Entrenamiento y 25% para Test.
2. __Número de Épocas:__ 300.
3. __Métricas de evaluación:__ Accuracy.

## Overfitting
Se realizaron dos gráficas: __Accuracy vs Epoch__ y __Loss vs. Epoch__, para poder observar de mejor manera la evolución del aprendizaje del modelo propuesto.

## Archivos Generados
1. __Modelos:__ Figuras de los modelos y los modelos en sí (Carpeta __Modelos__).
2. __Precisión y Otros:__ Reporte de los parámetros principales usados y precisiones logradas (Carpeta __Precision__).
3. __Resultados:__ Predicciones realizadas y gráficas de __Accuray vs. Epoch__ y __Loss vs. Epoch__ (Carpeta __Resultados__)
