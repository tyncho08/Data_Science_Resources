# Reporte Ejercicio 1

## Modelos utilizados
1. __Modelo 1:__
1 capa oculta de 540 neuronas, con función de activación __ReLU__ y regularización __L2__, más una capa de Dropout y la Salida compuesta por una capa de 2 neuronas con activación __sigmoid__.
2. __Modelo 2:__
1 capa oculta de 540 neuronas, con función de activación __ReLU__ y regularización __L2__, más una capa de Dropout, luego una capa de 520 neuronas con activación __ReLU__ y regularización __L2__, seguidamente otra capa de Dropout y finalmente la Salida compuesta por una capa de 2 neuronas con activación __sigmoid__.
3. __Modelo 3:__ 
1 capa oculta de 540 neuronas, con función de activación __ReLU__ y regularización __L2__, más una capa de Dropout, luego una capa de 420 neuronas con activación __ReLU__ y regularización __L2__, seguidamente otra capa de Dropout y luego una capa con 380 neuronas con activación __ReLU__ y regularización __L2__, finalmente una Salida compuesta por una capa de 2 neuronas con activación __sigmoid__.

## Procesado del Dataset mediante una representación TfIDf:
Parámetros del TfIDf:
1. __Bynary:__ True
2. __Ngram_range:__ (1,2)
3. __Stop_words:__ English
4. __Max_df:__ 1.0
5. __Norm:__ L2
2. __Vocabulary:__ None

## Decisiones de los Modelos
1. __Modelo 1:__ Se busco generar un modelo simple, con poco procesamiento para poder tomarlo como referencia para los otros dos modelos (Baseline).
   * __Capas:__ 1 de entrada, 2 ocultas (540 y dropout) y 1 de salida (2).
   * __Activación:__ ReLU y salida sigmoid.
   * __Regularización:__ L2.
   * __Dropout:__ Si.
   * __Función de costo:__ binary_crossentropy.
   * __Optimizador:__ Adam.
   * __Metrica:__ Accuracy.
2. __Modelo 2:__ Se generó un modelo de media complejidad, con una mayor cantidad de capas ocultas..
   * __Capas:__ 1 de entrada, 4 ocultas (540, 520 y 2 dropout) y 1 de salida (2).
   * __Activación:__ ReLU y salida sigmoid.
   * __Regularización:__ L2.
   * __Dropout:__ Si.
   * __Función de costo:__ binary_crossentropy.
   * __Optimizador:__ Adam.
   * __Metrica:__ Accuracy.
3. __Modelo 3:__ El modelo más complejo, se decidió utilizar una mayor cantidad de capas ocultas, junto con más capas de Dropout.
   * __Capas:__ 1 de entrada, 6 ocultas (540, 420, 380 y 3 dropout) y 1 de salida (2).
   * __Activación:__ ReLU y salida sigmoid.
   * __Regularización:__ L2.
   * __Dropout:__ Si.
   * __Función de costo:__ binary_crossentropy.
   * __Optimizador:__ Adam.
   * __Metrica:__ Accuracy.

## Proceso de Entrenamiento
1. __División de Train y Test:__ 75% para Entrenamiento y 25% para Test.
2. __Tamaño de Batch:__ 80.
3. __Número de Épocas:__ 50.
4. __Métricas de evaluación:__ Accuracy.

## Overfitting
Se realizaron dos gráficas: __Accuracy vs Epoch__ y __Loss vs. Epoch__, para poder observar de mejor manera la evolución del aprendizaje de los modelos propuestos.

## Archivos Generados
1. __Modelos:__ Figuras de los modelos y los modelos en sí (Carpeta __Modelos__).
2. __Precisión y Otros:__ Reporte de los parámetros principales usados y precisiones logradas (Carpeta __Precision__).
3. __Resultados:__ Predicciones realizadas y gráficas de __Accuray vs. Epoch__ y __Loss vs. Epoch__ (Carpeta __Resultados__)
