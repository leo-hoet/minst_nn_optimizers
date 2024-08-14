# MNIST Neural network optimizers

Este repositorio contiene la implementacion de una red neuronal
clasificadora del dataset MSNIST.

Se impemento el entrenamiento de la red utilizando PSO, un algoritmo genetico y Backpropagation.

El analisis de cada metodo para entrenar la red se puede ver en el archivo `analysis.ipynb`.


Las caracteristicas de la red a entrenar:
- Una capa de 784 neuronas de entrada
- Una capa fully connected de 128 neuronas con activacion relu
- Una capa fully connected de 64 neuronas con activacion relu
- Una capa de salida de 10 neuronas con activacion softmax

Tanto para PSO como para el algoritmo genetico se busca maximizar el `F1 score`.

#### Limitaciones y bugs

Encontramos que PSO tiene una mejor performance que GA. Las performace en general de las dos heuristicas no
supera un f1=0.2

Los script no posee la funcionalidad de captar comandos desde stdin, por lo que para modificar parametros
se debe hacer en el codigo fuente


## Instalacion


### Localmente
Requerimientos:
- python version >= 3.8
- pip

Inslatar las dependencias con: 
```bash
pip3 install -r requirements.txt
```

### Docker
Se puede instalar dentro de un [devcontainer](https://containers.dev/) especificado en el archivo `.devcontainer/devcontainer.json`


## Uso

### PSO

```bash
python3 main.py
```
Esto genera una salida como la siguiente:

```text
Iter 0 best fitness: 0.06971582693011343. Global best 0.06971582693011343
Iter 1 best fitness: 0.11955340748375988. Global best 0.11955340748375988
Iter 2 best fitness: 0.13728580116172837. Global best 0.13728580116172837
Iter 3 best fitness: 0.1371721612862173. Global best 0.13728580116172837
Iter 4 best fitness: 0.16144216222296612. Global best 0.16144216222296612
Iter 5 best fitness: 0.16154277058247113. Global best 0.16154277058247113
Iter 6 best fitness: 0.16322514027426566. Global best 0.16322514027426566
Iter 7 best fitness: 0.1689193325554236. Global best 0.1689193325554236
Iter 8 best fitness: 0.15849462577187212. Global best 0.1689193325554236
Iter 9 best fitness: 0.15916692239192232. Global best 0.1689193325554236

           0       0.28      0.35      0.31       980
           1       0.27      0.21      0.24      1135
           2       0.02      0.00      0.00      1032
           3       0.01      0.00      0.00      1010
           4       0.20      0.38      0.26       982
           5       0.10      0.05      0.06       892
           6       0.25      0.21      0.23       958
           7       0.19      0.07      0.10      1028
           8       0.16      0.63      0.25       974
           9       0.01      0.00      0.00      1009

    accuracy                           0.19     10000
   macro avg       0.15      0.19      0.15     10000
weighted avg       0.15      0.19      0.15     10000
```

### GA
```bash
python3 ga.py
```

Salida:

```text
Best fitnesses for each generation  [0.019355488418932528, 0.019355488418932528, 0.019355488418932528, 0.020562248995983936, 0.020562248995983936, 0.021177944862155386, 0.021177944862155386, 0.021177944862155386, 0.021177944862155386, 0.021177944862155386
```

El codigo se puede encontrar en https://github.com/leo-hoet/minst_nn_optimizers

Realizado por Leonardo Hoet y Brayan Segura
