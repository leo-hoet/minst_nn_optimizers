import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from model import NNModel
import tensorflow as tf
from typing import List
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Función para crear la población inicial, es decir, correr la red neuronal, obtener los pesos (genes) y el accuracy (aptitud). Toma como arg el tamaño de la población y el modelo, retorna la población inicial (df)


def firstPop(popSize, model, data):

    initInds = []  # Lista para almacenar los individuos iniciales
    initFit = []  # Lista para almacenar las aptitudes iniciales
    x_test, y_test = data

    for i in range(popSize):
        ind = model.get_weights_as_numpy()
        # fit = model.metrics(x_test, y_test).get("accuracy")
        fit = model.metrics(x_test, y_test).get("f1_score")
        initInds.append(ind)
        initFit.append(fit)
        # Concatenando individuos y aptitudes para formar la matriz de población inicial
        popInit = pd.DataFrame(list(zip(initInds, initFit)),
                               columns=['individuos', 'aptitudes'])

    return popInit

# Ahora que se tiene la población inicial y las aptitudes iniciales, hay que 1. elegir los mejores para ser padres, 2. cruzarlos y hacer hijos, 3. mutar los hijos

# Función para ordenar la población. Toma como arg la población (df), retorna la población ordenada (df)


def sortByFit(pop):
    popSorted = pop.sort_values(by=['aptitudes'], ascending=False)
    return popSorted

# Para la elegir a los mejores, hay que hacer "sort" de la población inicial (por aptitud) y tomar la mitad

# Función para retornar los padres. Por conveniencia se toma la mitad de la población con mejor fitness como padres. Toma como arg la población ordenada (df) y retorna los padres (df)


def selecParents(popSize, popSorted):
    parents = popSorted.head(int(popSize/2))
    return parents

# Para cruzarlos, hay que hacer alelos con un corte, tomar la mitad de un individuo (padre1) y la mitad de otro individuo(padre2)

# Función para hacer un individuo child a partir de parent1 y parent2. Toma como arg los parent1 y parent2 (listas) y retorna un hijo (list)


def breed(parent1, parent2):  # tomar como punto de corte la mitad de los padres (listas)
    tam = len(parent1[0])
    k = int(tam/2)
    child_w = list(parent1[0][:k]) + list(parent2[0][k:])  # Hijo (pesos)
    child_f = 0  # Hijo, aptitud
    child = [child_w, child_f]  # Hijo completo
    return child  # retorna un nuevo individuo hijo

# Para mutarlos, se puede hacer un plot de los pesos de un individuo con matplotlib para saber cómo se comportan los 109386 pesos y ver si se puede hacer algo arbitrario (como aumentar o disminuir x% ciertos valores)


# plt.plot(parent1[0][:100])
# plt.ylabel('some numbers')
# plt.show()


# Aparentemente son muy "variables" entre 0 y 1 y no hay un patrón que permita establecer un rango para hacer la mutación. Por eso se hará cambiando los valores de algunos pesos (aleatoriamente seleccionados) por otros nuevos con valores aleatorios

# Función para mutar un individuo. Toma como arg un individuo (lista) y retorna el nuevo mutado (lista)
def mutate(individual):

    randArr = np.random.rand(len(individual[0]))  # Arreglo aleatorio del tamaño de la lista de pesos del individuo

    booleanArr = list(np.round(randArr).astype(int))  # Convertir a booleanos
    mutated = []

    nums = range(0, len(booleanArr))  # Lista con los números de 0 al tamaño de la cantidad de genes de cada individuo
    for i in booleanArr:  # loop para agregar los randomicos (que mutan al individuo)
        if i == 1:
            mutated.append(random.random())
        else:
            mutated.append(1)

    for j in nums:  # loop que cambia los "1" que quedaron en el anterior loop por  el resto de valores de genes originales del individuo
        if mutated[j] == 1:
            mutated[j] = individual[0][j]

    mutated_w = mutated  # Mutado (pesos)
    mutated_f = 0  # Mutado, aptitud
    mutatedArr = [mutated_w, mutated_f]  # mutado completo

    return mutatedArr

# Función para hacer los cruces de toda la población, toma como arg los padres (df) y retorna los hijos (df) con aptitud 0 por ahora


def breedPop(parents):
    myParents = parents.copy()  # Crear una copia del df para desordenarlo
    shuffleParents = myParents.sample(n=len(myParents))
    shuffleParents = shuffleParents.reset_index(drop=True)  # desordenarlo
    # Crear el df que contendrá los hijos
    children = pd.DataFrame([], columns=['individuos', 'aptitudes'])

    for i in range(0, len(parents), 2):  # Cruzar los padres continuos para crear los hijos y añadirlos al df children
        if i < (len(parents)-1):
            newChild = breed(shuffleParents.iloc[i], shuffleParents.iloc[i+1])
            newChild_df = {'individuos': newChild[0], 'aptitudes': newChild[1]}
            children = children._append(newChild_df, ignore_index=True)

    return children

# Función para mutar la población. Toma como argumentos padres e hijos, los junta en un solo df, selecciona de forma aleatoria 1/4 de esos individuos y les aplica la función de mutación. Luego junta los nuevos individuos mutados con el resto de la población y retorna esa nueva población. Tiene un tercer argumento que es el tamaño de la población para llenar en caso de que haga falta, con individuos mutados


def mutatePop(parents, children, popSize):
    finalParents = parents.copy()
    finalChildren = children.copy()
    myPop = pd.concat([finalParents, finalChildren])  # Concatenar padres e hijos
    myPop = myPop.reset_index(drop=True)  # desordenarlo
    tam = len(myPop.index)

    # mutatedInds = pd.DataFrame({'individuos': pd.Series(dtype='float32'),
    #                'aptitudes': pd.Series(dtype='float32')})
    mutatedInds = pd.DataFrame([], columns=['individuos', 'aptitudes'])

    # Ahora que están desordenados, aplicar la mutación 1/4 de los individuos e irlos añadiendo a la misma población
    for i in range(0, (popSize - tam)):
        newMutated = mutate(list(myPop.iloc[i]))
        newMutated_df = {'individuos': newMutated[0], 'aptitudes': newMutated[1]}
        mutatedInds = mutatedInds._append(newMutated_df, ignore_index=True)

    mutatedPop = pd.concat([myPop, mutatedInds])  # Concatenar padres e hijos

    return mutatedPop

# Función para dejar lista la población de la siguiente generación. Calcula el fitness de los individuos, testeando el modelo con sus pesos. Toma como arg el tamaño de la población, el modelo y la población actual mutada. Retorna la nueva población


def fitnessNextGeneration(popSize, model, mutatedPop, data):

    x_test, y_test = data
    Inds = []  # Lista para almacenar los individuos de la siguiente generación
    Fits = []  # Lista para almacenar las aptitudes de la siguiente generación

    for i in range(0, popSize):
        t0 = time.perf_counter()
        # model.randomize_weights()
        myNewPop_np = np.array(mutatedPop.iloc[i][0])
        model.set_custom_weights(myNewPop_np)
        metrics = model.metrics(x_test, y_test)
        t1 = time.perf_counter()
        ind = model.get_weights_as_numpy()
        # fit = model.metrics(x_test, y_test).get("accuracy")
        fit = model.metrics(x_test, y_test).get("f1_score")
        Inds.append(ind)
        Fits.append(fit)
        # Concatenando individuos y aptitudes para formar la matriz de población inicial
        newPop = pd.DataFrame(list(zip(Inds, Fits)),
                              columns=['individuos', 'aptitudes'])

    return newPop

# Función para optimizar el modelo. Utiliza las funciones anteriores y las itera cierto número de generaciónes. Recibe como argumento todo lo que necesitan las anteriores funciones para operar. Retorna el individuo más óptimo de la última generación y la lista de más óptimos por cada generación


def optimizeModel(generations, popSize, model, data):

    popInit = firstPop(popSize, model, data)  # Creando la población inicial
    savedFit = []  # Guardar la aptitud del mejor indivuo por generación

    # Una vez se tienen los individuos de la primera poplación, se puede seguir iterando por la cantidad de generaciones necesarias

    newPopulation = pd.DataFrame([], columns=['individuos', 'aptitudes'])

    for i in range(0, generations):
        if i == 0:
            popSorted = sortByFit(popInit)  # Ordenando por fitness
        else:
            popSorted = sortByFit(newPopulation)  # Ordenando por fitness
        parents = selecParents(popSize, popSorted)  # Seleccionando los padres
        children = breedPop(parents)  # Cruzando los padres de la población
        mutatedPopulation = mutatePop(parents, children, popSize)  # Mutando la población
        # Nueva población para la siguiente generación
        newPopulation = fitnessNextGeneration(popSize, model, mutatedPopulation, data)
        savedFit.append(popSorted.iloc[0][1])

    finalPopSorted = sortByFit(newPopulation)

    return savedFit, finalPopSorted


def run_ga(generations=10, pop_size=80, sample_size=2000) -> List[float]:
    model = NNModel()

    data = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), _ = data
    x_train = x_train.reshape(x_train.shape[0], 784)[:sample_size]
    y_train = y_train[:sample_size]

    aptOptimos, pop = optimizeModel(generations, pop_size, model, (x_train, y_train))
    return aptOptimos, pop


if __name__ == '__main__':
    fitnesses, _ = run_ga(generations=10, pop_size=10, sample_size=600)
    print("Best fitnesses for each generation ", fitnesses)
