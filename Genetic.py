"""
The genetic algorithm lives here
"""
import random

import keras
import numpy as np

from TheNetwork import NeuralNetwork
from WorkinWithData import MessingWithData


class GeneticAlgorithm:
    def __init__(self):
        self.fitness = []
        self.population = []

    def mutate(self, individual):
        """
        One random element will be chosen for mutation from a candidate gene to further diversify the pool
        :param individual: candidate gene
        :return: mutated diverse candidate
        """
        mutation_pt = random.randint(0, len(individual))
        if individual[mutation_pt] == 0:
            individual[mutation_pt] = 1
        else:
            individual[mutation_pt] = 0
        return individual

    def crossover(self, individual_a, individual_b):
        """
        Does crossover from a random position
        :param individual_a: Candidate 1 to be crossedover
        :param individual_b: with candidate 2
        :return: 2 offsprings
        """
        crossover_pt = random.randint(0, len(individual_a))
        off_a = individual_a[0:crossover_pt] + individual_b[crossover_pt: len(individual_b)]
        off_b = individual_b[0:crossover_pt] + individual_a[crossover_pt:len(individual_a)]

        return off_a, off_b

    def calculate_fitness(self, population):
        for pop in population:
            drop_list = [x for x in pop if x == 0]
            X_train = np.delete(ga.X_train, drop_list, axis=1)
            print(X_train.shape, " After deletion")
            X_test = np.delete(ga.X_test, drop_list, axis=1)

            nn = NeuralNetwork()
            print("Defining Model")
            model = nn.define_mode(X_train.shape[1])
            print("Launching the NN now")
            acc = nn.train(model, X_train, X_test, ga.y_train, ga.y_test)
            print(acc)
            keras.backend.clear_session()


if __name__ == "__main__":
    ga = GeneticAlgorithm()
    print("Reading file")
    md = MessingWithData('/Users/k.n./Downloads/microsoft-malware-prediction', 'train.csv')
    print("Pre-Processing")
    ga.X_train, ga.X_test, ga.y_train, ga.y_test, cols_list = md.read_file()
    print("Defining Model")

    # initial population size
    population_diversity_size = 8
    pop_shape = (population_diversity_size, len(cols_list) - 1)

    # Creating the initial population.
    new_population = np.random.randint(low=0, high=2, size=pop_shape)

    best_outputs = []
    num_generations = 100

    for generation in range(num_generations):
        print("Generation : ", generation)
        ga.calculate_fitness(new_population)
