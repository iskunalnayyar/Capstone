"""
The genetic algorithm lives here
"""

import keras
import numpy as np
import pandas as pd

from TheNetwork import NeuralNetwork
from WorkinWithData import MessingWithData


class GeneticAlgorithm:
    def __init__(self):
        self.population = []

    def mutate(self, individual):
        """
        One random element will be chosen for mutation from a candidate gene to further diversify the pool
        :param individual: candidate gene
        :return: mutated diverse candidate
        """
        mutation_idx = np.random.randint(low=0, high=individual.shape[1], size=4)
        # Mutation changes a single gene in each offspring randomly.
        for idx in range(individual.shape[0]):
            # The random value to be added to the gene.
            individual[idx, mutation_idx] = 1 - individual[idx, mutation_idx]
        return individual

    def crossover(self, parents, offspring_size):
        """
        Does crossover from a random position
        :param individual_a: Candidate 1 to be crossedover
        :param individual_b: with candidate 2
        :return: 2 offsprings
        """
        offspring = np.empty(offspring_size)
        # The point at which crossover takes place between two parents. Usually, it is at the center.
        crossover_point = np.uint8(offspring_size[1] / 2)

        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k + 1) % parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

        return offspring

    def calculate_fitness(self, population):
        accuracies = np.zeros(population.shape[0])
        idx = 0
        for pop in population:
            drop_list = []
            for x in range(len(pop)):
                if pop[x] == 0:
                    drop_list.append(x)
            print(pop)
            X_train = pd.DataFrame(ga.X_train)
            X_train = X_train.drop(columns=drop_list)
            print(X_train.shape, " After deletion")
            X_test = pd.DataFrame(ga.X_test)
            X_test = X_test.drop(columns=drop_list)

            nn = NeuralNetwork()
            print("Defining Model")
            model = nn.define_mode(X_train.shape[1])
            print("Launching the NN now")
            acc = nn.train(model, X_train, X_test, ga.y_train, ga.y_test)
            print("Accuracy ", acc)
            accuracies[idx] = acc
            idx += 1

            keras.backend.clear_session()
            print("Clearing Keras backend")
        return accuracies

    def select_pool(self, pop, fitness, num_parents):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = np.empty((num_parents, pop.shape[1]))
        for parent_num in range(num_parents):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = pop[max_fitness_idx, :]
            fitness[max_fitness_idx] = -99999999999
        return parents


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
    num_generations = 2

    for generation in range(num_generations):
        print("Generation : ", generation)
        accuracies = ga.calculate_fitness(new_population)
        parents = ga.select_pool(new_population, accuracies, 4)
        offspring_crossover = ga.crossover(parents,
                                           offspring_size=(pop_shape[0] - parents.shape[0], len(cols_list) - 1))

        offspring_mutation = ga.mutate(offspring_crossover)
        print(offspring_mutation)
        print()
        print(parents)
        print()
        print(offspring_crossover)
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
