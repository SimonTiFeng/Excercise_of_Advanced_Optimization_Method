import numpy as np
import matplotlib.pyplot as plt

city_coords = np.array([
    [48, 21], [52, 26], [55, 50], [50, 50], [41, 46], [51, 42], [55, 45],
    [38, 33], [33, 34], [45, 35], [40, 37], [50, 30], [55, 34], [54, 38],
    [26, 13], [15, 5], [21, 48], [29, 39], [33, 44], [15, 19], [16, 19],
    [12, 17], [50, 40], [22, 53], [21, 36], [20, 30], [26, 29], [40, 20],
    [36, 26], [62, 48], [67, 41], [62, 35], [65, 27], [62, 24], [55, 20],
    [35, 51], [30, 50], [45, 42], [21, 45], [36, 6], [6, 25], [11, 28],
    [26, 59], [30, 60], [22, 22], [27, 24], [30, 20], [35, 16], [54, 10],
    [50, 15], [44, 13], [35, 60], [40, 60], [40, 66], [31, 76], [47, 66],
    [50, 70], [57, 72], [55, 65], [2, 38], [7, 43], [9, 56], [15, 56],
    [17, 64], [55, 57], [62, 57], [70, 64], [64, 4], [59, 5], [50, 4],
    [60, 15], [66, 14], [66, 8], [43, 26]
])

def init_population(pop_size, num_cities):
    return np.array([np.random.permutation(num_cities) for _ in range(pop_size)])

def calc_fitness(population, city_coords):
    return np.array([1 / np.sum([np.linalg.norm(city_coords[ind[i]] - city_coords[ind[i + 1]]) for i in range(len(ind) - 1)]) for ind in population])

def selection(population, fitness):
    probabilities = fitness / np.sum(fitness)
    selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
    return population[selected_indices]

def crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            swap_with = np.random.randint(0, len(individual))
            individual[i], individual[swap_with] = individual[swap_with], individual[i]
    return individual

def genetic_algorithm(city_coords, pop_size=100, generations=500, crossover_rate=0.8, mutation_rate=0.02):
    num_cities = len(city_coords)
    population = init_population(pop_size, num_cities)
    best_fitness_history = []

    for generation in range(generations):
        print(f'Generation {generation}')
        fitness = calc_fitness(population, city_coords)
        best_fitness_history.append(np.max(fitness))

        selected_population = selection(population, fitness)
        next_population = []
        for i in range(0, pop_size, 2):
            if np.random.rand() < crossover_rate and i + 1 < len(selected_population):
                child1, child2 = crossover(selected_population[i], selected_population[i + 1])
                next_population.extend([child1, child2])
            else:
                next_population.extend([selected_population[i], selected_population[i + 1]])

        next_population = next_population[:pop_size]
        population = np.array([mutate(ind, mutation_rate) for ind in next_population])

    return best_fitness_history

def plot_fitness_history(fitness_history):
    plt.plot(fitness_history)
    plt.title('Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.show()

fitness_history = genetic_algorithm(city_coords, pop_size=100, generations=500)
plot_fitness_history(fitness_history)