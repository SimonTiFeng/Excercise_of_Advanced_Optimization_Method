import numpy as np

population_size = 500
gene_length = 4
lower_bound = -10
upper_bound = 10
num_generations = 1000
crossover_rate = 0.6
mutation_rate = 0.05

def objective_function(x):
    return 1 / (np.sum(x) + 1)

def initialize_population(pop_size, gene_length, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, (pop_size, gene_length))

def selection(population, fitness, num_parents):
    parents_indices = np.argsort(fitness)[-num_parents:]
    return population[parents_indices]

def crossover(parents, pop_size):
    num_parents = parents.shape[0]
    offspring = np.empty((pop_size, parents.shape[1]))
    crossover_point = np.random.randint(1, parents.shape[1], size=pop_size)
    
    for i in range(pop_size):
        parent1 = parents[i % num_parents]
        parent2 = parents[(i + 1) % num_parents]
        crossover_idx = crossover_point[i]
        offspring[i, :crossover_idx] = parent1[:crossover_idx]
        offspring[i, crossover_idx:] = parent2[crossover_idx:]
    
    return offspring

def mutate(population, mutation_rate, lower_bound, upper_bound):
    mutations = np.random.rand(*population.shape) < mutation_rate
    population[mutations] = np.random.uniform(lower_bound, upper_bound, np.sum(mutations))

def genetic_algorithm(pop_size, gene_length, lower_bound, upper_bound, num_generations, crossover_rate, mutation_rate):
    population = initialize_population(pop_size, gene_length, lower_bound, upper_bound)
    
    for generation in range(num_generations):
        fitness = np.apply_along_axis(objective_function, 1, population)
        parents = selection(population, fitness, pop_size // 2)
        offspring = crossover(parents, pop_size)
        mutate(offspring, mutation_rate, lower_bound, upper_bound)
        population = np.vstack((parents, offspring))
        
        best_fitness = np.max(fitness)
        best_individual = population[np.argmax(fitness)]
        print(f"Generation {generation + 1}: Best fitness = {best_fitness}, Best individual = {best_individual}")

    return best_individual

best_solution = genetic_algorithm(population_size, gene_length, lower_bound, upper_bound, num_generations, crossover_rate, mutation_rate)
print("Best solution:", best_solution)
print("Objective value of the best solution:", objective_function(best_solution))
