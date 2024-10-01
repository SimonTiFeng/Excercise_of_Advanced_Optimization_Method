import numpy as np
import bisect
import matplotlib.pyplot as plt
import time

population_size = 500
gene_length = 4
crossover_rate = 0.6
mutation_rate = 0.05
num_generations = 1000
n_elites = 10
 
population = np.random.uniform(low=-10, high=10, size=(population_size, gene_length))
population = np.round(population, 4) 

start_time = time.time() 

def function(population):
    temporary = np.sum(population, axis=1)
    temporary += 1
    return 1 / temporary

def index_select(cumulative_probabilities):
    rand_value = np.random.random()
    index = bisect.bisect_left(cumulative_probabilities, rand_value)
    if index >= len(cumulative_probabilities):
        index = len(cumulative_probabilities) - 1
    return index

def crossover(population, crossover_rate):
    population_size, gene_length = population.shape
    child = np.zeros((population_size, gene_length))
    
    Cost = function(population)
    Positive_Cost = Cost + min(Cost)
    total_Positive_Cost = np.cumsum(Positive_Cost)
    total_sum = total_Positive_Cost[-1]
    probabilities = Positive_Cost / total_sum
    cumulative_probabilities = np.cumsum(probabilities)
    

    for i in range(0, population_size, 2): 
        if not np.isfinite(cumulative_probabilities).all():
            print("Cumulative probabilities contain invalid values.")
            break
        parent1_idx = index_select(cumulative_probabilities)
        parent2_idx = index_select(cumulative_probabilities)
        
        while parent1_idx == parent2_idx:
            parent2_idx = index_select(cumulative_probabilities)

        parent1 = population[parent1_idx]
        parent2 = population[parent2_idx]

        if np.random.random() < crossover_rate:
            alpha = np.random.random()  
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1
        else:
            child1, child2 = parent1, parent2
        
        child[i] = np.clip(child1, -10, 10)
        if i + 1 < population_size:
            child[i + 1] = np.clip(child2, -10, 10)
    
    return child

def mutation(population, mutation_rate):
    pop_size, gene_length = population.shape
    for i in range(pop_size):
        if np.random.random() < mutation_rate:
            mutation_point = np.random.randint(0, gene_length - 1)
            population[i, mutation_point] = np.random.uniform(low=-10, high=10)
    return population

best_solutions = []

def elitism(population, fitness, n_elites):
    elite_indices = np.argsort(fitness)[-n_elites:]  
    elites = population[elite_indices]              
    return elites

for generation in range(num_generations):
    print(f"Generation {generation+1}")
    fitness = function(population)
    
    best_fitness_idx = np.argmax(fitness)
    best_solution = population[best_fitness_idx]
    best_solutions.append(fitness[best_fitness_idx])
    elites = elitism(population, fitness, n_elites)
    child_population = crossover(population, crossover_rate)
    mutated_population = mutation(child_population, mutation_rate)
    mutated_population[:n_elites] = elites  
    population = mutated_population

plt.plot(best_solutions)
plt.title('Genetic Algorithm Optimization')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.show()

end_time = time.time()
print(f" {end_time - start_time:.2f}秒")