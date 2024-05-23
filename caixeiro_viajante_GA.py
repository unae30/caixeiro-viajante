import random
import itertools
from scipy.spatial import distance
import time


# Constants, experiment parameters
CITIES = ["Itaúna", "Divinópolis", "Belo Horizonte", "Montes Claros", "Viçosa", "Lavras", "Ouro Preto", "Janaúba", "Monte Verde", "Governador Valadares"]
NUM_CITIES = len(CITIES)
POPULATION_SIZE = 30
MIXING_NUMBER = 2
MUTATION_RATE = 0.05

# Define distances between cities (fictional distances)
CITY_DISTANCES = {
    "Itaúna": {
        "Itaúna": 0,
        "Divinópolis": 41.1,
        "Belo Horizonte": 86.3,
        "Montes Claros": 476,
        "Viçosa": 293,
        "Lavras": 198,
        "Ouro Preto": 169,
        "Janaúba": 608,
        "Monte Verde": 441,
        "Governador Valadares": 391,
    },

    "Divinópolis": {
        "Itaúna": 41.1,
        "Divinópolis": 0,
        "Belo Horizonte": 124,
        "Montes Claros": 467,
        "Viçosa": 331,
        "Lavras": 166,
        "Ouro Preto": 207,
        "Janaúba": 598,
        "Monte Verde": 408,
        "Governador Valadares": 429,
    },

    "Belo Horizonte": {
        "Itaúna": 86.3,
        "Divinópolis": 124,
        "Belo Horizonte": 0,
        "Montes Claros": 420,
        "Viçosa": 226,
        "Lavras": 238,
        "Ouro Preto": 102,
        "Janaúba": 553,
        "Monte Verde": 481,
        "Governador Valadares": 315,
    },

    "Montes Claros": {
        "Itaúna": 476,
        "Divinópolis": 467,
        "Belo Horizonte": 420,
        "Montes Claros": 0,
        "Viçosa": 637,
        "Lavras": 634,
        "Ouro Preto": 513,
        "Janaúba": 135,
        "Monte Verde": 877,
        "Governador Valadares": 497,
    },

    "Viçosa": {
        "Itaúna": 293,
        "Divinópolis": 331,
        "Belo Horizonte": 226,
        "Montes Claros": 637,
        "Viçosa": 0,
        "Lavras": 296,
        "Ouro Preto": 128,
        "Janaúba": 767,
        "Monte Verde": 586,
        "Governador Valadares": 305,
    },

    "Lavras": {
        "Itaúna": 198,
        "Divinópolis": 166,
        "Belo Horizonte": 238,
        "Montes Claros": 634,
        "Viçosa": 296,
        "Lavras": 0,
        "Ouro Preto": 281,
        "Janaúba": 762,
        "Monte Verde": 276,
        "Governador Valadares": 276,
    },

    "Ouro Preto": {
        "Itaúna": 169,
        "Divinópolis": 207,
        "Belo Horizonte": 102,
        "Montes Claros": 60,
        "Viçosa": 513,
        "Lavras": 128,
        "Ouro Preto": 0,
        "Janaúba": 645,
        "Monte Verde": 525,
        "Governador Valadares": 340,
    },

    "Janaúba": {
        "Itaúna": 608,
        "Divinópolis": 598,
        "Belo Horizonte": 553,
        "Montes Claros": 135,
        "Viçosa": 767,
        "Lavras": 762,
        "Ouro Preto": 645,
        "Janaúba": 0,
        "Monte Verde": 1180,
        "Governador Valadares": 629,
    },

    "Monte Verde": {
        "Itaúna": 441,
        "Divinópolis": 408,
        "Belo Horizonte": 481,
        "Montes Claros": 877,
        "Viçosa": 586,
        "Lavras": 276,
        "Ouro Preto": 525,
        "Janaúba": 1180,
        "Monte Verde": 0,
        "Governador Valadares": 793,
    },

    "Governador Valadares": {
        "Itaúna": 391,
        "Divinópolis": 429,
        "Belo Horizonte": 315,
        "Montes Claros": 497,
        "Viçosa": 305,
        "Lavras": 276,
        "Ouro Preto": 340,
        "Janaúba": 629,
        "Monte Verde": 793,
        "Governador Valadares": 0,
    }     
}


# Create the fitness score - Total distance of the route
def fitness_score(route, CITY_DISTANCES):
    total_distance = 0

    for i in range(NUM_CITIES - 1):
        city1 = route[i]
        city2 = route[i + 1]
        total_distance += CITY_DISTANCES[city1][city2]

    # Add distance from the last city back to the starting city
    total_distance += CITY_DISTANCES[route[-1]][route[0]]

    return total_distance


# Create the selection operator acording their fitness score
# Select best solutions for next step: crossover
def selection(population, CITY_DISTANCES):
    parents = []

    for ind in population:
        # Select parents with probability proportional to their fitness score
        #função de aptidão -> Isso significa que indivíduos com escores de aptidão 
        # mais altos têm uma probabilidade maior de serem selecionados.
        if random.random() < 1 / fitness_score(ind, CITY_DISTANCES):
            parents.append(ind)

    return parents

# Create the crossover operator
# Combine features of each solution using a crossover point
def crossover(parents):
    # Random indexes to cross routes
    cross_points = random.sample(range(1, NUM_CITIES), MIXING_NUMBER - 1)
    offsprings = []

    # All permutations of parents
    permutations = list(itertools.permutations(parents, MIXING_NUMBER))

    for perm in permutations:
        offspring = perm[0][:]  # Copy the first parent's route

        for parent_idx, cross_point in enumerate(cross_points):
            # Sublist of the parent to be crossed
            parent_part = perm[parent_idx + 1][cross_point:]
            for city in parent_part:
                if city not in offspring:
                    offspring[cross_point:] = parent_part
                    break

        offsprings.append(offspring)

    return offsprings

# Create the routine to mutate a solution
# A operator to create diversity in the population
def mutate(route):
    # Swap two random cities in the route
    idx1, idx2 = random.sample(range(NUM_CITIES), 2)
    route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

# Print the solution
def print_found_goal(population, CITY_DISTANCES, to_print=True):
    best_route = min(population, key=lambda route:fitness_score(route, CITY_DISTANCES))
    distance_travelled = fitness_score(best_route, CITY_DISTANCES)
    
    if to_print:
        print(f'Best Route: {best_route}. Distance: {distance_travelled}')

    return best_route

# Create the routine to implement the evolution
def evolution(population, CITY_DISTANCES):
    # Select individuals to become parents
    parents = selection(population, CITY_DISTANCES)

    # Recombination. Create new offsprings
    offsprings = crossover(parents)

    # Mutation
    offsprings = list(map(mutate, offsprings))

    # Introduce top-scoring individuals from the previous generation and keep top fitness individuals
    new_gen = offsprings

    for ind in population:
        new_gen.append(ind)

    new_gen = sorted(new_gen, key=lambda ind: fitness_score(ind, CITY_DISTANCES))[:POPULATION_SIZE]

    return new_gen

# Create the initial population (solutions)
def generate_population(CITY_DISTANCES):
    population = []

    for _ in range(POPULATION_SIZE):
        new_route = random.sample(CITIES, NUM_CITIES)
        population.append(new_route)

    return population

# Running the experiment
generation = 0

# Generate Random Population
population = generate_population(CITY_DISTANCES)

t1 = time.time()
#Generations until finding the solution
while generation < 1:  # Adjust the stopping criteria as needed
    print(f'Generation: {generation}')
    print_found_goal(population, CITY_DISTANCES)
    population = evolution(population, CITY_DISTANCES)
    generation += 1

t2 = time.time()

total_time = t2 - t1
print(f'Total time: {total_time} seconds')
