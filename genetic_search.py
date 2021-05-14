from search import *
from utils import *
import time


def genetic_search(problem, ngen=1000, pmut=0.1, n=20):
    """Call genetic_algorithm on the appropriate parts of a problem.
    This requires the problem to have states that can mate and mutate,
    plus a value method that scores states."""
    maxConflict = (problem.N * (problem.N - 1)) / 2 #get the most amount of conflicts. Found from resource - totaldatascience.com
    print("Maximum conflicts " + str(maxConflict))
    gene_pool = [x for x in range(problem.N)] #Get hte gene pool and use its length to have the length of each individual

    # print("GENEPOOL : " + str(gene_pool))
    # print("LENGTH GENE POOL : " + str(len(gene_pool)))
    
    population1 = init_population(n, gene_pool, len(gene_pool))
    print("POPULATION" + str(population1))

    return genetic_algorithm(population1, problem.h, gene_pool, f_thres=maxConflict, ngen = ngen)


def genetic_algorithm(population, fitness_fn, gene_pool=[0, 1], f_thres=None, ngen=1000, pmut=0.1):
    """[Figure 4.8]"""
    start = time.time()
    for i in range(ngen):
        population = [mutate(recombine(*select(2, population, fitness_fn)), gene_pool, pmut)
                      for i in range(len(population))]

        fittest_individual = fitness_threshold(fitness_fn, f_thres, population)
        if fittest_individual:
            end = time.time()
            print("Time elapsed: " + str(end - start))
            return fittest_individual

    end = time.time()
    print("Time elapsed: " + str(end - start))  
    return max(population, key=fitness_fn)


def fitness_threshold(fitness_fn, f_thres, population):
    if not f_thres:
        return None

    fittest_individual = max(population, key=fitness_fn)
    if fitness_fn(fittest_individual) >= f_thres:
        return fittest_individual

    return None


def init_population(pop_number, gene_pool, state_length):
    """Initializes population for genetic algorithm
    pop_number  :  Number of individuals in population
    gene_pool   :  List of possible values for individuals
    state_length:  The length of each individual"""
    g = len(gene_pool)
    population = []
    for i in range(pop_number):
        new_individual = [gene_pool[random.randrange(0, g)] for j in range(state_length)]
        population.append(new_individual)

    return population


def select(r, population, fitness_fn):
    fitnesses = map(fitness_fn, population)
    sampler = weighted_sampler(population, fitnesses)
    return [sampler() for i in range(r)]


def recombine(x, y):
    n = len(x)
    c = random.randrange(0, n)
    return x[:c] + y[c:]


def recombine_uniform(x, y):
    n = len(x)
    result = [0] * n
    indexes = random.sample(range(n), n)
    for i in range(n):
        ix = indexes[i]
        result[ix] = x[ix] if i < n / 2 else y[ix]

    return ''.join(str(r) for r in result)


def mutate(x, gene_pool, pmut):
    if random.uniform(0, 1) >= pmut:
        return x

    n = len(x)
    g = len(gene_pool)
    c = random.randrange(0, n)
    r = random.randrange(0, g)

    new_gene = gene_pool[r]
    return x[:c] + [new_gene] + x[c + 1:]


if __name__ == '__main__':
    # initial = 8
    size = eval(input(" - Please input the size of the board (4~15): "))
    print()
    ngen = eval(input(" - Please input the n amount of loop to do GA: "))
    print()
    prob = NQueensProblem(size)
    print("INITIAL STATE: " + str(prob.initial))
    print()
    print("-- INITIATING GENETIC SEARCH ALGORITHM --")
    result = genetic_search(prob, ngen)
    print("-- FINISHED FINDING NQUEEN RESULT --")

    #print(result)
    length = len(result)

    board = [[0 for x in range(length)] for y in range(length)]
    colC = 0
    for rowC in result:
        board[rowC][colC] = 1
        colC += 1

    for x in board:
        print(x)