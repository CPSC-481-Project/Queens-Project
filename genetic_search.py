from search import *
# ======================= NQUEEN's Code =======================


# class NQueensProblem(Problem):
#     """The problem of placing N queens on an NxN board with none attacking
#     each other. A state is represented as an N-element array, where
#     a value of r in the c-th entry means there is a queen at column c,
#     row r, and a value of -1 means that the c-th column has not been
#     filled in yet. We fill in columns left to right.
#     >>> depth_first_tree_search(NQueensProblem(8))
#     <Node (7, 3, 0, 2, 5, 1, 6, 4)>
#     """

#     def __init__(self, N):
#         super().__init__(tuple([-1] * N))
#         self.N = N

#     def actions(self, state):
#         """In the leftmost empty column, try all non-conflicting rows."""
#         if state[-1] != -1:
#             return []  # All columns filled; no successors
#         else:
#             col = state.index(-1)
#             return [row for row in range(self.N)
#                     if not self.conflicted(state, row, col)]

#     def result(self, state, row):
#         """Place the next queen at the given row."""
#         col = state.index(-1)
#         new = list(state[:])
#         new[col] = row
#         return tuple(new)

#     def conflicted(self, state, row, col):
#         """Would placing a queen at (row, col) conflict with anything?"""
#         return any(self.conflict(row, col, state[c], c)
#                    for c in range(col))

#     def conflict(self, row1, col1, row2, col2):
#         """Would putting two queens in (row1, col1) and (row2, col2) conflict?"""
#         return (row1 == row2 or  # same row
#                 col1 == col2 or  # same column
#                 row1 - col1 == row2 - col2 or  # same \ diagonal
#                 row1 + col1 == row2 + col2)  # same / diagonal

#     def goal_test(self, state):
#         """Check if all columns filled, no conflicts."""
#         if state[-1] == -1:
#             return False
#         return not any(self.conflicted(state, state[col], col)
#                        for col in range(len(state)))

#     def h(self, node):
#         """Return number of conflicting queens for a given node"""
#         num_conflicts = 0
#         for (r1, c1) in enumerate(node.state):
#             for (r2, c2) in enumerate(node.state):
#                 if (r1, c1) != (r2, c2):
#                     num_conflicts += self.conflict(r1, c1, r2, c2)

#         return num_conflicts


# ======================= Genetic Algorithm =======================


# TODO: THOUGHT PROCESS OF GA
'''
GA()
   initialize population - DONE
   find fitness of population - DONE

   while (termination criteria is reached) do
      parent selection
      crossover with probability pc
      mutation with probability pm
      decode and fitness calculation
      survivor selection
      find best
   return best

'''


def genetic_search(problem, ngen=1000, pmut=0.1, n=20):
    """Call genetic_algorithm on the appropriate parts of a problem.
    This requires the problem to have states that can mate and mutate,
    plus a value method that scores states."""

    # NOTE: This is not tested and might not work.
    # TODO: Use this function to make Problems work with genetic_algorithm.

    # NOTE: WHAT IS THE INITIAL STATE OF A NQUEENS PROBLEM?
    s = problem.initial
    states = [problem.result(s, a) for a in problem.actions(s)]
    random.shuffle(states)

    state_num = len(states)

    # GENE POOL UPDATE ACCORDINGLY:
    gene_pool = range(state_num)

    # INITIAL POPULATION OF CANDIDATE SOLUTIONS:
    population1 = init_population(
        state_num, gene_pool, state_num)

    # return genetic_algorithm(states[:n], problem.value, gene_pool, f_thres=None)
    print(problem.h)
    print()
    return genetic_algorithm(population1, problem.h, gene_pool, f_thres=None)


def random_chromosome(size):  # making random chromosomes
    return [random.randint(1, size) for _ in range(size)]


def genetic_algorithm(population, fitness_fn, gene_pool=[0, 1], f_thres=None, ngen=1000, pmut=0.1):
    """[Figure 4.8]"""

    print("-GENETIC ALGORITHM RUN-")
    pop_len = len(population)

    # FIND CONFLICTS OF INITIALIZED POPULATION
    conflicts = []
    ind_list = []
    for index in range(pop_len):
        individual = Node(population[index])
        ind_list.append(individual)
        conflicts.append(fitness_fn(individual))
        print(individual)

    print(ind_list)
    print()
    print("CONFLICTS FROM NQUEEN:")
    print(conflicts)
    print()

    # CALCULATE THE FITNESS OF THE POPULATION USING AMOUNT OF CONFLICTS
    fitness_population = []
    for x in range(len(conflicts)):
        fitness_population.append(1 / ((conflicts[x]) + 1))
    print("CALCULATING FITNESS WITH CONFLICTS:")
    print(fitness_population)
    print()

    # # GENE POOL UPDATE ACCORDINGLY:
    # gene_pool = range(pop_num)

    # # INITIAL POPULATION OF CANDIDATE SOLUTIONS:
    # population1 = init_population(pop_num, gene_pool, pop_num)
    # print(population)

    # print()
    # # FIND FITNESS OF INITIALIZE POPULATION:
    # print("FITNESS:")
    # fitness_population = np.zeros([len(population), 1])
    # for i, individual in enumerate(population):
    #     temp = individual

    # print(fitness_population)

    #     # Check for any queens conflicting one another ---idk if we're suppose to implement here or Class NQueens has something

    #     # Took the horizontal and diagonal checks on internet
    #     horizontal_queen_checks = len(temp) - len(set(temp))
    #     diagonal_queen_checks = 0
    #     for x in range(len(temp)):
    #         for y in range(x + 1, len(temp)):
    #             if temp[y] == temp[x] + y - x or temp[y] == temp[x] - y + x:
    #                 diagonal_queen_checks += 1

    #     # Find the fitness of the population using the fitness function
    #     fitness_population[i] = 1 / \
    #         ((horizontal_queen_checks+diagonal_queen_checks) + 1)
    # print(fitness_population)

    # NOTE: WHAT IS f_thres???? What does fitness_threshold do -- Find the most fit within population1

    print()

    new_population = []
    for i in range(ngen):
        sel = select(2, ind_list, fitness_fn)
        print(sel)
        selected = []
        for k in range(len(sel)):
            for j in range(len(population)):
                if sel[k].state == population[j]:
                    selected.append(population[j])

        print("SELECTED LIST:")
        print(selected)

        recomb = recombine(*selected)
        print(recomb)
        new_population = [mutate(recomb, gene_pool, pmut)
                          for i in range(len(population))]
        # print(population)
        print(new_population)

        # fittest_individual = fitness_threshold(fitness_fn, f_thres, population)
        # if fittest_individual:
        #     print(fittest_individual)

    # for i in range(ngen):
    #     population = [mutate(recombine(*select(2, population, fitness_fn)), gene_pool, pmut)
    #                   for i in range(len(population))]

        fittest_individual = fitness_threshold(fitness_fn, f_thres, population)
        if fittest_individual:
            print(fittest_individual)
    # return fittest_individual


#     return max(population, key=fitness_fn)

    # return max(population, key=fitness_fn)
    return None


def fitness_threshold(fitness_fn, f_thres, population):
    print("-FITNESS_THRESHOLD-")
    # if not f_thres:
    #     return None

    fittest_individual = max(population, key=fitness_fn)
    print(fittest_individual)
    if fitness_fn(fittest_individual) >= f_thres:
        return fittest_individual

    return None


def init_population(pop_number, gene_pool, state_length):
    """Initializes population for genetic algorithm
    pop_number  :  Number of individuals in population
    gene_pool   :  List of possible values for individuals
    state_length:  The length of each individual"""

    print("-INIT_POPULATION-")
    g = len(gene_pool)
    population = []
    for i in range(pop_number):
        new_individual = [gene_pool[random.randrange(
            0, g)] for j in range(state_length)]
        population.append(new_individual)

    return population


def select(r, population, fitness_fn):
    print("-SELECT-")
    fitnesses = map(fitness_fn, population)
    sampler = weighted_sampler(population, fitnesses)
    return [sampler() for i in range(r)]


def recombine(x, y):
    print("-RECOMBINE-")
    n = len(x)
    c = random.randrange(0, n)
    return x[: c] + y[c:]


def recombine_uniform(x, y):
    print("-RECOMBINE_UNIFORM-")
    n = len(x)
    result = [0] * n
    indexes = random.sample(range(n), n)
    for i in range(n):
        ix = indexes[i]
        result[ix] = x[ix] if i < n / 2 else y[ix]

    return ''.join(str(r) for r in result)


def mutate(x, gene_pool, pmut):
    print("-MUTATE-")
    if random.uniform(0, 1) >= pmut:
        return x

    n = len(x)
    g = len(gene_pool)
    c = random.randrange(0, n)
    r = random.randrange(0, g)

    new_gene = gene_pool[r]
    return x[: c] + [new_gene] + x[c + 1:]


# ======================= MAIN DRIVER =======================
if __name__ == '__main__':
    # initial = 8
    size = eval(input(" - Please input the size of the board (4~15): "))
    print()
    prob = NQueensProblem(size)
    print(prob.initial)
    genetic_search(prob)
