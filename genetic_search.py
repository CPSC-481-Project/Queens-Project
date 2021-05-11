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

# def h(self, node):
#      """Return number of conflicting queens for a given node"""
#       num_conflicts = 0
#        for (r1, c1) in enumerate(node.state):
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
      parent selection -DONE
      crossover with probability pc AKA RECOMBINE - DONE
      mutation with probability pm -DONE
      decode and fitness calculation AKA F-thres from Genetic_Search - Done

      ---BELOW IS BASICALLY FItness_Threshold FUNCTION ----
      survivor selection
      find best
      -Done

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

    state_len = len(states)

    # GENE POOL UPDATE ACCORDINGLY (1 to 8 for default):
    gene_pool = range(state_len)
    # print("GENE:")
    # print(gene_pool)
    best_fit = 0
    # INITIAL POPULATION OF CANDIDATE SOLUTIONS:
    population1 = init_population(
        state_len, gene_pool, state_len)

    # FIND CONFLICTS OF INITIALIZED POPULATION
    conflicts = []
    ind_list = []
    print("INITIAL POPULATION:")
    for index in range(state_len):
        individual = Node(population1[index])
        # individual = population1[index]
        ind_list.append(individual)
        conflicts.append(problem.h(individual))
        # conflicts.append(the_better_h(individual))
        print(individual)

    # print()
    # print("CONFLICTS FROM NQUEEN:")
    # print(conflicts)
    # print(min(conflicts))
    best_fit = (1 / (min(conflicts) + 1))
    print()

    return genetic_algorithm(ind_list, problem.h, gene_pool, f_thres=best_fit)


def genetic_algorithm(population, fitness_fn, gene_pool=[0, 1], f_thres=None, ngen=1000, pmut=0.1):
    # originally ngen=1000
    """[Figure 4.8]"""

    # print("-GENETIC ALGORITHM RUN-")

    for i in range(ngen):
        # print("FOR LOOP STARTING GA COUNTER = " + str(i))
        # NOTE: Seperated Select and recombine from mutate to visually see what is going on

        # NOTE: ORIGINAL CODE DOWN HERE
        population = [mutate(recombine(*select(2, population, fitness_fn)), gene_pool, pmut)
                      for i in range(len(population))]
        # print("PRINTING NEW POPULATION:")
        # print(population)

        fittest_individual = fitness_threshold(
            fitness_fn, f_thres,  population)

        # print()
        # print("-RETURN TO GENETIC ALGO FROM FITNESS_THRESHOLD W/ FITTEST_INDIVIDUAL-")

        # print(fittest_individual)

        if (1/(fitness_fn(fittest_individual) + 1)) > f_thres:
            # return fittest_individual
            minFit = fittest_individual
            f_thres = (1/(fitness_fn(fittest_individual) + 1))

        if f_thres == 1.0:
            print("FITNESS:")
            print(f_thres)
            print(minFit)
            # print("RETURNING FITTEST INDIVIDUAL TO MAIN:")
            # print(minFit)
            return minFit

        # Go through the loop till we reach 1.0 threshold or run out of ngen and return the biggest threshhold reached
    print("FITNESS:")
    print(f_thres)
    minFit = fittest_individual
    print(minFit)

    # return max(population, key=fitness_fn)
    return minFit


def fitness_threshold(fitness_fn, f_thres, population):
    # print("-FITNESS_THRESHOLD-")
    for i in range(len(population)):
        population[i] = Node(population[i])

    # print("NODE IN FITNESS:")
    # print(population)

    # NOTE: UNCOMMENT TO CHECK IF IT IS GETTING MAX IN FITTEST INDIVIDUAL
    # for x in range(len(population)):
    #     pop = fitness_fn(population[x])
    #     print(population[x])
    #     print(pop)

    # NOTE: OG CODE BELOW
    if not f_thres:
        return None

    fittest_individual = max(population, key=fitness_fn)
    # print("FITTEST INDIVIDUAL IN F_THRES:")
    # print(fittest_individual)
    if (1/(fitness_fn(fittest_individual) + 1)) >= f_thres:
        # print("CHECKING FITNESS_FN FUNCTION IN F_THRES:")
        # print("---AMOUNT OF CONFLICTS IN FITTEST INDIVIDUAL:")
        # print(fitness_fn(fittest_individual))
        # print("---FITNESS OF FITTEST INDIVIDUAL:")
        # print((1/(fitness_fn(fittest_individual) + 1)))
        # print("---F_THRES COMPARED TO FITNESS OF FITTEST INDIVIDUAL:")
        # print(f_thres)
        # NOTE: OG CODE BELOW
        return fittest_individual

    return fittest_individual


def init_population(pop_number, gene_pool, state_length):
    """Initializes population for genetic algorithm
    pop_number  :  Number of individuals in population
    gene_pool   :  List of possible values for individuals
    state_length:  The length of each individual"""

    # print("-INIT_POPULATION-")
    g = len(gene_pool)
    population = []
    for i in range(pop_number):
        new_individual = [gene_pool[random.randrange(
            0, g)] for j in range(state_length)]
        population.append(new_individual)

    return population


def select(r, population, fitness_fn):
    # print("-SELECT-")
    fitnesses = map(fitness_fn, population)
    sampler = weighted_sampler(population, fitnesses)

    node_select = [sampler() for i in range(r)]
    # print(node_select)
    list_select = []

    for i in range(2):
        list_select.append(node_select[i].state)
    # print(list_select)
    return list_select


def recombine(x, y):
    # print("-RECOMBINE-")
    n = len(x)
    c = random.randrange(0, n)
    # print(x[: c] + y[c:])
    return x[: c] + y[c:]


def recombine_uniform(x, y):
    # print("-RECOMBINE_UNIFORM-")
    n = len(x)
    result = [0] * n
    indexes = random.sample(range(n), n)
    for i in range(n):
        ix = indexes[i]
        result[ix] = x[ix] if i < n / 2 else y[ix]

    return ''.join(str(r) for r in result)


def mutate(x, gene_pool, pmut):
    # print("-MUTATE-")
    if random.uniform(0, 1) >= pmut:
        # print("random.uni:")
        # print(x)
        # print()
        return x

    n = len(x)
    g = len(gene_pool)
    c = random.randrange(0, n)
    r = random.randrange(0, g)

    new_gene = gene_pool[r]
    # print("NEW GENE:")
    # print(new_gene)
    return x[: c] + [new_gene] + x[c + 1:]


# ======================= MAIN DRIVER =======================
if __name__ == '__main__':
    # initial = 8
    size = eval(input(" - Please input the size of the board (4~15): "))
    print()
    prob = NQueensProblem(size)
    print(prob.initial)
    print()
    print("-- INITIATING GENETIC SEARCH ALGORITHM --")
    result = genetic_search(prob)
    print("-- FINISHED FINDING NQUEEN RESULT --")

    print(result)
    list_result = result.state
    length = len(list_result)

    board = [[0 for x in range(length)] for y in range(length)]
    colC = 0
    for rowC in list_result:
        board[rowC][colC] = 1
        colC += 1

    for x in board:
        print(x)
