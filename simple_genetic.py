from utils import *
import numpy as np
import random
import csv
import math

# Generates a random portfolio based on the list of available stocks.
# Assigns a random weight to each stock in the list. 
# Some stocks may end up with a weight of 0 due to randomness.
# Finally, the weights are normalized so that their total sum equals 1.
def get_random_portfolio():
    total_stocks = list(stocksRecords.keys())
    num_stocks = len(total_stocks)
    selected_stocks = total_stocks
    weights = np.random.rand(num_stocks)
    weights /= np.sum(weights)

    portfolio = {selected_stocks[i]: weights[i] for i in range(num_stocks)}

    return portfolio


# Implements the tournament selection method used in the genetic algorithm.
# The tournament size is always 5% of the population size.
# Each tournament selects a winner based on the evaluation function.
def tournament_selection(population, tournament_size, evaluation_function):
    tournament_contenders = random.sample(population, tournament_size)
    winner = max(tournament_contenders, key=lambda portfolio: evaluation_function(portfolio))
    return winner


# Implements a one-point crossover for two parent portfolios.
# A crossover point is randomly selected between 10% and 90% of the portfolio length.
# Assets from the first parent are added to the first child up to the crossover point,
# and assets from the second parent are added to the second child.
# After the crossover point, the remaining assets are added from the opposite parent.
# If an asset from one parent already exists in the child, instead of skipping the operation,
# the weights of the conflicting assets are swapped between the children.
# Finally, the weights of both children are normalized so that their total sum equals 1.
def crossover(parent1, parent2):
    len_parents = len(parent1)
    crossover_point = random.randint(int(0.1 * len_parents), int(0.9 * len_parents))
    child1 = {}
    child2 = {}

    stocks_parent1 = list(parent1.keys())
    stocks_parent2 = list(parent2.keys())

    for i in range(crossover_point):
        stock = stocks_parent1[i]
        child1[stock] = parent1[stock]
        stock = stocks_parent2[i]
        child2[stock] = parent2[stock]

    for j in range(crossover_point, len_parents):
        stock_parent1 = stocks_parent1[j]
        stock_parent2 = stocks_parent2[j]

        if stock_parent1 not in child2 and stock_parent2 not in child1:
            child1[stock_parent2] = parent2[stock_parent2]
            child2[stock_parent1] = parent1[stock_parent1]

        else:
            if stock_parent1 in child2:
                child1[stock_parent1] = child2[stock_parent1]
                child2[stock_parent1] = parent1[stock_parent1]
                if stock_parent2 not in child1:
                    child2[stock_parent2] = parent2[stock_parent2]

            if stock_parent2 in child1:
                child2[stock_parent2] = child1[stock_parent2]
                child1[stock_parent2] = parent2[stock_parent2]
                if stock_parent1 not in child2:
                    child1[stock_parent1] = parent1[stock_parent1]

    total_weight_child1 = sum(child1.values())
    total_weight_child2 = sum(child2.values())

    child1 = {stock: weight / total_weight_child1 for stock, weight in child1.items()}
    child2 = {stock: weight / total_weight_child2 for stock, weight in child2.items()}

    return child1, child2


# Implements the mutation operation for a portfolio.
# A random stock is selected, and its weight is adjusted by a random value 
# between -0.1 and 0.1. The weight is ensured to remain non-negative.
# After the mutation, the weights of all stocks in the portfolio are normalized 
# so that their total sum equals 1.
def mutate(portfolio):
    stock_to_mutate = random.choice(list(portfolio.keys()))
    portfolio[stock_to_mutate] = max(portfolio[stock_to_mutate] + np.random.uniform(-0.1, 0.1), 0)

    total_weight = sum(portfolio.values())
    portfolio = {stock: weight / total_weight for stock, weight in portfolio.items()}

    return portfolio

# Implements the genetic algorithm.
# The algorithm starts by initializing a population of random portfolios.
# For each generation, the following steps are performed:
# 1. Tournament selection is used to select the best portfolios (parents) based on the evaluation function.
# 2. A one-point crossover is applied to generate new portfolios (children) from the selected parents.
#    - parent1 is chosen from the best portfolios (i.e., one of the tournament winners).
#    - parent2 is randomly selected from the current population, excluding parent1.
# 3. Mutation is applied to some portfolios with a probability defined by the mutation rate.
# 4. The population is updated by combining the best portfolios from the previous generation 
#    with the newly generated portfolios, ensuring the population size remains constant.
# The algorithm logs the best score and average score for each generation to a CSV file.
# At the end, the best portfolio found during all generations is returned.
def genetic_algorithm(population_size, generations, mutation_rate, evaluation_function):
    # Initialize population
    population = [get_random_portfolio() for _ in range(population_size)]

    best_solution = max(population, key=evaluation_function)

    print(f"Initial best value: {evaluation_function(best_solution)}")

    # CSV file for logging
    log_file = 'genetic_algorithm_log.csv'
    with open(log_file, mode='w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Generation', 'BestScore', 'AverageScore'])
        writer.writerow([0, evaluation_function(best_solution), np.mean([evaluation_function(p) for p in population])])

    for generation in range(generations):
        # Tournament Selection
        tournament_size = math.ceil(0.05 * population_size)
        tournament_winners = [tournament_selection(population, tournament_size, evaluation_function) for _ in range(population_size)]

        sorted_tournament_winners = sorted([(portfolio, evaluation_function(portfolio)) for portfolio in tournament_winners], key=lambda x: x[1], reverse=True)

        best_parents = [portfolio for portfolio, _ in sorted_tournament_winners[:population_size // 2]]

        # Crossover between two parents
        new_generation = []
        while len(new_generation) < population_size:
            parent1 = random.choice(best_parents)
            parent2 = random.choice([p for p in population if p != parent1])

            #parent1 = random.choice(best_parents)
            #possible_mates = [p for p in population if p != parent1]
            #if not possible_mates:
            #    parent2 = random.choice(population)
            #else:
            #    parent2 = random.choice(possible_mates)

            child1, child2 = crossover(parent1, parent2)
            new_generation.append(child1)
            new_generation.append(child2)

        for i in range(len(new_generation)):
            if random.random() < mutation_rate:
                new_generation[i] = mutate(new_generation[i])

        auxiliary_population = ([portfolio for portfolio, _ in sorted_tournament_winners[:population_size // 4]] + new_generation)
        auxiliary_population = sorted(auxiliary_population, key=evaluation_function, reverse=True)
        population = auxiliary_population[:population_size]

        current_best = max(population, key=evaluation_function)
        if evaluation_function(current_best) > evaluation_function(best_solution):
            best_solution = current_best

        average_score = np.mean([evaluation_function(p) for p in population])

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([generation + 1, evaluation_function(best_solution), average_score])

        print(f"Generation {generation + 1}/{generations} - Best Score: {evaluation_function(best_solution)}")

    return best_solution
