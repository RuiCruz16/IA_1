import numpy as np
import random
import math
import csv
from utils import *

CHANCE_TO_SWAP = 0.3

# Generates a random portfolio by selecting a random number of stocks from the available list.
# Each selected stock is assigned a random weight, and the weights are normalized so that their total sum equals 1.
def get_random_portfolio():
    total_stocks = list(stocksRecords.keys())
    num_stocks = np.random.randint(1, len(total_stocks) + 1)
    selected_stocks = np.random.choice(total_stocks, num_stocks, replace=False)
    weights = np.random.rand(num_stocks)
    weights /= np.sum(weights)
    portfolio = {selected_stocks[i]: weights[i] for i in range(num_stocks)}

    return portfolio


# Generates a neighboring portfolio by slightly adjusting the weight of a random stock.
# Optionally, it may swap a stock with another available stock not currently in the portfolio.
# The weights are normalized after the adjustments.
def get_neighbor_portfolio(portfolio, perturbation=0.05):
    new_portfolio = portfolio.copy()
    stock_to_adjust = np.random.choice(list(new_portfolio.keys()))

    adjustment = np.random.uniform(-perturbation, perturbation)
    new_portfolio[stock_to_adjust] = max(new_portfolio[stock_to_adjust] + adjustment, 0)

    if np.random.rand() < CHANCE_TO_SWAP:
        del new_portfolio[stock_to_adjust]
        available_stocks = list(set(stocksRecords.keys()) - set(new_portfolio.keys()))
        if available_stocks:
            new_stock = np.random.choice(available_stocks)
            new_portfolio[new_stock] = np.random.rand() * perturbation

    total_weight = sum(new_portfolio.values())
    new_portfolio = {stock: weight / total_weight for stock, weight in new_portfolio.items()}

    return new_portfolio


# Compares two portfolios and returns True if the first portfolio is better than the second
# based on the evaluation function provided.
def is_better_than(portfolio1, portfolio2, evaluation_function):
    return evaluation_function(portfolio1) > evaluation_function(portfolio2)


# Calculates the "energy" of a portfolio, which is the negative of its evaluation score.
# Lower energy corresponds to a better portfolio.
def get_energy(portfolio, evaluation_function):
    return -evaluation_function(portfolio)

'''
    Implements the simulated annealing algorithm for portfolio optimization.
    The algorithm starts with an initial random portfolio and explores the neighborhood 
    of solutions by generating neighboring portfolios. It accepts a new portfolio if it 
    improves the score or with a probability that decreases with the worsening of the score 
    and the temperature. The temperature is reduced over iterations, reducing the chance 
    of accepting worse solutions over time. The best portfolio is logged and returned.
'''
def simulated_annealing(n_iterations, initial_temp, reduction_rate, eval):
    initial_portfolio = get_random_portfolio()
    current_portfolio = initial_portfolio
    best_portfolio = initial_portfolio


    current_temp = initial_temp

    iterations = 1

    log_file = 'simulated_annealing_log.csv'
    with open(log_file, mode='w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'BestScore', 'Temperature','Portfolio'])
        writer.writerow([0, eval(best_portfolio), current_temp, best_portfolio])

    for i in range(n_iterations):
        print("Iteration: ", i)
        new_portfolio = get_neighbor_portfolio(current_portfolio)

        if is_better_than(new_portfolio, best_portfolio, eval):
            best_portfolio = new_portfolio
            print("New Best Portfolio: ", eval(best_portfolio))
        else:
            delta_e = get_energy(new_portfolio, eval) - get_energy(current_portfolio, eval)
            if delta_e < 0 or random.random() < math.exp(-delta_e / current_temp):
                current_portfolio = new_portfolio
        current_temp *= reduction_rate

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([iterations, eval(best_portfolio), current_temp, best_portfolio])
        iterations += 1

    print("Initial Portfolio: ", initial_portfolio)
    print("Initial Portfolio Energy: ", get_energy(initial_portfolio, eval))
    print("Best Portfolio:", best_portfolio)
    print("Best Portfolio Energy:", get_energy(best_portfolio, eval))
    return best_portfolio
