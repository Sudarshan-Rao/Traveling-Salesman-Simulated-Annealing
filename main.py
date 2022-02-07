import numpy as np, random, operator, matplotlib.pyplot as plt,math

#Calculate distance for the route
def calcDistance(route, distanceMatrix):
    pathDistance = 0
    for i in range(0, len(route)):
        fromCity = route[i]
        if i + 1 < len(route):
            toCity = route[i + 1]
        else:
            toCity = route[0]
        pathDistance += distanceMatrix[fromCity][toCity]
    return pathDistance

def plotLearning(weights_list):
    plt.plot([i for i in range(len(weights_list))], weights_list)
    plt.ylabel('Weight')
    plt.xlabel('Iteration')
    plt.show()

def accept(candidate, distanceMatrix, current_weight, min_weight, temp):
    candidate_weight = calcDistance(candidate, distanceMatrix)
    if candidate_weight < current_weight:
        current_weight = candidate_weight
        current_solution = candidate
        if candidate_weight < min_weight:
            min_weight = candidate_weight
            best_solution = candidate
    else:
        if random.random() < (math.exp(-abs(candidate_weight - current_weight) / temp)):
            current_weight = candidate_weight
            current_solution = candidate
    return current_weight, min_weight

def simulatedAnnealing(max_temparature,stop_temparature, cooling_rate, stop_iteration, distMatrix, curr_solution):
    sample_size = len(curr_solution)
    iteration = 1
    best_solution = curr_solution
    curr_weight = calcDistance(curr_solution, distMatrix)
    initial_weight = curr_weight
    min_weight = curr_weight
    weight_list = [curr_weight]
    print('Initial weight: ', initial_weight)

    while max_temparature >= stop_temparature and iteration < stop_iteration:
        #swap cities
        candidate = list(curr_solution)
        l = random.randint(2, sample_size - 1)
        i = random.randint(0, sample_size - l)

        candidate[i: (i + l)] = reversed(candidate[i: (i + l)])
        #accept best routes or based on probability
        curr_weight, min_weight = accept(candidate, distMatrix,curr_weight, min_weight, max_temparature)
        max_temparature *= cooling_rate  # decreased temp
        iteration += 1
        weight_list.append(curr_weight)

    print('Minimum weight: ', min_weight)
    #print('Best Solution: ', best_solution)
    print('Improvement: ',
          round((initial_weight - min_weight) / (initial_weight), 4) * 100, '%')
    plotLearning(weight_list)

def main():
    temp = int(input("Please enter starting temperature:\n"))
    #temp = 1000
    stopping_temp = float(input("Please enter stopping temperature:\n"))
    #stopping_temp = 0.00000001
    alpha = float(input("Please enter cooling rate:\n"))
    #alpha = 0.9995
    stopping_iter = int(input("Please enter threshold/no of iterations:\n"))
    #stopping_iter = 50000

    #Number of cities
    population_size = int(input("Please enter number of cities:\n"))
    #population_size = 70

    # # Create n*n matrix and enter matrix or generate random numbers
    intermediateMatrix = np.random.randint(0, 200, size=(population_size, population_size))
    CityDistMatrix = np.tril(intermediateMatrix) + np.tril(intermediateMatrix, -1).T
    for i in range(0, population_size):
        CityDistMatrix[i][i] = 0
    #print(f'Distance Matrix: {CityDistMatrix}')

    #generate random route
    randomRoute = random.sample(range(population_size), population_size)

    #call simulated annealing
    simulatedAnnealing(temp, stopping_temp, alpha, stopping_iter, CityDistMatrix, randomRoute)

if __name__ == '__main__':
    main()
