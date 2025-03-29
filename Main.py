#Main.py
#This is an evolutionary algorithm to solve the following problem: 
#Given a 3-dimensional graph, consisting of X,Y,and Z. Where X, and Y are co-ordinates on a map, and Z is some value, place a series of n circles on the map such that:
#1. The sum of the Z value of the integer-points enclosed within the circles is maximised.
#2. Each point can only be counted once.
#In this instance, it relates to placing EV Charging stations on a real world map, where Z value represents traffic within that area. The question is, what is the most efficient placement of these stations given a limited budget.
#Matt Sefton, 29/03/2025

import numpy as np
import matplotlib.pyplot as plt
from ChargingStation import ChargingStation
from Solution import Solution
import random
from scipy.ndimage import gaussian_filter
import math

#Constants
size = 100                              #A size of 100 will create a map of -50, to 50 in x and y direction and consist of 10,000 points. Increasing this variable dramatically increases time complexity
chargingStationRadius = 10              #The area the station 'covers'
numberOfChargingStations = 10           #How many stations we have the budget for.

#Evolutionary Algorithm constants
numberOfGenerations=1000
initialPopulationSize=30


#Plotting Terrain
#----------------
# Generate a smoothed random value map
Z = np.random.rand(size, size) # values 0 to 5 (optional scaling)
Z = gaussian_filter(Z, sigma=10)    # smooth it out like terrain
Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))  # normalize again

# Create a coordinate grid
x_range = np.linspace(-(size/2), size/2, size)
y_range = np.linspace(-size/2, size/2, size)
X, Y = np.meshgrid(x_range, y_range)

# Plot the terrain
fig, ax = plt.subplots(figsize=(8, 6))
plt.ion()  # Turn on interactive mode for live updates
contour = ax.contourf(X, Y, Z, levels=20, cmap="jet")
plt.colorbar(contour, label="Value")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Charging Station Coverage")
ax.set_aspect('equal')
ax.set_xlim(-size/2, size/2)
ax.set_ylim(-size/2, size/2)

#Creates a set of n random stations within bounds of the map. Used for each Solution. Returns a set of stations
def createSetOfRandomChargingStations(n, radius, size):
    stations = []
    min_coord = -(size/2) + radius
    max_coord = (size/2) - radius
    while n > 0:
        x = random.uniform(min_coord, max_coord)
        y = random.uniform(min_coord, max_coord)
        stations.append(ChargingStation(x, y, radius))
        n -= 1
    return stations

#Calculates the total score of each solution by adding up the points within the stations radius. The 'counted' boolean ensures each point can only be counted once so we don't get multiple circles stacking,.
#The paramaters are the set of stations inside the solution, as well as the X,Y,Z representing our map
def evaluateSolution(stations, X, Y, Z):
    counted = np.zeros_like(Z, dtype=bool)
    total_value = 0

    for s in stations:
        # Compute distance grid from station to all grid points (vectorised)
        dist = np.sqrt((X - s.x)**2 + (Y - s.y)**2)

        # Create a mask where distance is within the radius AND not already counted
        within_radius = (dist <= s.radius) & (~counted)

        # Accumulate value where within radius
        total_value += np.sum(Z[within_radius])

        # Mark those points as counted
        counted[within_radius] = True

    return total_value

#This is our selection function inspired by simulated annealing (which I misunderstood). We take a max value and use it to assign a %chance of survival.
#Our 'temperature' in this case is set to the vaue of the best current solution (which will always survive) and then the chance of survival will decrease for lower solutions.
#The idea behind this is that we need diverse answers, and so there needs to be a possibility for solutions, however crap, to survive to the next generation.
#If the population exceeds the initial population size, the chance of survival is lowered so populations don't get out of control.
#
def probabilistic_survival(initialPopulationSize,population, temperature, bestSolution):
    survivors = []
    # Cap population growth impact so temperature doesn’t go to zero
    overcrowding_factor = max(1.0, len(population) / initialPopulationSize)
    penalty = math.exp(overcrowding_factor - 1)
    adjustedTemperature = temperature / penalty

    for solution in population:
            if solution==bestSolution:
                survivors.append(solution)
            else:
                delta = bestSolution.totalValue - solution.totalValue
                survival_prob = math.exp(-delta / adjustedTemperature)  # lower score = lower prob
                if random.random() < survival_prob:
                    survivors.append(solution)

    return survivors

#Just prints the current generation, it's population and the best value
def displayGenerationDetails():
    print(f"Generation {currentGeneration}: population= {len(population)} best value= {bestSolution.totalValue:.2f}")


#--------------------------
#EVOLUTIONARY ALGORITHM
#--------------------------
# This should probably be it's own class but It was a pain in the butthole trying to pass data into it and I couldn't be bothered. 
population=[]
currentPopulation=0
currentGeneration=0

#initialise a population
for _ in range(initialPopulationSize):
    stations = createSetOfRandomChargingStations(numberOfChargingStations, chargingStationRadius, size)
    thisSolutionScore = evaluateSolution(stations, X, Y, Z)
    individual=Solution(stations, thisSolutionScore)
    population.append(individual)

# Find the best solution
bestSolution = max(population, key=lambda s: s.totalValue)
formerbestSolution=bestSolution
currentPopulation=initialPopulationSize
#displayGenerationDetails()

#The main generational loop of the evolutionary algorithm
while numberOfGenerations>0:
    #kill off some of the population (population, maxValue, temperature)
    survivors = probabilistic_survival(initialPopulationSize, population, bestSolution.totalValue, bestSolution)
    #print(f"{len(survivors)} solutions survived out of {len(population)}")
    population=survivors

    # Ensure even number of survivors
    if len(population) % 2 != 0:
        population = population[:-1]
    
    #divide the population into two groups for imminent sexy times
    midpoint = len(population) // 2
    groupA = population[:midpoint]
    groupB = population[midpoint:]

    #Initialising sex (Not realistic)
    for parent1, parent2 in zip(groupA, groupB):
        reproduce=True
        if len(population)>initialPopulationSize:       #Another population control method, less chance of reproduction if large population.
            if random.random() > (initialPopulationSize/len(population)/4):
                    reproduce=False
        if reproduce:
            #Create a child with traits shared from both parents, the child then mutates and is added to the population
            child = parent1.sexual_reproduction(parent2)        
            child.mutate(minor_mutation_rate=0.15, major_mutation_rate=0.05,shift_amount=size/20,bounds=size/2)
            child.totalValue = evaluateSolution(child.stations, X, Y, Z)
            population.append(child)

    #the best solution also asexually reproduces with high mutation in addition to sexually reproducing. This is to help maintain solution diversity.
    #Otherwise, the best solution would just keep having kids and the population would become too homogenous.
    reproduce=True
    if len(population)>initialPopulationSize:
        if random.random() > (initialPopulationSize/len(population)/4): #Again, another population control measure.
                reproduce=False
    if reproduce:
        clone=bestSolution.clone()
        clone.mutate(minor_mutation_rate=0.05, major_mutation_rate=0.01,shift_amount=size/20,bounds=size/2)
        clone.totalValue = evaluateSolution(clone.stations, X, Y, Z)
        population.append(clone)

    bestSolution = max(population, key=lambda s: s.totalValue)
    currentGeneration+=1
    numberOfGenerations-=1

    #This is all just used for the graphics and output. We display new results and we also display every 100 generations just for an update.
    if bestSolution.totalValue>formerbestSolution.totalValue or currentGeneration % 100==0: 
        formerbestSolution=bestSolution
        #displayGenerationDetails()
        #plot
        ax.clear()
        ax.contourf(X, Y, Z, levels=20, cmap="jet")
        ax.set_xlim(-size/2, size/2)
        ax.set_ylim(-size/2, size/2)
        
        #Plot best solution
        for s in bestSolution.stations:
            circle = plt.Circle((s.x, s.y), s.radius, color='black', fill=False, linewidth=1)
            ax.add_patch(circle)

        #Labels and display
        ax.set_title(f"Generation {currentGeneration}- population: {len(population)}, Best Score: {bestSolution.totalValue:.2f}")
        plt.draw()
        plt.pause(0.01)  # Allows plot to update non-blocking
    
    if bestSolution.totalValue<formerbestSolution.totalValue:   #I think in some rare cases the best solution can die, so this is a backup to save it in this instance.
        bestSolution=formerbestSolution


plt.ioff()    # Turn OFF interactive mode
plt.show()    # This will BLOCK and keep the final plot open