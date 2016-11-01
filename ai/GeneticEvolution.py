from NeuralNetwork import NeuralNetwork
from operator import attrgetter

import numpy as np

class GeneticEvolution:
	def __init__(self):

		self.population = []
		self.generation = 180

		#evolution coefficients available for tuning
		self.move_fitness_coefficient = 10.0

	def init_population(self, population_limit=10):
		for x in range(0,population_limit):
			self.population.append(NeuralNetwork())
			self.population[x].set_random_seed((x*10)+5)
			self.population[x].randomize_weights(scale = 4)
			# self.population[x].fitness = x

		np.random.shuffle(self.population)	
		return self.population

	def fitness_factor(self,win,moves=1,draw=False):
		fitness = 0.0
		if draw:
			if win:
				fitness = 0.5 + (self.move_fitness_coefficient/moves)
			else:
				fitness = 0.5 - (self.move_fitness_coefficient/moves)
			return fitness
		if win:
			fitness = 1.0 + (self.move_fitness_coefficient/moves)
		else:
			fitness = -(self.move_fitness_coefficient/moves)
		return fitness

	def mutate(self,x):
		prob = np.random.rand()
		mutated_val = x
		if prob < 0.2:
			mu = 0
			sigma = 2
			mutated_val = np.random.normal(0,2)
			if mutated_val < -2:
				mutated_val = -2
			if mutated_val > 2:
				mutated_val = 2
		return mutated_val

	def get_next_generation(self):

		self.generation = self.generation+1
		new_gen=[]
		temp_pop = sorted(self.population, key=lambda nnet: nnet.fitness, reverse=True)

		#Natural selection - survival of the fittest
		temp_pop[0].tag = "alpha-gen: "+str(temp_pop[0].generation)
		new_gen.append(temp_pop[0])

		male_genes = new_gen[0].model.layers

		#Crossover
		print "starting crossover"
		for x in range(1, len(temp_pop)-3):
			for y in range(0, len(temp_pop[x].model.layers)):
				shape1 = temp_pop[x].model.layers[y].get_weights()[0].shape
				shape2 = temp_pop[x].model.layers[y].get_weights()[1].shape

				male_gene = np.array(male_genes[y].get_weights()[0]).flatten()
				female_gene = np.array(temp_pop[x].model.layers[y].get_weights()[0]).flatten()
				
				for z in range(0,len(female_gene)):
					prob = np.random.rand()
					if prob < 0.6:
						female_gene[z] = male_gene[z]

				female_gene = female_gene.reshape(shape1)
				sec_gene = np.zeros(shape2)

				temp_pop[x].generation = self.generation
				temp_pop[x].model.layers[y].set_weights([female_gene, sec_gene])

			temp_pop[x].tag = "child "+str(x)
			new_gen.append(temp_pop[x])
		
		print "crossover complete"
		# print len(new_gen)

		#Mutation
		print "Mutation starting"
		for x in range(len(temp_pop)-3, len(temp_pop)):
			for y in range(0,len(temp_pop[x].model.layers)):
				female_gene = temp_pop[x].model.layers[y].get_weights()[0]

				vector_mutate = np.vectorize(self.mutate)

				female_gene = vector_mutate(female_gene)
				sec_gene = np.zeros(temp_pop[x].model.layers[y].get_weights()[1].shape)

				temp_pop[x].generation = self.generation
				temp_pop[x].model.layers[y].set_weights([female_gene, sec_gene])

			temp_pop[x].tag = "mutant "+str(x)
			new_gen.append(temp_pop[x])

		print "mutation complete"
		# print len(new_gen)

		for x in range(0,len(new_gen)):
			new_gen[x].fitness = 0.0

		self.population = new_gen

		np.random.shuffle(self.population)
		return self.population
