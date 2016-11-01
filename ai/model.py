from NeuralNetwork import NeuralNetwork
from Tree import Node
from GeneticEvolution import GeneticEvolution

import numpy as np

import math
import random
import sys

'''
	Game Board Legend
		5 - invalid square
		0 - no coin
		-1 - player coin
		-2 - player king coin
		1 - opponent coin
		2 - opponent king coin
'''

class PlayingAgent:
	def __init__(self,population_limit=8):
		self.genetic_evolution = GeneticEvolution()
		self.population_limit = population_limit

	def init_generation(self):
		print "Creating Population"
		self.players = self.genetic_evolution.init_population(population_limit = self.population_limit)
		print "Population created"

	def in_boundary(self,pos):
		if pos[0] < 0 or pos[0] > 7:
			return False
		if pos[1] < 0 or pos[1] > 7:
			return False
		return True

	def exec_move(self, sim_board, start, end):
		sim_board = np.array(sim_board)
		moving_coin = sim_board.item(start)
		sim_board[start] = 0

		if end[0] == 7 and moving_coin > 0:
			sim_board[end] = 2
		elif end[0] == 0 and moving_coin < 0:
			sim_board[end] = -2
		else:
			sim_board[end] = moving_coin
		
		mov_vector = np.subtract(end,start)

		if abs(mov_vector[0]) == 2:
			cut_position = tuple(np.add(start,np.divide(mov_vector,2)))
			sim_board[cut_position] = 0

		return sim_board

	def get_possible_moves(self,simulated_board,x,y,mover):
		ret = []	#ret value for jump
		sur = []	#ret value for single cell hop

#abs(sim_board) == 2
		if x>0 and y>0 and ((mover%2!=0) or simulated_board[x][y]==2):
			
			if simulated_board.item(x-1, y-1)==0:
				sur.append(( x-1, y-1 ))
			
			elif simulated_board.item(x-1, y-1)<0:
				target = (x-2,y-2)

				if self.in_boundary(target) and simulated_board.item(target) == 0:
					ret.append(target)
					
					sim_temp = self.exec_move(simulated_board,(x,y),target)
					temp_ret = self.get_possible_moves(sim_temp,target[0],target[1],mover)
					
					if temp_ret[1]!=None and len(temp_ret[1]) > 0:
						ret.append(temp_ret[1])
						
		if x<7 and y>0 and ((mover%2==0) or simulated_board[x][y]==2):
			if simulated_board.item(x+1, y-1)==0:
				sur.append(( x+1, y-1 ))
			elif simulated_board.item(x+1, y-1)<0:
				target = (x+2,y-2)

				if self.in_boundary(target) and simulated_board.item(target) == 0:
					ret.append(target)
					
					sim_temp = self.exec_move(simulated_board,(x,y),target)
					temp_ret = self.get_possible_moves(sim_temp,target[0],target[1],mover)
					
					if temp_ret[1]!=None and len(temp_ret[1]) > 0:
						ret.append(temp_ret[1])
					
		if x<7 and y<7 and ((mover%2==0) or simulated_board[x][y]==2):
			if simulated_board.item(x+1, y+1)==0:
				sur.append(( x+1, y+1 ))
			elif simulated_board.item(x+1, y+1)<0:
				target = (x+2,y+2)

				if self.in_boundary(target) and simulated_board.item(target) == 0:
					ret.append(target)
					
					sim_temp = self.exec_move(simulated_board,(x,y),target)
					temp_ret = self.get_possible_moves(sim_temp,target[0],target[1],mover)

					if ret!=None and len(temp_ret[1]) > 0:
						ret.append(temp_ret[1])
						
		if x>0 and y<7 and ((mover%2!=0) or simulated_board[x][y]==2):
			if simulated_board.item(x-1, y+1)==0:
				sur.append(( x-1, y+1 ))
			elif simulated_board.item(x-1, y+1)<0:
				target = (x-2,y+2)

				if self.in_boundary(target) and simulated_board.item(target) == 0:
					ret.append(target)

					sim_temp = self.exec_move(simulated_board,(x,y),target)
					temp_ret = self.get_possible_moves(sim_temp,target[0],target[1],mover)
					
					if temp_ret[1]!=None and len(temp_ret[1]) > 0:
						ret.append(temp_ret[1])
		# print [sur,ret]
		return [sur,ret]

	def get_jump_combinations(self, jumps):
		
		a = []

		for x in range(0,len(jumps)):
			
			if str(type(jumps[x])) == "<type 'list'>":
				v = a.pop()
				b = self.get_jump_combinations(jumps[x])

				for y in range(0,len(b)):
					if str(type(v)) == "<type 'tuple'>":
						if str(type(b[y])) == "<type 'tuple'>":
							b[y] = [ v, b[y] ]
						else:
							b[y] = [ v ] + b[y]
					else:
						if str(type(b[y])) == "<type 'tuple'>":
							b[y] = v.append(b[y])
						else:
							b[y] = v + b[y]
				a = a+b
			else:
				a.append(jumps[x])
		return a

	def get_optimum_move(self, tree, depth, max_depth, alpha=-500, beta=500, maximising_player=True):

		# if depth == max_depth:
		# 	return [ 0, [] ]

		if len(tree.children) <= 0 or depth == max_depth:
			return [ tree.score, tree.moves ]

		req_score = 0
		req_index = []
		max = 0

		for x in range(0,len(tree.children)):			
			ret = self.get_optimum_move(tree.children[x], depth+1, max_depth, alpha, beta, not maximising_player)
			score = ret[0]

			if maximising_player:
				if x==0:
					req_score = score
					req_index = [0]
				elif score > req_score:
					req_score = score
					req_index = [x]
				elif score==req_score:
					req_index.append(x)

				#alpha beta pruning
				alpha = alpha if alpha>req_score else req_score
				# alpha = max(alpha,req_score)
				if beta <= alpha:
					# print "beta cutoff\n"
					break

			else:
				if x==0:
					req_score = score
					req_index = [0]
				elif score < req_score:
					req_score = score
					req_index = [x]
				elif score==req_score:
					req_index.append(x)

				#alpha beta pruning
				beta = beta if beta<req_score else req_score
				# beta = min(beta,req_score)
				if beta <= alpha:
					# print "alpha cutoff\n"
					break

		if len(req_index) > 0:
			index = random.choice(req_index)
			return [ req_score, tree.children[index].moves ]
		else:
			return [0 , []]

	def minmax(self, simulated_board, mover=0, depth=2, player1=0, player2=1):

		'''
			- keep mover as always zero and invert players for consistency in using neural network
			- global moves array(2D)
			- 2nd dimension is the current move
		'''

		simulated_board = np.array(simulated_board)

		self.Tree = Node(simulated_board = simulated_board)
		# current_node = self.Tree
		# root_node = self.Tree
		# current_node.simulated_board = simulated_board
		
		stack = []
		stack.append(self.Tree)

		while len(stack) > 0:

			current_node = stack.pop()
			simulated_board = current_node.simulated_board
			current_player = current_node.current_player

			if not current_player:
				simulated_board = self.invert_board(simulated_board)

			for x in range(0,8):
				for y in range(0,8):
					if simulated_board[x][y] != 5 and simulated_board[x][y] > 0:
						playing_coin = (x,y)

						moves = self.get_possible_moves(simulated_board,x,y,mover)
						jumps = moves[1]
						moves = moves[0]
						
						if len(jumps) > 0:
							moves = self.get_jump_combinations(jumps)

						for move in moves:
							temp_sim = np.array(simulated_board)	
							if str(type(move)) == "<type 'tuple'>":
								temp_sim = self.exec_move(temp_sim,(x,y),move)
								move = [move]
							else:
								source = (x,y)
								for z in move:
									temp_sim = self.exec_move(temp_sim,source,z)
									source = z
							move = [(x,y)] + move

							board_config = []

							temp_sim_for_cost = np.array(temp_sim)
							if not current_player:
								temp_sim_for_cost = self.invert_board(temp_sim_for_cost)
							
							for r in temp_sim_for_cost:
								for c in r:
									if c != 5:
										board_config.append(c)
							
							heuristic_cost = self.players[player1].get_heuristic_cost(board_config)

							if not current_player:
								temp_sim = self.invert_board(temp_sim)
							
							self.node = Node(score=heuristic_cost, moves=move, simulated_board=temp_sim)
							self.node.set_parent(current_node)
							current_node.add_child(self.node)
			
			if current_node.depth < depth-1:
				for x in current_node.children:
					stack.append(x)

		# self.Tree.postorder(self.Tree)
		
		return self.get_optimum_move(self.Tree,0,depth)

	def invert_board(self, board):
		
		board = np.array(board)

		for x in range(0,8):
			for y in range(0,8):
				if board[x][y] != 5:
					board[x][y] = -1 * board[x][y]

		board = np.rot90(board,2)

		return board

	def tournament(self, maxium_game_moves=100):

		for x in range(0,self.population_limit):
			self.players[x].reset_fitness()

		x = range(0,self.population_limit)

		# for x in range(0, self.population_limit, 2):

		print "-------------------tournament started---------------------"

		game_number=0
		number_draws=0
		number_wins=0
		while len(x) > 1:
			game_number = game_number+1
			player1_index = x.pop()
			player2_index = x.pop()

			print "\n----------\ngame: "+str(game_number)
			print "generation: "+str(self.genetic_evolution.generation)
			print "player1: "+str(player1_index)
			print "player2: "+str(player2_index)

			board = np.array([
				[ 1, 5, 1, 5, 1, 5, 1, 5],
				[ 5, 1, 5, 1, 5, 1, 5, 1],
				[ 1, 5, 1, 5, 1, 5, 1, 5],
				[ 5, 0, 5, 0, 5, 0, 5, 0],
				[ 0, 5, 0, 5, 0, 5, 0, 5],
				[ 5,-1, 5,-1, 5,-1, 5,-1],
				[-1, 5,-1, 5,-1, 5,-1, 5],
				[ 5,-1, 5,-1, 5,-1, 5,-1]
			]);

			board_f = True
			draw = False

			for y in range(0,maxium_game_moves):
				# player1 = x + (y%2)
				# player2 = x + ((player1+1)%2)
				sys.stdout.write(" moves %d/%d \r" % (y+1,maxium_game_moves))
				sys.stdout.flush()
				# print "\n"

				if y%2 == 0:
					player1 = player1_index
					player2 = player2_index
				else:
					player1 = player2_index
					player2 = player1_index

				ret = self.minmax(board,depth=3,player1=player1,player2=player2)
				
				if len(ret[1]) < 2:
					# print "game end"
					number_wins = number_wins+1
					print "\n"
					print "winner: "+str(player2)
					print "moves : "+str(y)
					# print board
					x.insert(0,player2)
					self.players[player2].fitness = self.players[player2].fitness + self.genetic_evolution.fitness_factor(True,moves=y)
					self.players[player1].fitness = self.players[player1].fitness + self.genetic_evolution.fitness_factor(False,moves=y)					
					break

				if y == maxium_game_moves-1 :
					draw = True

				# print "move "+str(y)
				# print "player "+str(player1)
				# print "opponent "+str(player2)
				# print "moves"

				for z in range(0,len(ret[1])-1):
					# print "from"
					# print ret[1][z]
					# print "to"
					# print ret[1][z+1]
					board = self.exec_move(board, ret[1][z], ret[1][z+1])
				
				# print "board "
				# print board

				board = self.invert_board(board)
				board_f = not board_f

			if draw:
				print "\n"
				print "game draw"
				number_draws = number_draws+1
				# print board_f
				
				coins1 = self.get_coins(board, positive=True)
				coins2 = self.get_coins(board, positive=False)

				if coins1[0] > coins2[0]:
					print "winner by coins: "+str(player1_index)
					x.insert(0,player1_index)
					self.players[player1_index].fitness = self.players[player2].fitness + self.genetic_evolution.fitness_factor(True,moves=y)
				elif coins1[0] < coins2[0]:
					print "winner by coins: "+str(player2_index)
					x.insert(0,player2_index)
					self.players[player2_index].fitness = self.players[player2].fitness + self.genetic_evolution.fitness_factor(True,moves=y)
				elif coins1[1] > coins2[1]:
					print "winner by kings: "+str(player1_index)
					x.insert(0,player1_index)
					self.players[player1_index].fitness = self.players[player2].fitness + self.genetic_evolution.fitness_factor(True,moves=y)
				elif coins1[1] < coins2[1]:
					print "winner by kings: "+str(player2_index)
					x.insert(0,player2_index)
					self.players[player2_index].fitness = self.players[player2].fitness + self.genetic_evolution.fitness_factor(True,moves=y)
				else:
					if self.players[player1_index].fitness > self.players[player2_index].fitness:
						print "winner by fitness "+str(player1_index)
						x.insert(0,player1_index)
					elif self.players[player1_index].fitness < self.players[player2_index].fitness:
						print "winner by fitness "+str(player2_index)
						x.insert(0,player2_index)
					else:
						prob = np.random.rand()
						if prob < 0.5:
							print "choosing "+str(player1_index)
							x.insert(0,player1_index)
						else:
							print "choosing "+str(player2_index)
							x.insert(0,player2_index)
			if board_f:
				print board
			else:
				print self.invert_board(board)

		print "tournament summary"
		print "winner: "+str(x[0])
		print "matches played: 7"
		print "matches drawn: "+str(number_draws)
		print "matches not draw: "+str(number_wins)

		# return x[0]

	def get_coins(self, board, positive=True):
		coins = 0
		kings = 0
		for x in range(0,7):
			for y in range(0,7):
				if board[x][y]!=5 and positive and board[x][y] > 0:
					coins = coins+1
					if board[x][y] == 2:
						kings = kings+1
				elif board[x][y]!=5 and (not positive) and board[x][y] < 0:
					coins = coins+1
					if board[x][y] == -2:
						kings = kings+1
		return [coins, kings]

	def trainer(self,gen_limit=100):

		while True:
			current_generation = self.genetic_evolution.generation
			print "Generation : "+str(current_generation)
			self.tournament(200)
			for x in range(0,self.population_limit):
				print str(x)+" : "+str(self.players[x].fitness)

			self.save_evolution()
			self.players = self.genetic_evolution.get_next_generation()
			# current_generation = current_generation+1

	def load_saved_evolution(self, file_path = "neural_net/saved_net_"):
		print "Loading saved evolution"
		
		file = open(file_path+"generations","r")
		gen = file.read()
		gen = gen.split(",")
		file.close()

		file = open(file_path+"fitness","r")
		fit = file.read()
		fit = fit.split(",")
		file.close()

		max_gen = 0

		for x in range(0,len(self.players)):
			self.players[x].model.load_weights(file_path+str(x))
			gen[x] = int(gen[x])
			fit[x] = float(fit[x])
			self.players[x].generation = gen[x]
			if gen[x] > max_gen:
				max_gen = gen[x]

		self.genetic_evolution.population = self.players
		self.genetic_evolution.generation = max_gen
		print "Load Complete"
		self.players = self.genetic_evolution.get_next_generation()	

	def save_evolution(self, file_path="neural_net/saved_net_"):
		print "Saving evolution"

		generations = []
		fitness = []

		for x in range(0,len(self.players)):
			self.players[x].model.save_weights(file_path+str(x))
			generations.append(str(self.players[x].generation))
			fitness.append(str(self.players[x].fitness))
		
		file = open(file_path+"generations", 'w')
		file.write(",".join(generations))
		file.close()

		file = open(file_path+"fitness", 'w')
		file.write(",".join(fitness))
		file.close()

		print "Save Complete"

ai_agent = PlayingAgent(population_limit=8) #use powers of two for playing tournaments

ai_agent.init_generation()
ai_agent.load_saved_evolution()
ai_agent.trainer()

board = np.array([
				[ 1, 5, 1, 5, 1, 5, 1, 5],
				[ 5, 1, 5, 1, 5, 1, 5, 1],
				[ 1, 5, 1, 5, 1, 5, 1, 5],
				[ 5, 0, 5, 0, 5, 0, 5, 0],
				[ 0, 5, 0, 5, 0, 5, 0, 5],
				[ 5,-1, 5,-1, 5,-1, 5,-1],
				[-1, 5,-1, 5,-1, 5,-1, 5],
				[ 5,-1, 5,-1, 5,-1, 5,-1]
			]);

# print ai_agent.minmax(board,0,2)
# print my_nn.exec_move(board,(2,0),(3,1))
# print ai_agent.get_possible_moves(board,0,0,2)

# jumps = [(2, 2), [(4, 0), [(6, 2)], (4, 4), [(6, 2), (6, 6)]]]

# print ai_agent.get_jump_combinations_1(jumps)