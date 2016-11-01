import numpy as np
from PlayingAgent import PlayingAgent

ai_agent = PlayingAgent(population_limit=8) #use powers of two for playing tournaments

ai_agent.init_generation()
ai_agent.load_saved_evolution()
# ai_agent.save_evolution()
ai_agent.trainer()

# board = np.array([
# 				[ 1, 5, 1, 5, 1, 5, 1, 5],
# 				[ 5, 1, 5, 1, 5, 1, 5, 1],
# 				[ 1, 5, 1, 5, 1, 5, 1, 5],
# 				[ 5, 0, 5, 0, 5, 0, 5, 0],
# 				[ 0, 5, 0, 5, 0, 5, 0, 5],
# 				[ 5,-1, 5,-1, 5,-1, 5,-1],
# 				[-1, 5,-1, 5,-1, 5,-1, 5],
# 				[ 5,-1, 5,-1, 5,-1, 5,-1]
# 			]);

# print ai_agent.minmax(board,0,2)
# print my_nn.exec_move(board,(2,0),(3,1))
# print ai_agent.get_possible_moves(board,0,0,2)

# jumps = [(2, 2), [(4, 0), [(6, 2)], (4, 4), [(6, 2), (6, 6)]]]

# print ai_agent.get_jump_combinations_1(jumps)
