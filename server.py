from flask import Flask, request
from flask import render_template
from ai import model

app = Flask(__name__)

"""
The route should return the checkers game page"
"""
@app.route('/')
def get_game():
	return render_template("index.html")

"""
should return the best move.
Request is an ajax Get Request with the gameboard
"""
@app.route("/moves", methods=["POST"])
def get_move():
	ai_agent = model.PlayingAgent(population_limit=8) #use powers of two for playing tournaments
	board = request.json["data"]
	#moves = ai_agent.minmax(board)
	print board
	return "Got Moves"

app.debug = True
app.run()