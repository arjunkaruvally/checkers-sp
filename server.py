from flask import Flask
from flask import render_template

app = Flask(__name__)

"""
The route should return the checkers game page"
"""
@app.route('/')
def get_game():
	return render_template("index.html")

"""
should return the best move.
Request is an ajax Get Request with all the Moves
"""
@app.route("/moves")
def get_move():
	return "Got Moves"

app.debug = True
app.run()