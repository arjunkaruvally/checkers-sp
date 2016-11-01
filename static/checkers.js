var gameboard = [
[ 1, 5, 1, 5, 1, 5, 1, 5],
[ 5, 1, 5, 1, 5, 1, 5, 1],
[ 0, 5, 0, 5, 0, 5, 1, 5],
[ 5, 0, 5,-1, 5, 0, 5, 1],
[ 0, 5, 0, 5, 0, 5, 0, 5],
[ 5,-1, 5,-1, 5,-1, 5,-1],
[ 1, 5, 1, 5,-1, 5,-1, 5],
[ 5,-1, 5, 0, 5,-1, 5,-1] ]

var curPlayer = 0;
var selected = -1;
var possible_moves = [];

function draw() {

	var board = document.getElementById("board");
	var tableText = "";
	for(var row = 0; row < 8; row++) {
		tableText += "<tr>";
		
		for(var col = 0; col < 8; col++) {
			var id = row*8 + col;
			var cl;
			// cl = "white"
			if(row%2!=col%2)
				cl = "white";
			else
				cl = "black";
			tableText +="<td id='"+id+"' class ='"+cl+"' onclick=select("+id+")>"
			if(gameboard[row][col] != 5 && gameboard[row][col] != 0)
			{
				// tableText += "<p>"+gameboard[row][col]+"</p>";
				cls = "";
				switch(gameboard[row][col])
				{
					case -1:
						style = "player"
						break
					case 1:
						style = "opponent"
						break
				}

				tableText = tableText+'<div id="div'+id+'" style="'+style+'" ></div>'
			}
			tableText +="</td>"
		}
		tableText += "</tr>";
	}
	board.innerHTML = tableText;
}

function select(id) {
	var x = Math.floor(id/8);
	var y = id%8;
	console.log(possible_moves)

	if(gameboard[x][y] == -1 && curPlayer == 0) {

//Clearing previous move
		var ele = 0
		for(ele in possible_moves)
		{
			document.getElementById(""+possible_moves[ele]+"").style.border="none"
			document.getElementById("div"+selected+"").style.border="none"
		}

		selected = id
		document.getElementById("div"+id).style.border="solid blue 2px"
		possible_moves = []

//top left
		if(gameboard[x-1][y-1]==0)
		{
			var target = (x-1)*8+(y-1)
			possible_moves.push(target)
		}
		else if(gameboard[x-1][y-1]>0 && gameboard[x-2][y-2]==0)
		{
			var target = (x-2)*8+(y-2)
			possible_moves.push(target)
		}

//top right
		if(gameboard[x-1][y+1]==0)
		{
			var target = (x-1)*8+(y+1)
			possible_moves.push(target)
		}
		else if(gameboard[x-1][y+1]>0 && gameboard[x-2][y+2]==0)
		{
			var target = (x-2)*8+(y+2)
			possible_moves.push(target)
		}

//bottom right
		if(gameboard[x+1][y+1]==0)
		{
			var target = (x+1)*8+(y+1)
			possible_moves.push(target)
		}
		else if(gameboard[x+1][y+1]>0 && gameboard[x+2][y+2]==0)
		{
			var target = (x+2)*8+(y+2)
			possible_moves.push(target)
		}

//bottom left
		if(gameboard[x+1][y-1]==0)
		{
			var target = (x+1)*8+(y-1)
			possible_moves.push(target)
		}
		else if(gameboard[x+1][y-1]>0 && gameboard[x+2][y-2]==0)
		{
			var target = (x+2)*8+(y-2)
			possible_moves.push(target)
		}

		for(x in possible_moves)
		{
			// console.log(possible_moves[x])
			document.getElementById(""+possible_moves[x]+"").style.border="solid blue 2px"	
		}
		console.log(possible_moves)
	}
	else if(possible_moves.indexOf(id) > 0){

	}

	return
		console.log(x, y);
		//change the gameboard
		draw();
		curPlayer = 1;
		getAIMove();
	}
}
var gameOb = {"data": gameboard};

function getAIMove() {
	//send an ajax request to /moves
	$.ajax({
	    type: "POST",
	    url: "/moves",
	    // The key needs to match your method's input parameter (case-sensitive).
	    data: gameOb,
	    contentType: "application/json; charset=utf-8",
	    dataType: "json",
	    success: function(data){alert(data);},
	    failure: function(errMsg) {
	        alert(errMsg);
	    }
});

	//get Response
	//change gameboard
	curPlayer = 0;
	draw();
}

window.onload = function () {
	draw();

}