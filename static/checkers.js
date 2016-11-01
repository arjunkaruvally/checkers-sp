var gameboard = [
[ 1, 5, 1, 5, 1, 5, 1, 5],
[ 5, 1, 5, 1, 5, 1, 5, 1],
[ 1, 5, 1, 5, 1, 5, 1, 5],
[ 5, 0, 5, 0, 5, 0, 5, 0],
[ 0, 5, 0, 5, 0, 5, 0, 5],
[ 5,-1, 5,-1, 5,-1, 5,-1],
[-1, 5,-1, 5,-1, 5,-1, 5],
[ 5,-1, 5,-1, 5,-1, 5,-1] ]

var curPlayer = 0;
var selected = -1;
var possible_moves = [];
var jumps = false;
var move_in_progress=false;
var exec_jump = false;

function draw() {

	var board = document.getElementById("board");
	board.innerHTML = ""
	var tableText = "";
	for(var row = 0; row < 8; row++) {
		tableText += "<tr>";
		
		for(var col = 0; col < 8; col++) {
			var id = row*8 + col;
			var cl;
			if(row%2!=col%2)
				cl = "white";
			else
				cl = "black";
			tableText +="<td id='"+id+"' class ='"+cl+"' onclick=select("+id+") class='not-selected'>"
			if(gameboard[row][col] != 5 && gameboard[row][col] != 0)
			{
				// tableText += "<p>"+gameboard[row][col]+"</p>";
				var style = "";
				var inner_text = "";
				switch(gameboard[row][col])
				{
					case -2:
						style = "player-coin"
						inner_text = "K"
						if(check_jumps(row,col)){
							jumps = true
							$("#jumpsPresent").text("true")
						}
						break
					case -1:
						style = "player-coin"
						if(check_jumps(row,col)){
							jumps = true
							$("#jumpsPresent").text("true")
						}
						break
					case 1:
						style = "opponent-coin"
						break
					case 2:
						style = "opponent-coin"
						inner_text = "K"
						break
				}

				tableText = tableText+'<div id="div'+id+'" class="'+style+' not-selected" >'+inner_text+'</div>'
			}
			tableText +="</td>"
		}
		tableText += "</tr>";
	}
	board.innerHTML = tableText;
}

function check_jumps(x,y){

	var signs = [ [-1,-1], [-1,1], [1,1], [1,-1] ]

	for(sign in signs)
	{
		vec_1 = [ x+signs[sign][0], y+signs[sign][1] ]
		vec_2 = [ x+(2*signs[sign][0]), y+(2*signs[sign][1]) ]
		
		if(in_boundary(vec_2))
		{
			if(gameboard[vec_1[0]][vec_1[1]]>0 && gameboard[vec_2[0]][vec_2[1]]==0)
			{
				if(signs[sign][0]==-1 || gameboard[x][y]==-2)
					return true
			}	
		}		
	}
	return false
}

function in_boundary(v)
{
	x = v[0]
	y = v[1]
	if(x<0 || x>7 || y<0 || y>7)
		return false
	return true
}

function select(id) {
	var x = Math.floor(id/8);
	var y = id%8;

	// console.log(8)
	// console.log(id)
	// console.log(x, y)
	// console.log(possible_moves)

	if(gameboard[x][y] < 0 && curPlayer == 0) {

		//Clearing previous move
		var ele = 0
		var sign=0
		var signs = [ [-1,-1], [-1,1], [1,1], [1,-1] ]

		for(ele in possible_moves)
		{
			$("#"+possible_moves[ele]).removeClass("selected")
		}

		if(!move_in_progress)
		{
			$("#div"+selected).removeClass("selected")
			selected = id
			$("#div"+id).addClass("selected")
		}
		else
		{
			x = Math.floor(selected/8)
			y = selected%8
		}

		possible_moves = []

		for(sign in signs)
		{
			vec_1 = [ x+signs[sign][0], y+signs[sign][1] ]
			vec_2 = [ x+(2*signs[sign][0]), y+(2*signs[sign][1]) ]

			if(in_boundary(vec_1) && gameboard[vec_1[0]][vec_1[1]]==0 && !jumps)
			{
				if(signs[sign][0] == -1 || gameboard[x][y]==-2)
				{
					var target = (vec_1[0])*8+(vec_1[1])
					possible_moves.push(target)
				}
			}
			else if(in_boundary(vec_2) && gameboard[vec_1[0]][vec_1[1]]>0 && gameboard[vec_2[0]][vec_2[1]]==0)
			{
				if(signs[sign][0] == -1 || gameboard[x][y]==-2)
				{
					if(!jumps)
					{
						possible_moves=[]
						jumps = true
						$("#jumpsPresent").text("true")
					}
					var target = vec_2[0]*8+vec_2[1]
					possible_moves.push(target)
					}
			}			
		}

		for(ele in possible_moves)
		{
			$("#"+possible_moves[ele]).addClass("selected")
		}
	}
	else if(possible_moves.indexOf(id) > -1){
		move_in_progress = true
		$("#moveInProgress").text("true")
		exec_move(selected,id)
		selected = id
		draw()

		if(exec_jump && check_jumps(x,y))
		{
			console.log("jump executed and further jumps possible")
			select(x*8+y)
		}
		else
		{
			$("#currentPlayer").text("Computer")
			curPlayer = 1
			getAIMove()
		}
	}

	// draw();
	// curPlayer = 1;
	// getAIMove();
}

function exec_move(start,end)
{
	exec_jump = false
	start_x = Math.floor(start/8)
	start_y = start%8
	end_x = Math.floor(end/8)
	end_y = end%8
	moving_piece = gameboard[start_x][start_y]
	gameboard[start_x][start_y] = 0
	
	if(moving_piece==1 && end_x==7)
		moving_piece = 2
	else if(moving_piece==-1 && end_x==0)
		moving_piece = -2

	gameboard[end_x][end_y] = moving_piece

	// func = "select("+end+");"
	// var newClick = new Function(func)
	// $("#div"+start).attr("onclick",'').click(newClick)
	// $("#div"+start).attr("id",""+end)

	mov_vec_x = end_x - start_x
	mov_vec_y = end_y - start_y

	if(Math.abs(mov_vec_x) == 2)
	{
		gameboard[start_x+(mov_vec_x/2)][start_y+(mov_vec_y/2)] = 0
		exec_jump = true
	}

	// console.log(gameboard)
}

var gameOb = {"data": gameboard};

function getAIMove() {
	//send an ajax request to /moves
	$.ajax({
		    type: "POST",
		    url: "/moves",
		    // The key needs to match your method's input parameter (case-sensitive).
		    data: JSON.stringify(gameOb),
		    contentType: "application/json; charset=utf-8",
		    dataType: "json",
		    success: function(data){
		    	// alert(data);
		    	var x=0
		    	var coin = ""
		    	var move = ""
		    	data = data['moves']

		    	console.log(data)

		    	for(x=0;x<data.length-1;x++)
		    	{
		    		start = data[x][0]*8+data[x][1]
		    		end = data[x+1][0]*8+data[x+1][1]
		    		console.log(data[x][0])
		    		console.log(data[x][1])
		    		// console.log("coin: "+gameboard[data[x][0]][data[x][1]])
		    		coin = "coin: "+gameboard[data[x][0]][data[x][1]]
		    		move = move+"from: ("+data[x][0]+","+data[x][1]+") to: ("+data[x+1][0]+","+data[x+1][1]+"); "
		    		exec_move(start,end)
		    		console.log(gameboard)
		    	}

		    	console.log(coin)
		    	console.log(move)
		    	$("#coin").text(coin)
		    	$("#compMove").text(move)

		    	curPlayer=0
		    	$("#currentPlayer").text("Player")
		    	//update board also
		    	move_in_progress = false
		    	$("#moveInProgress").text("false")
		    	jumps = false
		    	$("#jumpsPresent").text("false")

		    	exec_jump = false
		    	selected = -1
		    	possible_moves = []
		    	draw()
		    },
		    failure: function(errMsg) {
		        alert(errMsg);
		    }
	});

	//get Response
	//change gameboard
	// curPlayer = 0;
	// draw();
}

window.onload = function () {
	draw();

}