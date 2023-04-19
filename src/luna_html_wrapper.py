"""
    An Html wrapper for Luna
"""

from flask import Flask, Response, request
import chess.svg
import base64
import luna
import traceback
import os
app = Flask(__name__)

class LunaHtmlWrapper(luna.Luna):
    """Html GUI for Luna-Chess using flask"""
    """Luna gives us:
        1. luNNa
        2. luna_eval
        3. board
    """

    def __init__(self, verbose=False) -> None:
        # Init Luna
        super().__init__(verbose)

    def board_to_svg(self, s:luna.LunaState):
        """Parse board into svg so we can use it in UI"""
        return base64.b64encode(chess.svg.board(board=s.board).encode('utf-8')).decode('utf-8')

htmlWrap = LunaHtmlWrapper(verbose=True) 

@app.route("/")
def index():
    """Index page"""
    html = open("src/index.html").read()
    return html.replace('start', htmlWrap.board.fen())

@app.route("/selfplay")
def selfplay():
    """Self play page"""
    # reset state
    htmlWrap.board_state = luna.LunaState()
    
    ret = '<html><head>'
    # self play
    while not htmlWrap.board.is_game_over():
        htmlWrap.computer_move(htmlWrap.board_state, htmlWrap.luna_eval)
        ret += '<img width=600 height=600 src="data:image/svg+xml;base64,%s"></img><br/>' % htmlWrap.board_to_svg(htmlWrap.board_state)
        
    if htmlWrap.verbose: print(f"[SELFPLAY] SELFPLAY OVER, RESULT: {htmlWrap.board.result()}")

    return ret 

# move given in algebraic notation
@app.route("/move")
def move():
    if not htmlWrap.is_game_over():
        move = request.args.get('move',default="")
        if move is not None and move != "":
            if htmlWrap.verbose: print("[HUMAN MOVES] move")
            
            try:
                htmlWrap.board.push_san(move)
                htmlWrap.computer_move(htmlWrap.board_state, htmlWrap.luna_eval)
            except Exception:
                traceback.print_exc()
            response = app.response_class(
                response=htmlWrap.board.fen(),
                status=200
            )
            return response
    else:
        if htmlWrap.verbose: print("[GAME STATE] GAME IS OVER")
        response = app.response_class(
        response="game over",
        status=200
        )
        return response
    
    if htmlWrap.verbose: print("[FUNCTION CALLS] luna_html_wrapper.py.index() ran")
    return index()

# moves given as coordinates of piece moved
@app.route("/move_coordinates")
def move_coordinates():
    if not htmlWrap.is_game_over():
        source = int(request.args.get('from', default=''))
        target = int(request.args.get('to', default=''))
        promotion = True if request.args.get('promotion', default='') == 'true' else False

        move = htmlWrap.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

        if move is not None and move != "":
            if htmlWrap.verbose: print("[HUMAN MOVES] move")
            try:
                htmlWrap.board.push_san(move)
                htmlWrap.computer_move(htmlWrap.board_state, htmlWrap.luna_eval)
            except Exception:
                traceback.print_exc()
            response = app.response_class(
            response=htmlWrap.board.fen(),
            status=200
            )
        return response

    if htmlWrap.verbose: print("[GAME STATE] GAME IS OVER")
    response = app.response_class(
        response="game over",
        status=200
    )
    return response

@app.route("/newgame")
def newgame():
    htmlWrap.board.reset()
    response = app.response_class(
        response=htmlWrap.board.fen(),
        status=200
    )
    return response

if __name__ == "__main__":
    app.run()