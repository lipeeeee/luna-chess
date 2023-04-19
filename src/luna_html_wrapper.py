"""
    An Html wrapper for Luna
"""

from flask import Flask, Response, Request
import chess.svg
import base64
import luna
app = Flask(__name__)

class LunaHtmlWrapper(luna.Luna):
    """Html GUI for Luna-Chess using flask"""
    """Luna gives us:
        1. luNNa
        2. luna_eval
        3. board
    """

    def __init__(self, verbose=False) -> None:
        super().__init__(verbose)
        
    @app.route("/")
    def index():
        """Index page"""
        #html = open("index.html").read()
        #return html.replace('start')
        return "TEST"

    def board_to_svg(self):
        """Parse board into svg so we can use it in UI"""
        return base64.b64encode(chess.svg.board(board=self.board).encode('utf-8')).decode('utf-8')


if __name__ == "__main__":
    l = LunaHtmlWrapper(True)
    app.run()