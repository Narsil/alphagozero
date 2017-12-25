#!/usr/bin/env python
import sys
from conf import conf
from play import game_init, make_play, show_board
from engine import ModelEngine
from model import load_best_model
import string
from __init__ import __version__

COLOR_TO_PLAYER = {'B': 1, 'W': -1}
SIZE = conf['SIZE']

class Engine(object):
    def __init__(self):
        self.board, self.player = game_init()
        self.start_engine()

    def start_engine(self):
        model = load_best_model()
        self.engine = ModelEngine(model, conf['MCTS_SIMULATIONS'], self.board)

    def name(self):
        return "AlphaGoZero Python - {} - {} simulations".format(self.engine.model.name, conf['MCTS_SIMULATIONS']) 

    def version(self):
        return __version__

    def protocol_version(self):
        return "2"

    def list_commands(self):
        return ""

    def boardsize(self, size):
        return ""

    def parse_move(self, move):
        letter = move[0]
        number = move[1:]

        x = string.ascii_uppercase.index(letter)
        if x >= 9:
            x -= 1 # I is a skipped letter
        y = int(number) - 1
        x, y = x, SIZE - y - 1
        return x, y

    def print_move(self, x, y):
        x, y = x, SIZE - y - 1

        if x >= 8:
            x += 1 # I is a skipped letter

        move = string.ascii_uppercase[x] + str(y + 1)
        return move

    def play(self, color, move):
        announced_player = COLOR_TO_PLAYER[color]
        assert announced_player == self.player
        x, y = self.parse_move(move)
        self.engine.play(color, x, y)
        self.board, self.player = make_play(x, y, self.board)
        return ""

    def genmove(self, color):
        announced_player = COLOR_TO_PLAYER[color]
        assert announced_player == self.player

        x, y, policy_target, value = self.engine.genmove(color)
        self.board, self.player = make_play(x, y, self.board)
        move_string = self.print_move(x, y)
        result = move_string + "\n\n" + show_board(self.board)
        return result

    def clear_board(self):
        self.board, self.player = game_init()
        return ""

    def parse_command(self, line):
        tokens = line.strip().split(" ")
        command = tokens[0]
        args = tokens[1:]
        method = getattr(self, command)
        result = method(*args)
        return "= " + result + "\n\n"

def main():
    engine = Engine()
    with open('logs/gtp.log', 'w') as f:
        for line in sys.stdin:
            f.write("<<<" + line)
            result = engine.parse_command(line)

            if result.strip():
                sys.stdout.write(result)
                sys.stdout.flush()
                f.write(">>>" + result)
                f.flush()


if __name__ == "__main__":
    main()
