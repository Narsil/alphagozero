#!/usr/bin/env python
import sys
from conf import conf
from play import game_init
from engine import ModelEngine, COLOR_TO_PLAYER
from model import load_best_model
import string
import os
from __init__ import __version__

SIZE = conf['SIZE']

class Engine(object):
    def __init__(self, model, logfile):
        self.board, self.player = game_init()
        self.start_engine(model)
        self.logfile = logfile

    def start_engine(self, model):
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
        size = int(size)
        if size != SIZE:
            raise Exception("The board size in configuration is {0}x{0} but GTP asked to play {1}x{1}".format(SIZE, size))
        return ""

    def komi(self, komi):
        # Don't check komi in GTP engine. The algorithm has learned with a specific
        # komi that we don't have any way to influence after learning.
        return ""

    def parse_move(self, move):

        if move.lower() == 'pass':
            x, y = 0, SIZE
            return x, y
        else:
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
        self.board, self.player = self.engine.play(color, x, y)
        return ""

    def genmove(self, color):
        announced_player = COLOR_TO_PLAYER[color]
        assert announced_player == self.player

        x, y, policy_target, value, self.board, self.player, policy = self.engine.genmove(color)
        self.player = self.board[0, 0, 0, -1]  # engine updates self.board already 
        with open(self.logfile, 'a') as f:
            f.write("PLAYER" + str(self.player) + '\n')
        move_string = self.print_move(x, y)
        result = move_string
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
        if not result.strip():
            return "=\n\n"
        return "= " + result + "\n\n"

def main():
    model = load_best_model()
    gtp_log = os.path.join(conf['ROOT_DIR'], conf['LOG_DIR'], conf['GTP_LOGFILE'])
    engine = Engine(model, gtp_log)
    with open(gtp_log, 'w') as f:
        for line in sys.stdin:
            f.write("<<<" + line)
            result = engine.parse_command(line)

            if result.strip():
                sys.stdout.write(result)
                sys.stdout.flush()
                f.write("'''" + str(engine.player) + '\n')
                f.write(">>>" + result)
                f.flush()


if __name__ == "__main__":
    main()
