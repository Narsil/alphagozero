from conf import conf
from sgfmill import sgf
import os
from collections import defaultdict
from math import sqrt


GAMES_DIR = conf['GAMES_DIR']

def analysis():
    directory = GAMES_DIR
    results = defaultdict(list)
    i = 0
    for model_directory in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, model_directory)):
            if filename.endswith('.sgf'): # Ignore sgf files.
                with open(os.path.join(directory, model_directory, filename), 'rb') as f:
                    game = sgf.Sgf_game.from_bytes(f.read())
                    i += 1

                    playerb = game.get_player_name('b')
                    playerw = game.get_player_name('w')
                    winner_is_black = game.get_winner() == 'b'
                    results[(playerb, playerw)].append( winner_is_black)
    return results


if __name__ == "__main__":
    results = analysis()
    total = 0
    n = 0
    for key, l in results.items():
        mean = sum(l) /  float(len(l))
        delta = 2./sqrt(len(l))
        interval = [mean - delta, mean + delta]

        n += len(l)
        total += sum(l)

    m = total / n
    delta = 1/sqrt(n)
    print("Average black winrate [", m - delta, ",", m + delta, "]")


