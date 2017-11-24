from sgfmill import sgf
from conf import conf
from play import get_real_board
import os

SIZE = conf['SIZE']


def save_game_sgf(model_name, game_n, game_data):
    game = sgf.Sgf_game(size=SIZE)

    modelB_name = game_data['modelB_name']
    modelW_name = game_data['modelW_name']
    winner = game_data['winner']

    game.root.set_raw("PB", modelB_name.encode('utf-8'))
    game.root.set_raw("PW", modelW_name.encode('utf-8'))
    game.root.set_raw("KM", str(conf['KOMI']).encode('utf-8'))

    winner_b = b"B" if winner == 1 else b"W"
    game.root.set_raw("RE", winner_b)
    for move_data in game_data['moves']:
        node = game.extend_main_sequence()
        color = 'b' if move_data['player'] == 1 else 'w'
        x, y = move_data['move']

        move = (SIZE - 1 - y, x) if y != SIZE else None  # Different orienation

        move_n = move_data['move_n']
        next_board = game_data['moves'][(move_n + 1) % len(game_data['moves'])]['board']
        comment = "Value %s\n %s" % (move_data['value'], get_real_board(next_board))
        node.set("C", comment)
        node.set_move(color, move)


    try:
        os.makedirs(os.path.join(conf['GAMES_DIR'], model_name))
    except OSError:
        pass

    filename = os.path.join(conf["GAMES_DIR"], model_name, "game_%03d.sgf" % game_n)
    while os.path.isfile(filename):
        game_n += 1
        filename = os.path.join(conf["GAMES_DIR"], model_name, "game_%03d.sgf" % game_n)

    with open(filename, "wb") as f:
        f.write(game.serialise())
