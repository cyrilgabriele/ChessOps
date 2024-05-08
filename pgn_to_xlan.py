# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None

# -

upstream = []


def pgn_to_xlan(pgn_path, xlan_path):
    from src.data_preprocessing.pgn_to_xlan import pgn_to_xlan

    min_number_of_moves_per_game = 0
    number_of_games_to_write = -1  # -1 for all games

    pgn_to_lan = pgn_to_xlan(
        pgn_path,
        xlan_path,
        min_number_of_moves_per_game=min_number_of_moves_per_game,
        number_of_games_to_write=number_of_games_to_write,
        generate_all_moves=False,
        log=False,
        xLanPlus=True,
    )

    pgn_to_lan.convert_pgn_parallel()


def remove_duplicates(file_path):
    from collections import OrderedDict

    with open(file_path, "r") as f:
        lines = OrderedDict((line, None) for line in f)

    with open(file_path, "w") as f:
        f.writelines(lines.keys())


pgn_to_xlan(pgn_path, lan_path)
remove_duplicates(training_file)
