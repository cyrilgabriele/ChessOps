# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None

# -

upstream = []
def pgn_to_xlan():
    from src.data_preprocessing.pgn_to_xlan import pgn_to_xlan

    # TODO change to valid paths
    pgn_path = "/Users/cyrilgabriele/Documents/School/00_Courses/03_MLOPS/04_Project/ChessOps/data/pgn/Carlsen.pgn"
    lan_path = "/Users/cyrilgabriele/Documents/School/00_Courses/03_MLOPS/04_Project/ChessOps/data/xlan/carlsen.xlanplus"

    min_number_of_moves_per_game = 0
    number_of_games_to_write = -1  # -1 for all games

    pgn_to_lan = pgn_to_xlan(
        pgn_path,
        lan_path,
        min_number_of_moves_per_game=min_number_of_moves_per_game,
        number_of_games_to_write=number_of_games_to_write,
        generate_all_moves=False,
        log=False,
        xLanPlus=True,
    )

    pgn_to_lan.convert_pgn_parallel()


pgn_to_xlan()

def check_duplicates():
    from src.data_preprocessing.check_duplicates_and_common_lines import (
        check_duplicates_and_common_lines,
    )   
    
    # TODO what is the validation file?
    training_file = "/Users/cyrilgabriele/Documents/School/00_Courses/03_MLOPS/04_Project/ChessOps/data/xlan/carlsen.xlanplus"
    validation_file = (
        "/Users/cyrilgabriele/Documents/School/00_Courses/03_MLOPS/04_Project/ChessOps/data/xlan/carlsen.xlanplus"
    )

    check_duplicates_and_common_lines(training_file, validation_file, delete_common=False, delete_duplicates_from_file_1=True, delete_duplicates_from_file_2=False)


check_duplicates()
