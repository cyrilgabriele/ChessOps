# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None

# -

upstream = ["pgn_to_xlan"]

# +
"""
Use check_duplicates_and_common_lines to check if there are duplicates or common lines in two files.
"""


# -

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
