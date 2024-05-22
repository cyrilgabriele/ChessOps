# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None


# +
upstream = ["checkDuplicates"]


# -

import os


def tokenize_data(xLAN_path, token_path, tokenized_path):
    from src.tokenizer.tokenizer import tokenize_file

    # if file is already tokenized, remove all content
    if os.path.exists(tokenized_path):
        os.remove(tokenized_path)

    tokenize_file(
        token_path, xLAN_path, tokenized_path, multiprocessing=False, batch_size=20000
    )  # eventually smaller batch size


tokenize_data(xLAN_path, token_path, tokenized_path)
