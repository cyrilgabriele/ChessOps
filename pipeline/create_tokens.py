# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None


# +
upstream = ["checkDuplicates"]


# -


def tokenize_data(xLAN_path, token_path, tokenized_path):
    from src.tokenizer.tokenizer import tokenize_file

    tokenize_file(
        token_path, xLAN_path, tokenized_path, batch_size=20000
    )  # eventually smaller batch size


tokenize_data(xLAN_path, token_path, tokenized_path)
