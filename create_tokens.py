# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None


# +
upstream = ["checkDuplicates"]


# -

def tokenize_data():
    from src.tokenizer.tokenizer import tokenize_file
    
    xLAN_path = "/Users/cyrilgabriele/Documents/School/00_Courses/03_MLOPS/04_Project/ChessOps/data/xlan/carlsen.xlanplus" # Inout Path 
    token_path = "./src/tokenizer/xlanplus_tokens.json" # keep this, correct like this
    tokenized_path = "/Users/cyrilgabriele/Documents/School/00_Courses/03_MLOPS/04_Project/ChessOps/data/tokens/carlsen.tok" # Output path

    tokenize_file(token_path, xLAN_path, tokenized_path, batch_size=20000) # eventually smaller batch size


tokenize_data()
