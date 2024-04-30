# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None


# +
upstream = ["create_tokens"]

# +
def remove_lines_with_too_many_tokens(input_file_path, output_file_path, token_limit=768):
    with open(input_file_path, "r") as file:
        lines = file.readlines()

    print(f"Number of lines in {input_file_path}: {len(lines)}")
    lines_to_keep = []
    removed_count = 0

    for line in lines:
        if len(line.split()) <= token_limit:
            lines_to_keep.append(line)
        else:
            removed_count += 1

    print(f"Number of lines in {output_file_path}: {len(lines_to_keep)}")
    with open(output_file_path, "w") as file:
        file.writelines(lines_to_keep)

    return removed_count

# +
def remove(): 
    input_file_path = "/Users/cyrilgabriele/Documents/School/00_Courses/03_MLOPS/04_Project/ChessOps/data/tokens/carlsen.tok"
    output_file_path = "/Users/cyrilgabriele/Documents/School/00_Courses/03_MLOPS/04_Project/ChessOps/data/tokens/carlsen_max_768.tok"
    removed_lines = remove_lines_with_too_many_tokens(input_file_path, output_file_path)
    print(f"Number of removed lines: {removed_lines}")


# -

remove()
