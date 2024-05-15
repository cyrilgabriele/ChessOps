# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None

# -


# open text file
def add_bos(text_file_in, text_file_out):
    with open(text_file_in, "r") as file:
        lines = file.readlines()

    # add <BOS> to each line
    lines_with_bos = ["75 " + line for line in lines]

    # write to the same file
    with open(text_file_out, "w") as file:
        file.writelines(lines_with_bos)


def clean_up_file(text_file_out):
    # replace "  " with " "
    with open(text_file_out, "r") as file:
        lines = file.readlines()
        lines = [line.replace("  ", " ") for line in lines]

    # write to the same file
    with open(text_file_out, "w") as file:
        file.writelines(lines)


add_bos(text_file_in, text_file_out)
clean_up_file(text_file_out)
