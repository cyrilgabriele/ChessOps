from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from src.generate_prediction import generate_prediction
import src.notation_converter as converter
import torch
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM
import os
import regex as re
import chess.pgn
import io


class Sequences(BaseModel):
    fen: str
    history: str
    model: str


app = FastAPI()

# Set up CORS => simply allow everything
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/get_move")
async def get_move(sequences: Sequences):
    print("Received payload:", sequences)

    """
    model_name_map = {
        # TODO: add the correct models! => they are not working yet...
        "G. Kasparov": "larscarl/Leon-Chess-350k-Plus_LoRA_kasparov_5E_0.0001LR",
        "M. Carlsen": "larscarl/Leon-Chess-350k-Plus_LoRA_carlsen_5E_0.0001LR"
    }
    """
    model_name_map = {
        "G. Kasparov": "Leon-LLM/Leon-Chess-1M-BOS",
        "M. Carlsen": "Leon-LLM/Leon-Chess-1M-BOS",
    }

    model_name = model_name_map.get(sequences.model, "Leon-LLM/Leon-Chess-350k-Plus")

    print(f"model_name: {model_name}")

    # Load the selected model
    # TODO: do this outside get_move so it gets loaded during startup
    model = AutoModelForCausalLM.from_pretrained(model_name)

    token_path = "./src/tokenizer/xlanplus_tokens.json"

    if not os.path.exists(token_path):
        return {"error": f"Token file not found: {token_path}"}

    pgn = io.StringIO(sequences.history)
    game = chess.pgn.read_game(pgn)

    uci_moves = []
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        uci_moves.append(board.uci(move))

    uci_sequence = " ".join(uci_moves)
    print(f"uci_moves: {uci_sequence}")
    x_lan_sequence = converter.uci_sequence_to_xlan(uci_sequence)
    print(f"xlan_moves: {x_lan_sequence}")
    moves_in_xlan_plus = converter.xlan_sequence_to_xlanplus(x_lan_sequence)
    print(f"xlan_plus_moves: {moves_in_xlan_plus}")
    input_string = moves_in_xlan_plus

    num_tokens_to_generate = 1  # Number of moves to generate
    temperature = 1.0
    seed = None  # Optional: set a seed for reproducibility if needed

    # Call the generate_prediction function with the correct parameters
    detokenized_output, predicted_token_string, tokenized_string = generate_prediction(
        input_string,
        num_tokens_to_generate=num_tokens_to_generate,
        model=model,
        token_path=token_path,
        temperature=temperature,
        seed=seed,
    )

    print(f"Tokenized string: {tokenized_string}")
    print(f"Predicted token string: {predicted_token_string}")
    print(f"Detokenized output: {detokenized_output}")
    print(f"Type of detokenized output: {type(detokenized_output)}")
    last_move = detokenized_output.split(" ")[-1]

    return {"move": last_move}