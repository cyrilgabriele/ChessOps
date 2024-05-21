from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM
import chess.pgn
import io

from src.generate_prediction import generate_prediction
import src.notation_converter as converter

app = FastAPI()

# Set up CORS to allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configurations
model_name_map = {
    "G. Kasparov": "Leon-LLM/Leon-Chess-350k-Plus",
    "M. Carlsen": "Leon-LLM/Leon-Chess-350k-Plus",
}

adapter_map = {
    "G. Kasparov": "larscarl/Leon-Chess-350k-Plus_LoRA_kasparov_10E_0.0001LR",
    "M. Carlsen": "larscarl/Leon-Chess-350k-Plus_LoRA_carlsen_10E_0.0001LR",
}

# Initialize and load models with adapters if available
models = {}
base_model_id = "Leon-LLM/Leon-Chess-350k-Plus"
models["base"] = AutoModelForCausalLM.from_pretrained(base_model_id)

for name, model_path in model_name_map.items():
    models[name] = AutoModelForCausalLM.from_pretrained(model_path)
    if name in adapter_map:
        models[name].load_adapter(adapter_map[name])


class Sequences(BaseModel):
    fen: str
    history: str
    model: str


@app.post("/get_move")
async def get_move(sequences: Sequences):
    model = models.get(sequences.model)
    print("Received payload:", sequences)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        input_string = process_game_history(sequences.history, sequences.fen)
        board = chess.Board(sequences.fen)
        prediction = generate_move(input_string, model)
        last_move_uci = process_prediction(prediction, board)
        return {"move": last_move_uci}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def process_game_history(history, fen):
    if len(history) > 1:
        game = chess.pgn.read_game(io.StringIO(history))
        board = game.board()
        uci_moves = [board.uci(move) for move in game.mainline_moves()]
        x_lan_sequence = converter.uci_sequence_to_xlan(" ".join(uci_moves))
        return converter.xlan_sequence_to_xlanplus(x_lan_sequence)
    else:
        return chess.Board(fen), ["75"]


def generate_move(input_string, model):
    return generate_prediction(
        input_string,
        num_tokens_to_generate=3,
        model=model,
        token_path="./src/tokenizer/xlanplus_tokens.json",
        temperature=0.01,
    )[0]


def process_prediction(prediction, board):
    last_move = prediction.split(" ")[-1]
    return "".join(converter.xlanplus_move_to_uci(board, last_move)[0])
