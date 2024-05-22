from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM
import chess
import chess.engine
import chess.pgn
import io
import platform

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


os_name = platform.system()

if os_name == "Windows":
    engine_path = "./backend/stockfish/stockfish-windows-x86-64-avx2.exe"
elif os_name == "Darwin":
    engine_path = "./backend/stockfish/stockfish-macOS"
else:
    print(f"You are using another operating system: {os_name}")
    print(
        f"Please download the Stockfish binary for {os_name} from https://stockfishchess.org/download/"
    )

engine = chess.engine.SimpleEngine.popen_uci(engine_path)


@app.post("/get_move")
async def get_move(sequences: Sequences):
    print("Sequence:", sequences)
    if sequences.model == "stockfish":
        return await get_stockfish_move(sequences.fen)
    else:
        return await get_player_move(sequences)


async def get_player_move(sequences):
    model = models.get(sequences.model)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        board = chess.Board(sequences.fen)
        input_string = process_game_history(sequences.history, sequences.fen)
        prediction = generate_move(input_string, model)
        last_move_uci = process_prediction(prediction, board)
        print("GPT move:", last_move_uci)
        return {"move": last_move_uci}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_stockfish_move(fen):
    board = chess.Board(fen)
    with engine.analysis(board, multipv=1) as analysis:
        for info in analysis:
            if info.get("pv"):
                move = info["pv"][0]
                break
    return {"move": move.uci()}


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
