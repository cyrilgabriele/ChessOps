from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from src.generate_prediction import generate_prediction
import torch
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM


# Initialize model
# TODO update according to selected player!! 
# => and of course do NOT use the base model, instead the finetuned!
model_name = "Leon-LLM/Leon-Chess-350k-Plus"
model = AutoModelForCausalLM.from_pretrained(model_name)



class Sequences(BaseModel):
    fen: str
    history: str 

app = FastAPI()


# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity, but you can restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/get_move")
async def get_move(sequences: Sequences):
    # TODO: return here the new move (most likely in UCI?)
    # TODO: sequences => generatePrediction.py => use only first element of return 
    token_path = "../src/tokenizer/xlanplus_tokens.json"# input_string = f"{sequences.fen} {sequences.history}"
    # Example history: "1. d3 c6 2. d4 d6 3. d5 c5"
    # You need to convert this to "Pd2d3 Pc7c6 Pd2d4 Pd7d6 Pd4d5 Pc6c5"
    
    history = sequences.history.strip()
    moves = history.split()
    
    # Reconstruct input string in the format "Pd2d4 Pe7e5"
    formatted_moves = []
    for move in moves:
        if '.' not in move:
            formatted_moves.append(f"P{move[:2]}{move[2:]}")

    input_string = ' '.join(formatted_moves)
    
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
        seed=seed
    )

    # Debugging output
    print(f"Input string: {input_string}")
    print(f"Tokenized string: {tokenized_string}")
    print(f"Predicted token string: {predicted_token_string}")
    print(f"Detokenized output: {detokenized_output}")
    print(f"Type of detokenized output: {type(detokenized_output)}")

    # Remove unwanted tokens from the detokenized output
    valid_tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "1", "2", "3", "4", "5", "6", "7", "8", "P", "R", "N", "B", "Q", "K"]
    move = "".join([char for char in detokenized_output.strip().split()[-1] if char in valid_tokens])
    
    print(f"Predicted move: {move}")

    return {"move": move}
    # return {"move": sequences.history}
