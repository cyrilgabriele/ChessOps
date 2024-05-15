from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from src.generate_prediction import generate_prediction
import torch

class Sequences(BaseModel):
    fen: str
    history: str 

app = FastAPI()


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
    input_string = f"{sequences.fen} {sequences.history}"
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
    print(f"detokenized_output: {detokenized_output}")
    # The output_text should contain the predicted move
    return {"move": detokenized_output}
