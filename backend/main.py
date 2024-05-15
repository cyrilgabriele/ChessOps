from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel


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
    return sequences 
