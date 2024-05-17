import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import html
import chess
import streamlit_scrollable_textbox as stx
from st_bridge import bridge
from mychess import MyChess
from utility import set_page, init_states
from streamlit_js_eval import streamlit_js_eval

import datetime as dt

set_page(title="Chess", page_icon="♟️")
init_states()
st.session_state.board_width = 400

# Dropdown to select model
model_options = ["G. Kasparov", "M. Carlsen"]
selected_model = st.selectbox("Select Model for Prediction", model_options)
st.session_state.selected_model = selected_model

# Get the info from current board after the user made the move.
# The data will return the move, fen and the pgn.
# The move contains the from sq, to square, and others.
data = bridge("my-bridge")

if data is not None:
    st.session_state.lastfen = st.session_state.curfen
    st.session_state.curfen = data["fen"]
    st.session_state.curside = (
        data["move"]["color"].replace("w", "white").replace("b", "black")
    )
    st.session_state.moves.update(
        {
            st.session_state.curfen: {
                "side": st.session_state.curside,
                "curfen": st.session_state.curfen,
                "last_fen": st.session_state.lastfen,
                "last_move": data["pgn"],
                "data": None,
                "timestamp": str(dt.datetime.now()),
            }
        }
    )

if st.button("start new game"):
    streamlit_js_eval(js_expressions="parent.window.location.reload()")


cols = st.columns([1, 1])

with cols[0]:
    game = MyChess(
        st.session_state.board_width,
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    )

    components.html(
        game.game_board(), height=st.session_state.board_width + 75, scrolling=False
    )
    board = chess.Board(st.session_state.curfen)
    outcome = board.outcome()
    st.warning(f"is_check: {board.is_check()}")
    st.warning(f"is_checkmate: {board.is_checkmate()}")
    st.warning(f"is_stalemate: {board.is_stalemate()}")
    st.warning(f"is_insufficient_material: {board.is_insufficient_material()}")
    if outcome:
        st.warning(f"Winner: { {True:'White',False:'Black'}.get(outcome.winner) }")

with cols[1]:
    with st.container():
        records = [
            f"##### {value['timestamp'].split('.')[0]} \n {value['side']} - {value.get('last_move','')}"
            for key, value in st.session_state["moves"].items()
        ]
        # html( "<p>" + '\n\n'.join(records) + "</p>", scrolling=True)
        stx.scrollableTextbox("\n\n".join(records), height=500, border=True)


# Update the fetch request to include the selected model
def get_prediction(selected_model, fen, history):
    import requests

    url = "http://localhost:8000/get_move"
    payload = {"fen": fen, "history": history, "model": selected_model}

    try:
        response = requests.post(url, json=payload)
        print("======================================================")
        print(response)
        print("======================================================")

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return {"error": str(e)}


if data:
    response = get_prediction(st.session_state.selected_model, st.session_state.curfen, data["pgn"])
    print(f"response: {response['move']}")
    st.session_state.predicted_move = response["move"]
    st.info(f"Predicted move: {st.session_state.predicted_move}")
