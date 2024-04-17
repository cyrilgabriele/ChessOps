import streamlit as st
import datetime as dt


# Add a block of HTML code to the app
def apply_css():
    """
    Apply CSS styling to the app.
    """
    st.markdown(
        '<link rel="stylesheet" href="streamlit\style.css">', unsafe_allow_html=True
    )


def set_page(title="Chess", page_icon="♟️"):
    st.set_page_config(
        page_title=title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/dakotalock/QuantumBlue",
            "Report a bug": "https://github.com/dakotalock/QuantumBlue",
            "About": "# Streamlit chessboard",
        },
    )

    apply_css()

    st.title("Streamlit Chessboard")


"""Manages session states variables.
"""


def init_states():
    if "next" not in st.session_state:
        st.session_state.next = 0

    if ("curfen" not in st.session_state) or ("moves" not in st.session_state):
        st.session_state.curside = "white"
        st.session_state.curfen = (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        st.session_state.moves = {
            st.session_state.curfen: {
                "side": "GAME START",
                "curfen": st.session_state.curfen,
                "last_fen": "",
                "last_move": "",
                "data": None,
                "timestamp": str(dt.datetime.now()),
            }
        }

    # Get the info from current board after the user made the move.
    # The data will return the move, fen and the pgn.
    # The move contains the from sq, to square, and others.
