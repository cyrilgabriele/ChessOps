import streamlit as st
import chess


def set_up_board():
    """Initialize or reset the chess board."""
    st.session_state.board = chess.Board()


def make_move(move: str):
    """Attempt to make a move on the board."""
    try:
        st.session_state.board.push_san(move)
        return True
    except ValueError:
        return False


# Set up the page
st.title("Chess Game")
if "board" not in st.session_state:
    set_up_board()

# Display the board using unicode characters
st.write("Current Board:")
st.text(str(st.session_state.board))

# Input for moves
move = st.text_input("Enter your move (e.g., e2e4):", "")

if st.button("Make Move"):
    if not make_move(move):
        st.error("Invalid move, try again.")

# Display game status
outcome = st.session_state.board.outcome()
if st.session_state.board.is_check():
    st.warning("Check!")
if outcome:
    if outcome.winner is not None:
        winner = "White" if outcome.winner else "Black"
        st.success(f"Checkmate! {winner} wins.")
    else:
        st.info("Stalemate or draw.")

# Button to start a new game
if st.button("Start New Game"):
    set_up_board()
