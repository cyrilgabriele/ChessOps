import json

import torch
import src.notation_converter as converter
from src.generate_prediction import generate_batch_predictions
from src.generate_prediction import generate_beam
from src.chess_game import ChessGame


def load_data(file_path: str) -> list[dict[int, str, list[str]]]:
    """
    Load the JSON file containing the chess positions and legal moves.

    Parameters:
    file_path (str): Path to the JSON file.

    Returns:
    list: A list of dictionaries, each containing a position ID, the position itself, and its legal moves.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def generate_predictions(
    model: torch.nn.Module, positions: list[str], tokens_path: str, tokens_per_ply: int
) -> list[str]:
    """
    Generate predictions for a list of positions.

    Parameters:
    model (Model): The chess model to be evaluated.
    positions (list): A list of positions.
    tokens_path (str): Path to the JSON file containing the tokens.
    tokens_per_ply (int): Number of tokens to generate per ply.

    Returns:
    list: A list of the new predicted moves for each position.
    """
    predicted_sequences, _, _ = generate_batch_predictions(
        positions,
        num_tokens_to_generate=tokens_per_ply,
        model=model,
        token_path=tokens_path,
        max_batch_size=512,
        temperature=0.001,  # get the most likely move
    )

    predicted_moves = [output.split(" ")[-1] for output in predicted_sequences]
    return predicted_moves


def generate_predictions_beam_search(
    model: torch.nn.Module, position: str, tokens_path: str
) -> list[str]:
    """
    Generate predictions for a sequence of moves using beam search.

    Parameters:
    model (Model): The chess model to be evaluated.
    position (str): The position to generate predictions for.
    tokens_path (str): Path to the JSON file containing the tokens.

    Returns:
    list: A list of the new predicted moves for each position. Lower index means higher probability.
    """
    beam_predictions = generate_beam(
        position,
        model,
        tokens_path,
        num_tokens_to_generate=1,
        beam_size=27,
    )
    predicted_moves = [output[0][0].split(" ")[-1] for output in beam_predictions]
    return predicted_moves


def predict_moves_for_all_positions(
    model: torch.nn.Module, tokens_path: str, positions: list[str]
) -> list[str]:
    """
    Predict the beam for the next token for all positions.

    Parameters:
    model (Model): The chess model to be evaluated.
    tokens_path (str): Path to the JSON file containing the tokens.
    positions (list): A list of positions.

    Returns:
    list: A list of the new predicted moves for each position. Lower index means higher probability.
    """
    if torch.cuda.is_available():
        model.cuda()
    predicted_moves = []
    for position in positions:
        predicted_moves.append(
            generate_predictions_beam_search(model, position, tokens_path)
        )
    return predicted_moves


def is_move_legal(legal_moves: list[str], move: str) -> bool:
    """
    Check if the move made by the model is in the list of legal moves.

    Parameters:
    legal_moves (list): A list of legal moves for a position.
    move (str): The move suggested by the model.

    Returns:
    bool: True if the move is legal, False otherwise.
    """
    return move in legal_moves


def calculate_accuracy(total_positions: int, correct_predictions: int) -> float:
    """
    Calculate the accuracy of the model.

    Parameters:
    total_positions (int): Total number of positions evaluated.
    correct_predictions (int): Number of correct predictions made by the model.

    Returns:
    float: The accuracy of the model as a float between 0 and 1.
    """
    return correct_predictions / total_positions if total_positions > 0 else 0


def evaluate_hard_positions(
    json_path: str,
    model: torch.nn.Module,
    tokens_path: str = "./src/tokenizer/xlan_tokens.json",
    tokens_per_ply: int = 3,
) -> tuple[float, list[tuple[int, str, bool]]]:
    """
    Evaluate a chess model using a set of positions and their legal moves.

    Parameters:
    json_path (str): Path to the JSON file containing positions and legal moves.
    model (Model): The chess model to be evaluated.
    tokens_path (str): Path to the JSON file containing the tokens.
    tokens_per_ply (int): Number of tokens to generate per ply.

    Returns:
    tuple: A tuple containing the model's accuracy and a list of tuples with position ID, predicted move, and correctness.
    """
    data = load_data(json_path)
    board_states = [board_state["board_state"] for board_state in data]
    predicted_moves = generate_predictions(
        model, board_states, tokens_path, tokens_per_ply
    )

    correct_predictions = 0
    results = []

    for idx, board_state in enumerate(data):
        predicted_correct = is_move_legal(
            board_state["legal_positions"], predicted_moves[idx]
        )
        results.append((board_state["id"], predicted_moves[idx], predicted_correct))
        if predicted_correct:
            correct_predictions += 1

    accuracy = calculate_accuracy(len(data), correct_predictions)
    return accuracy, results


def evaluate_legal_piece_moves(
    json_path: str,
    model: torch.nn.Module,
    tokens_path: str = "./src/tokenizer/xlan_tokens.json",
) -> tuple[float, list[tuple[int, list[str], list[str], bool, str, str]]]:
    """
    Evaluate a chess model using a set of positions with the next piece to move included.
    E.g. "Pd2d4 P" would indicate that the next piece for black to move is a Pawn. It should predict now
    all legal start positions for a pawn. For that we beam search the model for the next token and
    compare it to the legal moves in json file.

    Parameters:
    json_path (str): Path to the JSON file containing positions and legal positions.
    model (Model): The chess model to be evaluated.
    tokens_path (str): Path to the JSON file containing the tokens.

    Returns:
        tuple: A tuple containing the model's accuracy and a list of tuples with position ID, predicted moves, correct moves, correctness, piece and tag.
    """
    data = load_data(json_path)
    positions = [position["board_state"] for position in data]
    predicted_moves_per_position = predict_moves_for_all_positions(
        model, tokens_path, positions
    )

    correct_predictions = 0
    results = []
    for idx, position in enumerate(data):
        number_of_legal_moves = len(position["legal_positions"])
        all_predicted_correct = True
        for i in range(number_of_legal_moves):
            predicted_correct = is_move_legal(
                position["legal_positions"], predicted_moves_per_position[idx][i]
            )
            all_predicted_correct = all_predicted_correct and predicted_correct

        results.append(
            (
                position["id"],
                predicted_moves_per_position[idx],
                position["legal_positions"],
                all_predicted_correct,
                position["piece"],
                position["tag"],
            )
        )
        if all_predicted_correct:
            correct_predictions += 1

    accuracy = calculate_accuracy(len(data), correct_predictions)
    return accuracy, results


def get_position_by_id(
    position_id: int,
    json_path: str = "./data/validation/hard_positions/hard_positions.json",
) -> str:
    """
    Retrieves a chess position by its ID from a JSON file.

    Parameters:
    position_id (int): The ID of the position to retrieve.
    json_path (str): Path to the JSON file containing positions and legal moves.

    Returns:
    str: The chess position in unique move notation, or None if not found.
    """
    with open(json_path, "r") as file:
        data = json.load(file)

    for position in data:
        if position["id"] == position_id:
            return position["board_state"]

    return None  # Return None if the position with the given ID is not found


def get_data_by_id(
    position_id: int,
    json_path: str = "./data/validation/hard_positions/hard_positions.json",
) -> tuple[str, list[str], str, str]:
    """
    Retrieves board_state, legal_positions, piece and tag by its ID from a JSON file.

    Parameters:
    position_id (int): The ID of the position to retrieve.
    json_path (str): Path to the JSON file containing positions and legal moves.

    Returns:
    A tuple containing board_state, legal_positions, piece and tag
    """
    with open(json_path, "r") as file:
        data = json.load(file)

    for position in data:
        if position["id"] == position_id:
            return (
                position["board_state"],
                position["legal_positions"],
                position["piece"],
                position["tag"],
            )

    return None  # Return None if the position with the given ID is not found


def show_position_by_id(
    position_id: int,
    json_path: str = "./data/validation/hard_positions/hard_positions.json",
    save_to_file=False,
) -> None:
    """
    Shows a chess position by its ID from a JSON file. Automatically converts xLAN+ to xLAN.

    Parameters:
    position_id (int): The ID of the position to retrieve.
    json_path (str): Path to the JSON file containing positions and legal moves.
    """
    position = get_position_by_id(position_id, json_path)
    if position.find("-") != -1:
        position = converter.xlanplus_sequence_to_xlan(position)
    game = ChessGame(
        player1_type="player",
        player2_type="player",
        model_p1=None,
        model_p2=None,
        token_path="",
    )
    game.show_position_from_sequence(position, save_to_file)
