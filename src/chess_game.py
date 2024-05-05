"""
Chess Game with AI Integration
------------------------------

This class encapsulates a chess game, providing functionalities to play the game between two entities, 
which could be human players, AI models, or a combination of both. The class uses the `pythonChess` library 
to handle game logic and the IPython `display` function to visualize the chess board in Jupyter notebooks.

The game supports input from players in xLAN format and can also utilize an AI model to generate 
moves. This AI integration requires a separate model that can be trained using the train.ipynb notebook or
downloaded from https://huggingface.co/Leon-LLM.

xLAN is a unique move notation:
- The first letter represents the piece moved, e.g., P for pawn, N for knight, etc.
- The next two letters represent the start square, e.g., e2, e4, etc.
- The next two letters represent the end square, e.g., e2, e4, etc.
    -Pawn promotion is represented by the piece to promote to, e.g., Q for queen and the start and end square, e.g., Qe7e8.
    -Castling is represented by the start and end square of the king, e.g., Ke1g1.

Attributes:
- player1_type (str): The type of player 1, which can be "player" for a human or "model" for an AI.
- player2_type (str): The type of player 2, similar to player 1.
- model_p1 (torch.nn.Module): The AI model instance for player 1 if player1_type is "model". If player1 is "player" use None.
- model_p2 (torch.nn.Module): The AI model instance for player 2 if player2_type is "model". If player1 is "player" use None.
- token_path (str): The path to the authentication token for tokenizer/detokenizer.
- max_model_tries (int): The maximum number of attempts the AI model should make to produce a valid move.
- temperature (float): A parameter for the AI model that may affect move diversity and unpredictability.
- starting_sequence (str): A string of moves to start the game in xLAN with, e.g., "Pe2e4 Pe7e5 Ng1f3".

Example usage:
To initiate a game with two human players:

    >> from chess_game import ChessGame
    >> game = ChessGame(player1_type='player', player2_type='player', model_p1=None, model_p2=None, token_path='')
    >> game.play_game()

For a game between a human and an AI:

    >> game =    ChessGame(player1_type='player',
                 player2_type='model',
                 model_p1=None, 
                 model_p2=ai_model, 
                 token_path='path/to/token',
                 max_model_tries=10,
                 temperature=0.3)
    >> game.play_game()
"""

import chess
import src.notation_converter as converter
from IPython.display import display, clear_output
from ipywidgets import Output
from src.generate_prediction import generate_prediction
from IPython.display import SVG
import chess.svg


class ChessGame:
    def __init__(
        self,
        player1_type,
        player2_type,
        model_p1,
        model_p2,
        token_path,
        max_model_tries=5,
        temperature=0.5,
        starting_sequence="",
        show_game_history=True,
    ):
        self.player1_type = player1_type
        self.player2_type = player2_type
        self.model_p1 = model_p1
        self.model_p2 = model_p2
        self.token_path = token_path
        self.max_model_tries = max_model_tries
        self.temperature = temperature
        self.movehistory = starting_sequence
        self.show_game_history = show_game_history

        self.TOKENS_TO_PREDICT = 3  # Number of tokens to predict: 3 Tokens = 1 Move

        self.board = chess.Board()
        self.display_output = Output()  # Output widget to display the board
        self.ply = 1

    def display_board(self, size=600, save_to_file=False):
        """
        Displays the current state of the board.
        """
        with self.display_output:
            clear_output(wait=True)
            board_svg = chess.svg.board(board=self.board, size=size)
            # display(self.board)
            display(SVG(board_svg))
            if save_to_file:
                import os

                filename = f"{save_to_file}.svg"

                # Ensure the directory exists
                os.makedirs("./results/plots/saved_boards/", exist_ok=True)
                filepath = os.path.join("./results/plots/saved_boards", filename)

                # Save the SVG data to the file
                with open(filepath, "w") as file:
                    file.write(board_svg)

                print(f"Board saved to {filepath}")

    def get_next_move(self, player_type, model):
        """
        Gets the next move from the player.
        If the player type is 'player', it prompts the user for input.
        If the player type is 'model', it attempts to get a move from the AI model.

        Args:
        player_type (str): The type of the player, either 'player' or 'model'.
        model: The AI model to predict the move if player_type is 'model'.

        Returns:
        str: Next move in xLAN format, e.g., 'Pe2e4'.
        """
        if player_type == "player":
            user_move = self.get_next_move_from_player()
            if self.show_game_history:
                print("Your input: ", user_move)
            return user_move
        elif player_type == "model":
            return self.get_next_move_from_model(model)

    def get_next_move_from_player(self):
        """
        Prompts the user for input and returns the move in xLAN format.

        Returns:
        str: Next move in xLAN format, e.g., 'Pe2e4'.
        """
        user_move = input("Your move (e.g., Pe2e4) 'exit' to terminate program: ")
        if user_move == "exit":
            print("Exiting game...")
            raise ExitGameException("Player requested to exit the game.")
        return user_move

    def get_next_move_from_model(self, model):
        """
        Attempts to get a move from the AI model. If the model fails to produce a valid move, it prompts the user
        for input.

        Args:
        model (torch.nn.Module): The AI model to predict the move.

        Returns:
        str: Next move in xLAN format, e.g., 'Pe2e4'.
        """
        for _ in range(self.max_model_tries):
            model_move = self.model_prediction(model)
            UCI_move = converter.xlan_move_to_uci(self.board, model_move)
            try:
                if chess.Move.from_uci(UCI_move) in self.board.legal_moves:
                    if self.show_game_history:
                        print("Model move: ", model_move)
                    return model_move
            except ValueError:
                print("Model move is not legal, please try again. Move: ", model_move)

        # If the model fails to produce a valid move, ask for manual input
        print("Model failed to produce a valid move. Please enter a move manually.")
        user_move = self.get_next_move_from_player()
        print("Your played for model move: ", user_move)
        return user_move

    def make_move(self, move):
        """
        Attempts to make a move on the board. Validates if the move is legal, converts it to UCI format,
        and applies it to the board.

        Args:
        move (str): The move in xLAN format, e.g., 'Pe2e4'.

        Returns:
        bool: True if the move was successfully made, False otherwise.
        """
        UCI_move = converter.xlan_move_to_uci(self.board, move)
        if len(UCI_move) != 4 and len(UCI_move) != 5:
            print("Invalid move length, please try again. Your input: ", move)
            return False
        try:
            if chess.Move.from_uci(UCI_move) in self.board.legal_moves:
                self.board.push(chess.Move.from_uci(UCI_move))
                # Add move to move history in unique move notation - model uses this to predict next move
                self.movehistory += move + " "
                return True
            else:
                print("Invalid move, please try again. Move: ", move)
        except ValueError:
            print("Move is not legal, please try again. Move: ", move)
        return False

    def model_prediction(self, model):
        """
        Uses the AI model to predict a move based on the current move history.

        Args:
        model (torch.nn.Module): The AI model to predict the move.

        Returns:
        str: The predicted move by the model in xLAN format, e.g., 'Pe2e4'.
        """
        output, _, _ = generate_prediction(
            self.movehistory,
            self.TOKENS_TO_PREDICT,
            model,
            self.token_path,
            temperature=self.temperature,
        )
        if self.show_game_history:
            print("Model input: ", self.movehistory)
            print("Model Output: ", output)

        return output.split(" ")[-1]

    def check_result(self):
        """
        Checks the current state of the game and prints out the result if the game has ended.
        It handles various conditions such as checkmate, stalemate, insufficient material, and the 75-move rule.
        """
        if self.board.is_checkmate():
            print("Checkmate.")
        elif self.board.is_stalemate():
            print("Stalemate.")
        elif self.board.is_insufficient_material():
            print("Insufficient material.")
        elif self.board.is_seventyfive_moves():
            print("75-move rule.")
        elif self.board.is_fivefold_repetition():
            print("Fivefold repetition.")
        else:
            print("Game over by unknown reason.")

    def push_starting_sequence(self):
        """
        Pushes the starting sequence of moves to the board. The starting sequence is provided as a string
        of moves separated by spaces, e.g., "Pe2e4 Pe7e5 Ng1f3". It is converted to UCI format and pushed to the board.
        """
        starting_seqenze = self.movehistory.strip().split(" ")
        for move in starting_seqenze:
            try:
                # Convert the move to UCI format and push it to the board
                UCI_move = converter.xlan_move_to_uci(self.board, move)
                if chess.Move.from_uci(UCI_move) in self.board.legal_moves:
                    self.board.push(chess.Move.from_uci(UCI_move))
                else:
                    print(f"Invalid move encountered: {move}. Ending sequence.")
                    break
            except ValueError:
                print(
                    f"Move is not legal or incorrectly formatted: {move}. Ending sequence."
                )
                break

    def play_game(self):
        """
        The main method to start and play through the game. It handles turn-taking, move making, and game state updates.
        The game continues until a terminal state (checkmate, stalemate, etc.) is reached. It also handles the case
        where the maximum token length is reached and the player can choose to continue or remove the first moves.
        Model maximum input token length is 512 tokens
        """
        self.start_informations()
        display(self.display_output)
        self.push_starting_sequence()
        while not self.board.is_game_over():
            self.display_board()
            if self.show_game_history:
                display(self.board)
                print("Move ply (number): ", self.ply)

            if self.ply * self.TOKENS_TO_PREDICT >= 508:
                #  Player can choose to finish or remove first moves
                continue_game = input(
                    "Max moves reached. Remove first move now. Continue? (y/n): "
                )
                if continue_game == "y":
                    moves = self.movehistory.strip().split(" ")
                    moves = moves[1:]
                    self.movehistory = " ".join(moves)
                    self.movehistory += " "
                else:
                    print("Game stopped - Max moves reached.")
                    break

            try:
                if self.board.turn == chess.WHITE:
                    move = self.get_next_move(self.player1_type, self.model_p1)
                else:
                    move = self.get_next_move(self.player2_type, self.model_p2)

                while not self.make_move(move):
                    if self.board.turn == chess.WHITE:
                        move = self.get_next_move(self.player1_type, self.model_p1)
                    else:
                        move = self.get_next_move(self.player2_type, self.model_p2)

                self.ply += 1
            except ExitGameException as e:
                print(e)
                break

        self.display_board()
        self.check_result()
        display(self.board)

    def start_informations(self):
        """
        Prints out the starting information for the game.
        """

        print("Welcome to the Chess Game!")
        print(
            "Instructions: Enter your moves in xLAN format [Piece][Startsquare][Endsquare]. eg. Pe2e4"
        )
        print(
            "For Pawn promotion, enter the piece you want to promote instead of Pawn. eg. Qe7e8"
        )
        print(
            "The game will continue until a checkmate, stalemate, or other draw condition is reached."
        )
        print("To exit the game, enter 'exit'.")

    def show_position_from_sequence(self, move_sequence, save_to_file):
        """
        Takes a string of moves, plays them on the board, and displays the final position.

        Args:
        moves_string (str): A string containing moves in xLAN separated by spaces, e.g., "Pe2e4 Pe7e5 Ng1f3"
        """
        self.movehistory = move_sequence
        self.push_starting_sequence()

        display(self.display_output)
        self.display_board(save_to_file=save_to_file)


class ExitGameException(Exception):
    """Custom exception for exiting the game."""

    pass
