
# ChessOps

## Project Description

Welcome to **ChessOps**, a Streamlit application that allows you to play chess against some of the greatest chess players in history. This was a project of the MLOps course of the Zurich University of Applied Sciences. Currently, you can challenge the legendary Garry Kasparov and the reigning champion Magnus Carlsen. Our goal is to provide an engaging and educational experience for chess enthusiasts of all levels.


### Features

- **Play Against Legends**: Challenge Garry Kasparov and Magnus Carlsen, two of the most iconic chess players.
- **Interactive Chess Board**: An easy-to-use, interactive chess board where you can make your moves and see the computer's responses in real-time.
- **Real-Time Feedback**: Get instant feedback on your moves and learn from the best.
- **User-Friendly Interface**: A clean and intuitive interface designed to enhance your chess-playing experience.
### Technology Stack

- **Streamlit**: For the front-end application, providing a seamless and interactive user interface.
- **Ploomber**: Used for the MLOps pipeline, ensuring efficient data processing and model deployment.
### How It Works

1. **Select Your Opponent**: Choose between Garry Kasparov and Magnus Carlsen.
2. **Make Your Moves**: Use the interactive chess board to make your moves.
3. **AI Response**: Our AI, trained on the playing styles of Kasparov and Carlsen, will respond with their characteristic moves.
4. **Learn and Improve**: Analyze the game, get real-time feedback, and improve your chess skills by learning from the masters.
### Installation

To run the application locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/cyrilgabriele/ChessOps.git
   ```
2. Navigate to the project directory:
   ```sh
   cd ChessOps
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the backend server (FastAPI):
   ```sh
   fastapi dev backend/backend.py 
   ```
5. Run the Streamlit app:
   ```sh
   streamlit run streamlit/app.py
   ```
