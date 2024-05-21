## Table of Contents

- [Table of Contents](#table-of-contents)
- [Project Description](#project-description)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Technology and Frameworks](#technology-and-frameworks)
- [Pipeline](#pipeline)
- [Evaluation](#evaluation)
- [Team Members](#team-members)
- [Trained Models](#trained-models)
- [Future Work](#future-work)

## Project Description

Welcome to **ChessOps**, a Streamlit application that allows you to play chess against some of the greatest chess players in history. This project was part of the course "Machine Learning Operations (MLOps)" course at [Zurich University of Applied Sciences (ZHAW)](https://www.zhaw.ch/). Currently, you can challenge the legendary Russian chess grandmaster Garry Kasparov and the five-time World Chess Champion Magnus Carlsen. Our goal is to provide an engaging, entertaining and educational experience for chess enthusiasts of all levels.

Of course, these are not these grandmasters themselves, but chess bots programmed to play in a similar way to the real players. Two AI models were trained on the playing styles of Kasparov and Carlsen, allowing you to experience the thrill of playing against these chess legends. The application provides an interactive chess board where you can make your moves and see the computer's responses in real-time.

The base model used for fine-tuning was trained by Jerome Maag and Lars Schmid during their [project work](https://github.zhaw.ch/schmila7/leon-llm) in the 5th semester of their bachelor's degree studies in computer science at ZHAW. It uses the GPT-2 transformer architecture developed by [OpenAI](https://openai.com/), which has been pretrained using a custom tokenizer on 350k chess games from the [Lichess database](https://database.lichess.org/).

The models were then fine-tuned on datasets containing thousands of games played by Kasparov and Carlsen respectively. The MLOps pipeline was implemented using Ploomber, ensuring efficient data processing and model deployment. The backend server was built using FastAPI, providing a robust and scalable API for the application. The front-end was developed using Streamlit, offering a clean and intuitive interface for users to play chess against the trained models.

## Features

- **Play Against Chess Legends**: Challenge Garry Kasparov and Magnus Carlsen, two of the most iconic chess players.
- **Interactive Game Board**: An easy-to-use, interactive chess board where you can make your moves and see the computer's responses in real-time.
- **User-Friendly Interface**: A clean and intuitive interface designed to enhance your chess-playing experience.

## How It Works

1. **Select Your Opponent**: Choose between Garry Kasparov and Magnus Carlsen in the dropdown menu above the game board.
2. **Make Your Moves**: Drag and drop playing pieces on the chess board to make your moves.
3. **Model Response**: Our GPT-2 models, fine-tuned to imitate the playing styles of Kasparov and Carlsen respectively, will respond with their characteristic moves.

## Installation

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

   - **Optional:** To use CUDA for training/prediction, download the appropriate PyTorch version listed on [PyTorch's official site](https://pytorch.org/get-started/locally/).

4. Run the backend server (FastAPI):
   ```sh
   fastapi dev backend/backend.py
   ```
5. Run the Streamlit app:
   ```sh
   streamlit run streamlit/app.py
   ```

## Technology and Frameworks

- **[FastAPI](https://fastapi.tiangolo.com/)**: Robust and scalable API used for the backend server.
- **[GPT-2](https://huggingface.co/docs/transformers/en/model_doc/gpt2)**: Transformer architecture developed by OpenAI, pre-trained specifically for chess.
- **[Hugging Face](https://huggingface.co/)**: Free and easy-to-use platform for hosting and deploying machine learning models.
- **[PEFT](https://huggingface.co/docs/peft/index)**: Library for fine-tuning the GPT-2 model on the datasets containing thousands of games played by Kasparov and Carlsen respectively. This project specifically makes use of the [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) method.
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation library used for data cleaning and processing.
- **[Ploomber](https://ploomber.io/)**: Used for the MLOps pipeline, ensuring efficient data processing and model deployment and providing an automated workflow for data cleaning, data processing, and fine-tuning.
- **[PyTorch](https://pytorch.org/)**: Open-source machine learning library used to train and fine-tune the GPT-2 model.
- **[Python-Chess](https://pypi.org/project/python-chess/)**: Python library for chess, used to implement the chess game logic.
- **[Streamlit](https://streamlit.io/)**: Library for the front-end, providing a seamless and interactive user interface.

## Pipeline

The MLOps pipeline consists of the following steps (see `pipeline.yaml`):

- **Data Collection**: Games in the form of PGN files are collected manually from various sources:
  - [Magnus Carlsen Lichess Games Dataset](https://www.kaggle.com/datasets/zq1200/magnus-carlsen-lichess-games-dataset)
  - [Garry Kasparov](https://www.pgnmentor.com/players/Kasparov.zip)
- **Data Cleaning**: The PGN files are cleaned by filtering only the game sequences and removing any metadata or comments. Duplicate games are removed, and games longer than 192 plies are omitted.
- **Data Processing**: The PGN files are converted into the format xLANplus (a notation that had been developed specifically for the project work), which is then tokenized using a custom tokenizer.
- **Fine-Tuning**: The [GPT-2 model](https://github.com/openai/gpt-2) is fine-tuned using the [PEFT (Parameter-Efficient Fine-Tuning) library](https://huggingface.co/docs/peft/index) from [Hugging Face](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora), making use of the [LoRA (Low-Rank Adaptation) method](https://arxiv.org/abs/2106.09685), a technique that facilitates the fine-tuning of large models while consuming very little memory. Fine-tuning in this way creates so-called "adapters" (see `pipeline/models`) which then have to be loaded onto the base model in order to make predictions.

## Evaluation

The fine-tuned models have been evaluated by making use of the Wasserstein Distance to compare the distribution of the moves generated by the models and the moves from the training data. The Wasserstein Distance is a metric that measures the distance between two probability distributions over a metric space. The lower the Wasserstein Distance, the better the model imitates the playing style of the respective chess player.

For each of the models, 1000 games with a maximum length of 125 plies (half-moves in chess) have been generated (see `src/generate_sequences_for_statistical_evaluation.ipynb`) and evaluated (see `src/statistical_evaluations_from_pgn.ipynb` and `src/statistical_evaluations_from_sequences.ipynb`) by distributing the plies into buckets of 10 and then comparing them as follows (see `src/statistical_evaluations_wasserstein.ipynb`):

The Wasserstein Distances between the distribution of the moves generated by the base model and the distribution of the moves contained in the training data is, as well as the distances between the moves generated by the fine-tuned models and the distribution of the training data. The aim is for the fine-tuned models to have a Wasserstein Distance to the training data as low as possible, and preferably lower than the distance between the distribution of the base model and that of the training data. The results of the evaluation as well as the generated game sequences can be found in the `evaluation` folder.

## Team Members

- **[Cyril Gabriele](https://github.com/cyrilgabriele) (MLOps Engineer)**: responsible for automation and maintenance of pipeline as well as the deployment infrastructure.
- **[Jerome Maag](https://github.com/JeromeMaag) (Data Engineer)**: responsible for data collection, data cleaning, and data processing.
- **[Lars Schmid](https://github.com/larscarl) (Machine Learning Engineer)**: responsible for training and fine-tuning the AI models as well as evaluating the model performance.

## Trained Models

- **Base Model** (350k games):
  - [GPT-2 Architecture pre-trained on games from the Lichess database played in September 2023](https://huggingface.co/Leon-LLM/Leon-Chess-350k-BOS)
- **Garry Kasparov** (2122 games):
  - [Kasparov-Model fine-tuned for 5 Epochs](https://huggingface.co/larscarl/Leon-Chess-350k-Plus_LoRA_kasparov_5E_0.0001LR)
  - [Kasparov-Model fine-tuned for 10 Epochs](https://huggingface.co/larscarl/Leon-Chess-350k-Plus_LoRA_kasparov_10E_0.0001LR)
- **Magnus Carlsen** (5597 games):
  - [Carlsen-Model fine-tuned for 5 Epochs](https://huggingface.co/larscarl/Leon-Chess-350k-Plus_LoRA_carlsen_5E_0.0001LR)
  - [Carlsen-Model fine-tuned for 10 Epochs](https://huggingface.co/larscarl/Leon-Chess-350k-Plus_LoRA_carlsen_10E_0.0001LR)

## Future Work

Future enhancements may include:

- Adding more chess players.
- Improving the models' accuracy as well as their adherence to the playing style of the respective chess players.
- Enhancing the user interface.
