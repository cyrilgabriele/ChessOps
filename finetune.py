# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None


# +
import torch
from transformers import AutoModelForCausalLM
from src.train import ChessTrainer
from peft import LoraConfig, get_peft_model

# -


## HYPERPARAMETERS
BATCH_SIZE = 4  # use the largest batch size that fits on your GPU
SAVE_STEPS = 2000  # how often to save a checkpoint
LOGGING_STEPS = 50  # how often to validate model and publish it to Weights & Biases
EPOCHS = 1  # how many epochs to train for - how many times to go through the dataset
LEARNING_RATE = 0.0001  # learning rate - how fast the model should learn
SKIP_VALIDATION = True  # skip validation and only save model checkpoints
WEIGHTS_AND_BIASES_ENABLED = True  # enable logging to Weights & Biases
USE_FP16 = True  # enable mixed precision training (GPU only)
XLANPLUS_ENABLED = True  # use xLanPlus tokenizer

## MODEL
PEFT_BASE_MODEL = "Leon-LLM/Leon-Chess-350k-Plus"

## CONFIG FOR FINE-TUNING
R = 128  # lower means faster training, but might underfit because of less complexity (experiments don't show that training time increases, which is rather weird)
LORA_ALPHA = 32  # scaling factor that adjusts the magnitude of the combined result (balances the pretrained model’s knowledge and the new task-specific adaptation)
LORA_DROPOUT = 0.1

## PATHS
# dataset = "/Users/cyrilgabriele/Documents/School/00_Courses/03_MLOPS/04_Project/ChessOps/data/tokens/carlsen_max_768.tok"
# dataset = "./data/tokens/carlsen_max_768.tok"
# output_dir = f"/Users/cyrilgabriele/Documents/School/00_Courses/03_MLOPS/04_Project/ChessOps/models/"
output_dir = "models/"
model_name = f"{PEFT_BASE_MODEL.split('/')[1]}_LoRA_{chess_player}".replace("'", "")


def create_model():
    peft_config = LoraConfig(  # https://huggingface.co/docs/peft/v0.10.0/en/package_reference/lora#peft.LoraConfig
        task_type="CAUSAL_LM",  # This does not need to be changed for our use case
        inference_mode=False,  # don't change this for training, only later for inference
        r=R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
    )

    peft_model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(PEFT_BASE_MODEL), peft_config
    )

    return peft_model


def train_model(model, dataset, output_dir, debug=True):
    if debug:
        print(f"model: {model}")
        print(f"dataset: {dataset}")
        print(f"output_dir: {output_dir}")

    trainer = ChessTrainer(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        input_file=dataset,
        output_dir=output_dir,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        skip_validation=SKIP_VALIDATION,
        weight_and_biases=WEIGHTS_AND_BIASES_ENABLED,
        use_FP16=USE_FP16,
        notation="xLANplus" if XLANPLUS_ENABLED else "xLAN",
        peft=model,
    )

    trainer.train()
    #print("trainer.train()")


def push_model_to_hf(model, name):
    print(f"push_model_to_hf(model={model}, name={name})")
    model.push_to_hub(model_name)
    # pass


model = create_model()
train_model(model, dataset, output_dir + model_name)
push_model_to_hf(model, model_name)
