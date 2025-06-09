import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate
import numpy as np
import sys

def get_user_choice(prompt_message, options_dict):
    """
    Displays a numbered list of options and prompts the user for a choice.
    Validates the input and returns the selected option's value.
    """
    print(f"\n{prompt_message}")
    # Create a mapping from number to option key
    valid_choices = {}
    for i, key in enumerate(options_dict.keys(), 1):
        print(f"  {i}: {key}")
        valid_choices[str(i)] = key
    
    while True:
        try:
            choice = input(f"Please enter your choice (1-{len(valid_choices)}): ")
            if choice in valid_choices:
                selected_key = valid_choices[choice]
                print(f"You selected: '{selected_key}'")
                return options_dict[selected_key]
            else:
                print("Invalid choice. Please enter a number from the list.")
        except (ValueError, KeyError):
            print("Invalid input. Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nSelection cancelled. Exiting.")
            sys.exit()


def run_finetuning_experiment():
    """
    Main function to run a single fine-tuning experiment.
    """
    # Interactive Configuration 

    # Define paths to local models and data
    script_dir = Path(__file__).parent  # directory containing this script
    project_root = script_dir.parent  # Project Root
    base_models_path = project_root / 'pretrained_models'  # directory containing models and tokenizers
    data_path = project_root / 'Data' / 'finetuning' # directory containing finetuning corpera
    
    model_choices = {
        "EN-DE Parent Model": {
            "model": base_models_path / "en-de-model",
            "tokenizer": base_models_path / "en-de-tokenizer",
            "name": "en-de"
        },
        "EN-FR Parent Model": {
            "model": base_models_path / "en-fr-model",
            "tokenizer": base_models_path / "en-fr-tokenizer",
            "name": "en-fr"
        }
    }

    corpus_choices = {
        "Creative Corpus (Fictional Literature)": {
            "path": data_path / "creative_corpus.json",
            "name": "creative"
        },
        "Instructive Corpus (Non-Creative)": {
            "path": data_path / "Instructive texts-aggregate.json",
            "name": "instructive"
        }
    }

    # Get user selections
    selected_model_info = get_user_choice("Select the pre-trained model to fine-tune:", model_choices)
    selected_corpus_info = get_user_choice("Select the corpus for fine-tuning:", corpus_choices)
    
    model_to_finetune = selected_model_info["model"]
    tokenizer_to_use = selected_model_info["tokenizer"]
    training_file = str(selected_corpus_info["path"])
    
    version_tag   = input(
        "\nOptional version label for this run. "
        "Leave blank to skip: "
    ).strip()
    version_suffix = f"_v{version_tag}" if version_tag else ""
    
    # Dynamically create the output directory name
    output_dir = Path(
        f"./results/{selected_model_info['name']}{version_suffix}"
        f"_finetuned_on_{selected_corpus_info['name']}"
    )
    print(f"\nFinal model will be saved to: {output_dir}")

    # Training Hyperparameters
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 16
    NUM_EPOCHS = 200
    WEIGHT_DECAY = 0.01
    EARLY_STOPPING_PATIENCE = 5

    # Load Model and Tokenizer FROM LOCAL PATHS
    print(f"Loading tokenizer from local path: {tokenizer_to_use}...")
    tokenizer = MarianTokenizer.from_pretrained(tokenizer_to_use)
    
    print(f"Loading model from local path: {model_to_finetune} for transfer learning...")
    model = MarianMTModel.from_pretrained(model_to_finetune)

    # Load and Prepare the Dataset
    print(f"Loading and processing dataset from: {training_file}")
    raw_dataset = load_dataset("json", data_files=training_file, split="train")
    split_datasets = raw_dataset.train_test_split(train_size=0.9, seed=42)
    split_datasets["validation"] = split_datasets.pop("test")
    print(f"Dataset splits: {split_datasets}")

    # Tokenization
    max_input_length = 128
    max_target_length = 128
    source_lang = "en"
    target_lang = "nl"
    
    validation_sources = [
        ex["translation"][source_lang]
        for ex in split_datasets["validation"]
    ]

    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing the dataset...")
    tokenized_datasets = split_datasets.map(preprocess_function, batched=True)

    # Set up Trainer and Metrics
    metric_bleu  = evaluate.load("sacrebleu")
    metric_chrf  = evaluate.load("chrf")
    metric_comet = evaluate.load("comet")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Decode predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Decode labels, handling the padding tokens
        # Replace -100 with the pad_token_id to ensure proper decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Post-process to strip whitespace
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        # BLEU
        result = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        
        # chrF
        chrf_res = metric_chrf.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )
        result["chrf"] = chrf_res["score"]

        # COMET (expects flat refs list)
        comet_res = metric_comet.compute(
            predictions=decoded_preds,
            references=[r[0] for r in decoded_labels],
            sources=validation_sources
        )
        result["comet"] = comet_res["mean_score"]
        
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",      # Evaluate at the end of each epoch
            save_strategy="epoch",            # Save a checkpoint at the end of each epoch
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            weight_decay=WEIGHT_DECAY,
            save_total_limit=3,
            num_train_epochs=NUM_EPOCHS,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            load_best_model_at_end=True,      # This will load the best model at the end of training
            metric_for_best_model="bleu",
            greater_is_better=True,
        )
    
    # Create and Run the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")

    # Save the Final Model
    print(f"Saving the best model to {output_dir}")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    run_finetuning_experiment()