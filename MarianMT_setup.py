from transformers import MarianMTModel, MarianTokenizer
from pathlib import Path

# Define model names
model_name_en_fr = "Helsinki-NLP/opus-mt-en-fr"
model_name_en_de = "Helsinki-NLP/opus-mt-en-de"

project_root = Path(__file__).parent
models_dir = project_root / "pretrained_models"

# Create the directory to save models
models_dir.mkdir(parents=True, exist_ok=True)

#  EN-FR Model
print(f"Attempting to load EN-FR tokenizer: {model_name_en_fr}")
tokenizer_en_fr = MarianTokenizer.from_pretrained(model_name_en_fr)
print(f"Attempting to load EN-FR model: {model_name_en_fr}")
model_en_fr = MarianMTModel.from_pretrained(model_name_en_fr)

# Define save paths
save_path_tokenizer_en_fr = models_dir / "en-fr-tokenizer"
save_path_model_en_fr = models_dir / "en-fr-model"

# Save them
tokenizer_en_fr.save_pretrained(save_path_tokenizer_en_fr)
model_en_fr.save_pretrained(save_path_model_en_fr)
print(f"EN-FR model and tokenizer downloaded and saved to {save_path_model_en_fr}.")

# EN-DE Model
print(f"\nAttempting to load EN-DE tokenizer: {model_name_en_de}")
tokenizer_en_de = MarianTokenizer.from_pretrained(model_name_en_de)
print(f"Attempting to load EN-DE model: {model_name_en_de}")
model_en_de = MarianMTModel.from_pretrained(model_name_en_de)

# Define save paths
save_path_tokenizer_en_de = models_dir / "en-de-tokenizer"
save_path_model_en_de = models_dir / "en-de-model"

# Save them
tokenizer_en_de.save_pretrained(save_path_tokenizer_en_de)
model_en_de.save_pretrained(save_path_model_en_de)
print(f"EN-DE model and tokenizer downloaded and saved to {save_path_model_en_de}.")

print("\nPre-trained models acquired successfully!")