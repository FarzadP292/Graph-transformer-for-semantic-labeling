from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
os.makedirs("./ptmodel")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

model.save_pretrained("./ptmodel")

tokenizer.save_pretrained("./ptmodel")
