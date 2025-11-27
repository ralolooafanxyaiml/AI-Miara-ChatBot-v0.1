import tensorflow as tf
import pickle
import os

from data_loader import prepare_dataset
from transformer_model import create_model

VOCAB_SIZE = 15000
SEQUENCE_LENGTH = 20
EMBED_DIM = 256
LATENT_DIM = 2048
NUM_HEADS = 8
EPOCHS = 30

DATA_PATH = "./raw_data"

if not os.path_exists(DATA_PATH) or not os.listdir(DATA_PATH):
    print("ERROR!")
    exit()

try:
    dataset, vectorizer = prepare_dataset(DATA_PATH)
except Exception as e:
    print("ERROR!)
    exit()

if dataset is None:
    print("ERROR!)
    exit()

pickle.dump({"config": vectorizer.get_config()
             "weights": vectorizer.get_weights()},
            open("vectorizer.pkl",  "wb))

model = create_model(
    vocab_size=VOCAB_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    embed_dim=EMBED_DIM,
    latent_dim=LATENT_DIM,
    num_heads=NUM_HEADS,
)

model.summary()

history = model.fit(
    dataset,
    epochs=EPOCHS
)

model.save("miara_llm_brain.keras")
