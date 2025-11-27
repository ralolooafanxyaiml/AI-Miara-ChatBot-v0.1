import numpy as np
import tensorflow as tf
import pickle
import os
from tensorflow.keras.layers import TextVectorization

SEQUENCE_LENGTH = 20
MAX_DECODED_LENGTH = 20

if not os.path.exists("miara_llm_brain.keras") or not os.path.exists("vectorizer.pkl"):
    print("ERROR!")
    exit()

from_disk = pickle.load(open("vectorizer.pkl", "rb"))
vectorizer = TextVectorization.from_config(from_disk["config])
vectorizer.set_weights(from_disk["weights])

model = tf.keras.models.load_model("miara_llm_brain.keras")

vocab = vectorizer.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))

def generate_response(input_text):
    tokenized_input = vectorizer([input_text])
    decoded_sentence = "[start]"

    for i in range(MAX_DECODED_LENGTH):
        tokenized_target = vectorizer([decoded_sentence])[:, :-1]

        predictions = model.predict(
            {"encoder_inputs": tokenized_input, "decoder_inputs": tokenized_target},
            verbose=0
        )
  
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup.get(sampled_token_index, "")
 
        if sampled_token == "[end]":
            break

        decoded_sentence += " " + sampled_token

     return decoded_sentence.replace("[start]", "").strip()

print("\n" + "="*40)
print("   I'm Miara, it's pleasure to meet you.")
print("   (Text q or exit if you want to quit.)")
print("="*40 + "\n")

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["q", "exit"]:
            print("Goodbye, i'll be here to help if you need.")
            break
        if not user_input.strip():
            continue

        response = generate_response(user_input)
        print(f"Miara : {response}\n")

     except KeyboardInterrupt:
         print("\nQuiting...")
         break
     except Exception as e:
         print(f"Error: {e}")