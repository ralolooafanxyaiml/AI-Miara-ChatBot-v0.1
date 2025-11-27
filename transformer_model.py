import tensorflow as tf
from tensorflow.keras import layers

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)

        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)

        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)

        return embedded_tokens + embedded_positions
    
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
   
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

# -----------------------------------------------------------------

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )

        self.dense_proj = tf.keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

     def call(self, inputs, mask=None):
         if mask is not None:
             mask = mask[:, tf.newaxis, :]

         attention_output = self.attention(
             inputs, inputs, attention_mask=mask
         )

         proj_input = self.layernorm_1(inputs + attention_output)
         proj_output = self.dense_proj(proj_input)
       
         return self.layernorm_2(proj_input + proj_output)

     def get_config(self):
         config = super().get_config()
         config.update({
             "embed_dim": self.embed_dim,
             "dense_dim": self.dense_dim,
             "num_heads": self.num_heads,
         })
         return config

def create_model(vocab_size, sequence_length, embed_dim, latent_dim, num_heads):
    inputs = layers.Input(shape=(sequence_length,), dtype="int64")
  
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
    x = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(vocab.size, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model






