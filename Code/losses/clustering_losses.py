from tensorflow import keras


cosine_similarity_loss = keras.losses.CosineSimilarity(
    axis=-1,
    name='cosine_similarity'
)
