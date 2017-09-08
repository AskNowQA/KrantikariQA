import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model


# Declaring input tensors
x_ques = Input(shape=(20, 300))
x_path = Input(shape=(20, 300))
# neg_path = Input(shape=(20, 300))

'''
    Encoder
'''
# Create two different encoders
path_encoder = LSTM(64)
ques_encoder = LSTM(64)

# Encode the question
encoded_question = ques_encoder(x_ques)

# Encode the pos and neg path using the same path encoder
encoded_path = path_encoder(x_path)
# encoded_neg_path = path_encoder(neg_path)

# Concatenate question with the two paths
merged_path_ques = keras.layers.concatenate([encoded_question, encoded_path], axis=-1)
# merged_negpath_ques = keras.layers.concatenate([encoded_question, encoded_pos_path], axis=-1)

'''
    Classifier
'''
# Declare a dense layer
classifier = Dense(1, activation='sigmoid')

# Use it on the pos and neg examples
y = classifier(merged_path_ques)
# y_neg = classifier(merged_negpath_ques)

model = Model(inputs=[x_ques, x_path], outputs=y)
model.compile(optimizer=["sgd"],
              loss="binary_crossentropy",
              metrics=["precision", "recall", "accuracy"])