import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model


# Declaring input tensors
question = Input(shape=(20, 300))
path = Input(shape=(20, 300))
# neg_path = Input(shape=(20, 300))

'''
    Encoder
'''
# Create two different encoders
path_encoder = LSTM(64)
ques_encoder = LSTM(64)

# Encode the question
encoded_question = ques_encoder(question)

# Encode the pos and neg path using the same path encoder
encoded_path = path_encoder(path)
# encoded_neg_path = path_encoder(neg_path)

# Concatenate question with the two paths
merged_path_ques = keras.layers.concatenate([encoded_question, encoded_pos_path], axis=-1)
# merged_negpath_ques = keras.layers.concatenate([encoded_question, encoded_pos_path], axis=-1)

'''
    Classifier
'''
# Declare a dense layer
classifier = Dense(1, activation='sigmoid')

# Use it on the pos and neg examples
y_pos = classifier(merged_pospath_ques)
# y_neg = classifier(merged_negpath_ques)

model = Model(inputs=[ques_encoder, pos_path, neg_path], outputs=y_pos)
model.compile(optimizer=["sgd"],
              loss="binary_crossentropy",
              metrics=["precision","recall","accuracy"])