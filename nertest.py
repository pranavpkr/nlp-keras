from keras_contrib.layers import CRF
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import re,numpy as np
import string,pickle

path = '/path/were/this/file/is/saved/lstm_crf_weights'

def create_custom_objects():
    instanceHolder = {"instance": None}
    class ClassWrapper(CRF):
        def __init__(self, *args, **kwargs):
            instanceHolder["instance"] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)
    def loss(*args):
        method = getattr(instanceHolder["instance"], "loss_function")
        return method(*args)
    def accuracy(*args):
        method = getattr(instanceHolder["instance"], "accuracy")
        return method(*args)
    return {"ClassWrapper": ClassWrapper ,"CRF": ClassWrapper, "loss": loss, "accuracy":accuracy}

def load_keras_model(path):
    model = load_model(path, custom_objects=create_custom_objects())
    return model

# Custom Tokenizer
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

f1 = open('/path/were/this/file/is/saved/word_to_index.pickle', 'rb')
word2idx = pickle.load(f1)

f2 = open('/path/were/this/file/is/saved/tag_to_index.pickle', 'rb')
tag2idx = pickle.load(f2)
# Vocabulary Key:tag_index -> Value:Label/Tag
idx2tag = {i: w for w, i in tag2idx.items()}


model = load_keras_model(path) 
test_sentence = tokenize('I am Garaang') # Tokenization
# Preprocessing
x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
                          padding="post", value=word2idx["PAD"], maxlen=75)
# Evaluation
p = model.predict(np.array([x_test_sent[0]]))
p = np.argmax(p, axis=-1)
# Visualization
print("{:15}|{}".format("Word", "Prediction"))
print(30 * "=")
for w, pred in zip(test_sentence, p[0]):
    print("{:15}: {:5}".format(w, idx2tag[pred]))
        


