# @Time    : 2019/5/21 19:24
# @Author  : shakespere
# @FileName: baseline3.py
import sys, os, re, csv, codecs, numpy as np, pandas as pd

# =================Keras==============
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Conv2D, Embedding, Dropout, Activation
from keras.layers import Bidirectional, MaxPooling1D, MaxPooling2D, Reshape, Flatten, concatenate, BatchNormalization,CuDNNGRU
from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers, backend
# =================nltk===============
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn import svm,metrics

import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
path = './data/'
BATCH_SIZE = 512
EMBEDDING_FILE = f'{path}glove.6B.50d.txt'
EMBEDDING_FILE1 = f'{path}glove.840B.300d.txt'
EMBEDDING_FILE2 = f'{path}crawl-300d-2M.vec'
TRAIN_DATA_FILE = f'{path}train.csv'
TEST_DATA_FILE = f'{path}20190520_test.csv'

embed_size = 300  # how big is each word vector
max_features = 20000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100  # max number of words in a comment to use
number_filters = 100  # the number of CNN filters

train = pd.read_csv(TRAIN_DATA_FILE,lineterminator='\n')
test = pd.read_csv(TEST_DATA_FILE,lineterminator='\n')
#
# train_text = pd.read_csv(TRAIN_DATA_FILE,index_col='ID',lineterminator='\n')
# test_text = pd.read_csv(TEST_DATA_FILE,index_col='ID',lineterminator='\n')

list_sentences_train = train["review"].fillna("_na_").values
y = train['label'].map({'Negative':0, 'Positive': 1})
list_sentences_test = test["review"].fillna("_na_").values
print(y[:10])

special_character_removal = re.compile(r'[^a-z\d ]', re.IGNORECASE)
replace_numbers = re.compile(r'\d+', re.IGNORECASE)

# preprocess
#
# import re
#
# import random
#
#
# def set_seed(seed=0):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#
#
# set_seed(2411)
# SEED = 42
# import psutil
# from multiprocessing import Pool
# import multiprocessing
#
# num_partitions = 10  # number of partitions to split dataframe
# num_cores = psutil.cpu_count()  # number of cores on your machine
#
# print('number of cores:', num_cores)
#
#
# def df_parallelize_run(df, func):
#     df_split = np.array_split(df, num_partitions)
#     pool = Pool(num_cores)
#     df = pd.concat(pool.map(func, df_split))
#     pool.close()
#     pool.join()
#
#     return df
# # remove space
# spaces = ['\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\x10', '\x7f', '\x9d', '\xad', '\xa0']
#
#
# def remove_space(text):
#     """
#     remove extra spaces and ending space if any
#     """
#     for space in spaces:
#         text = text.replace(space, ' ')
#     text = text.strip()
#     text = re.sub('\s+', ' ', text)
#     return text
#
#
# # replace strange punctuations and raplace diacritics
# from unicodedata import category, name, normalize
#
#
# def remove_diacritics(s):
#     return ''.join(
#         c for c in normalize('NFKD', s.replace('ø', 'o').replace('Ø', 'O').replace('⁻', '-').replace('₋', '-'))
#         if category(c) != 'Mn')
#
#
# special_punc_mappings = {"—": "-", "–": "-", "_": "-", '”': '"', "″": '"', '“': '"', '•': '.', '−': '-',
#                          "’": "'", "‘": "'", "´": "'", "`": "'", '\u200b': ' ', '\xa0': ' ', '،': '', '„': '',
#                          '…': ' ... ', '\ufeff': ''}
#
#
# def clean_special_punctuations(text):
#     for punc in special_punc_mappings:
#         if punc in text:
#             text = text.replace(punc, special_punc_mappings[punc])
#     text = remove_diacritics(text)
#     return text
#
#
# # clean numbers
# def clean_number(text):
#     text = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', text)  # digits followed by a single alphabet...
#     text = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', text)  # 1st, 2nd, 3rd, 4th...
#     text = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', text)
#     return text
#
#
# import string
#
# regular_punct = list(string.punctuation)
# extra_punct = [
#     ',', '.', '"', ':', ')', '(', '!', '?', '|', ';', "'", '$', '&',
#     '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£',
#     '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›',
#     '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”',
#     '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',
#     '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼',
#     '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
#     'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',
#     '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
#     '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡', '§', '£', '₤',
#     ':)', ': )', ':-)', '(:', '( :', '(-:', ':\')',
#     ':D', ': D', ':-D', 'xD', 'x-D', 'XD', 'X-D',
#     '<3', ':*',
#     ';-)', ';)', ';-D', ';D', '(;', '(-;',
#     ':-(', ': (', ':(', '\'):', ')-:',
#     '-- :', '(', ':\'(', ':"(\'', ]
#
#
# def handle_emojis(text):
#     # Smile -- :), : ), :-), (:, ( :, (-:, :')
#     text = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', text)
#     # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
#     text = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', text)
#     # Love -- <3, :*
#     text = re.sub(r'(<3|:\*)', ' EMO_POS ', text)
#     # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
#     text = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', text)
#     # Sad -- :-(, : (, :(, ):, )-:
#     text = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', text)
#     # Cry -- :,(, :'(, :"(
#     text = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', text)
#     return text
#
#
# def stop(text):
#     from nltk.corpus import stopwords
#
#     text = " ".join([w.lower() for w in text.split()])
#     stop_words = stopwords.words('english')
#
#     words = [w for w in text.split() if not w in stop_words]
#     return " ".join(words)
#
#
# all_punct = list(set(regular_punct + extra_punct))
# # do not spacing - and .
# all_punct.remove('-')
# all_punct.remove('.')
#
#
# # clean repeated letters
# def clean_repeat_words(text):
#     text = re.sub(r"(I|i)(I|i)+ng", "ing", text)
#     text = re.sub(r"(L|l)(L|l)(L|l)+y", "lly", text)
#     text = re.sub(r"(A|a)(A|a)(A|a)+", "a", text)
#     text = re.sub(r"(C|c)(C|c)(C|c)+", "cc", text)
#     text = re.sub(r"(D|d)(D|d)(D|d)+", "dd", text)
#     text = re.sub(r"(E|e)(E|e)(E|e)+", "ee", text)
#     text = re.sub(r"(F|f)(F|f)(F|f)+", "ff", text)
#     text = re.sub(r"(G|g)(G|g)(G|g)+", "gg", text)
#     text = re.sub(r"(I|i)(I|i)(I|i)+", "i", text)
#     text = re.sub(r"(K|k)(K|k)(K|k)+", "k", text)
#     text = re.sub(r"(L|l)(L|l)(L|l)+", "ll", text)
#     text = re.sub(r"(M|m)(M|m)(M|m)+", "mm", text)
#     text = re.sub(r"(N|n)(N|n)(N|n)+", "nn", text)
#     text = re.sub(r"(O|o)(O|o)(O|o)+", "oo", text)
#     text = re.sub(r"(P|p)(P|p)(P|p)+", "pp", text)
#     text = re.sub(r"(Q|q)(Q|q)+", "q", text)
#     text = re.sub(r"(R|r)(R|r)(R|r)+", "rr", text)
#     text = re.sub(r"(S|s)(S|s)(S|s)+", "ss", text)
#     text = re.sub(r"(T|t)(T|t)(T|t)+", "tt", text)
#     text = re.sub(r"(V|v)(V|v)+", "v", text)
#     text = re.sub(r"(Y|y)(Y|y)(Y|y)+", "y", text)
#     text = re.sub(r"plzz+", "please", text)
#     text = re.sub(r"(Z|z)(Z|z)(Z|z)+", "zz", text)
#     text = re.sub(r"(-+|\.+)", " ", text)  # new haha #this adds a space token so we need to remove xtra spaces
#     return text
#
#
# def spacing_punctuation(text):
#     """
#     add space before and after punctuation and symbols
#     """
#     for punc in all_punct:
#         if punc in text:
#             text = text.replace(punc, f' {punc} ')
#     return text
#
#
# def preprocess(text):
#     """
#     preprocess text main steps
#     """
#     text = remove_space(text)
#     text = clean_special_punctuations(text)
#     text = handle_emojis(text)
#     text = clean_number(text)
#     text = spacing_punctuation(text)
#     text = clean_repeat_words(text)
#     text = remove_space(text)
#     # text = stop(text)# if changing this, then chnage the dims
#     # (not to be done yet as its effecting the embeddings..,we might be
#     # loosing words)...
#     return text
#
#
# mispell_dict = {'😉': 'wink', '😂': 'joy', '😀': 'stuck out tongue', 'theguardian': 'the guardian',
#                 'deplorables': 'deplorable', 'theglobeandmail': 'the globe and mail', 'justiciaries': 'justiciary',
#                 'creditdation': 'Accreditation', 'doctrne': 'doctrine', 'fentayal': 'fentanyl',
#                 'designation-': 'designation', 'CONartist': 'con-artist', 'Mutilitated': 'Mutilated',
#                 'Obumblers': 'bumblers', 'negotiatiations': 'negotiations', 'dood-': 'dood', 'irakis': 'iraki',
#                 'cooerate': 'cooperate', 'COx': 'cox', 'racistcomments': 'racist comments',
#                 'envirnmetalists': 'environmentalists', }
# contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
#                        "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
#                        "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
#                        "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
#                        "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would",
#                        "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
#                        "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
#                        "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
#                        "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
#                        "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
#                        "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
#                        "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
#                        "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
#                        "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
#                        "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
#                        "she'll": "she will", "she'll've": "she will have", "she's": "she is",
#                        "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
#                        "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",
#                        "that'd've": "that would have", "that's": "that is", "there'd": "there would",
#                        "there'd've": "there would have", "there's": "there is", "here's": "here is",
#                        "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
#                        "they'll've": "they will have", "they're": "they are", "they've": "they have",
#                        "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
#                        "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
#                        "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
#                        "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
#                        "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have",
#                        "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
#                        "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
#                        "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
#                        "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
#                        "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
#                        "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
#                        "you'll've": "you will have", "you're": "you are", "you've": "you have"}
#
#
# def correct_spelling(x, dic):
#     for word in dic.keys():
#         x = x.replace(word, dic[word])
#     return x
#
#
# def correct_contraction(x, dic):
#     for word in dic.keys():
#         x = x.replace(word, dic[word])
#     return x
#
#
# from tqdm import tqdm
#
# tqdm.pandas()
#
#
# def text_clean_wrapper(df):
#     print(df)
#     df["review"] = df["review"].transform(preprocess)
#     df['review'] = df['review'].transform(lambda x: correct_spelling(x, mispell_dict))
#     df['review'] = df['review'].transform(lambda x: correct_contraction(x, contraction_mapping))
#     return df
#
# train = df_parallelize_run(train_text, text_clean_wrapper)
# test  = df_parallelize_run(test_text, text_clean_wrapper)
# preprocess




def text_to_wordlist(text, remove_stopwords=True, stem_words=True):
    # Remove Special Characters
    text = special_character_removal.sub('', text)

    # Replace Numbers
    text = replace_numbers.sub('n', text)
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text)


comments = []
for text in list_sentences_train:
    comments.append(text_to_wordlist(text))

test_comments = []
for text in list_sentences_test:
    test_comments.append(text_to_wordlist(text))

tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'', lower=True)
# tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train['review']) + list(test['review']))
comments_sequence = tokenizer.texts_to_sequences(list(train['review']))
test_comments_sequence = tokenizer.texts_to_sequences(list(test['review']))
word_index = tokenizer.word_index
X_t = pad_sequences(comments_sequence, maxlen=maxlen)
X_te = pad_sequences(test_comments_sequence, maxlen=maxlen)

# X_t = X_t.reshape((X_t.shape[0], 1, X_t.shape[1]))
# X_te = X_te.reshape((X_te.shape[0], 1, X_te.shape[1]))


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


# embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
def load_embeddings(embed_dir=EMBEDDING_FILE):
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embed_dir))
    return embeddings_index
def build_embedding_matrix(word_index,embeddings_index,max_features,lower=True,verbose=True):
    embedding_matrix = np.zeros((max_features, 300))
    for word, i in tqdm(word_index.items(), disable=not verbose):
        if lower:
            word = word.lower()
        if i >= max_features: continue
        try:
            embedding_vector = embeddings_index[word]
        except:
            embedding_vector = embeddings_index["unknown"]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
def build_matrix(word_index, embeddings_index):
    embedding_matrix = np.zeros((len(word_index) + 1,embed_size))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embeddings_index[word]
        except:
            embedding_matrix[i] = embeddings_index["unknown"]
    return embedding_matrix
# all_embs = np.stack(embeddings_index.values())
# emb_mean, emb_std = all_embs.mean(), all_embs.std()
#
#
# word_index = tokenizer.word_index
# nb_words = min(max_features, len(word_index))
# # embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
#
# embedding_matrix = l
# for word, i in word_index.items():
#     if i >= max_features: continue
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None: embedding_matrix[i] = embedding_vector
embedding_index = load_embeddings(embed_dir=EMBEDDING_FILE1)
embedding_matrix = build_matrix(word_index=word_index,embeddings_index=embedding_index)

def model_text_cnn():
    inp = Input(shape=(1, maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x1 = Conv2D(number_filters, (3, embed_size), data_format='channels_first')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((int(int(x1.shape[2]) / 1.5), 1), data_format='channels_first')(x1)
    x1 = Flatten()(x1)

    x2 = Conv2D(number_filters, (4, embed_size), data_format='channels_first')(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation('elu')(x2)
    x2 = MaxPooling2D((int(int(x2.shape[2]) / 1.5), 1), data_format='channels_first')(x2)
    x2 = Flatten()(x2)

    x3 = Conv2D(number_filters, (5, embed_size), data_format='channels_first')(x)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = MaxPooling2D((int(int(x3.shape[2]) / 1.5), 1), data_format='channels_first')(x3)
    x3 = Flatten()(x3)

    x4 = Conv2D(number_filters, (6, embed_size), data_format='channels_first')(x)
    x4 = BatchNormalization()(x4)
    x4 = Activation('elu')(x4)
    x4 = MaxPooling2D((int(int(x4.shape[2]) / 1.5), 1), data_format='channels_first')(x4)
    x4 = Flatten()(x4)

    x5 = Conv2D(number_filters, (7, embed_size), data_format='channels_first')(x)
    x5 = BatchNormalization()(x5)
    x5 = Activation('relu')(x5)
    x5 = MaxPooling2D((int(int(x5.shape[2]) / 1.5), 1), data_format='channels_first')(x5)
    x5 = Flatten()(x5)


    x = concatenate([x1, x2, x3, x4, x5])

    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers,regularizers,constraints,optimizers,layers
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

def model_gru_attn():
    inp = Input(shape=(maxlen,))
    x = Embedding(len(word_index) + 1, embed_size, weights=[embedding_matrix],trainable=False)(inp)
    # x = Bidirectional(CuDNNGRU(128,return_sequences=True))(x)
    # x = Bidirectional(CuDNNGRU(100,return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(64,return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(1,activation='sigmoid')(x)
    model = Model(inputs=inp,outputs=x)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
# folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=2019)
predictions = np.zeros(X_te.shape[0])
aucs = []
oof_preds = np.zeros(X_t.shape[0])

# for fold_, (train_index, test_index) in enumerate(folds.split(X_t, y)):
#     print("Fold :{}".format(fold_ + 1))
#     cv_train_data, cv_train_label= X_t[train_index], y[train_index]
#     cv_test_data, cv_test_label = X_t[test_index], y[test_index]
#
#     model.fit(cv_train_data, cv_train_label)
#     auc = metrics.roc_auc_score(cv_test_label, model.predict([cv_test_data], batch_size=1024, verbose=1))
#     preds = model.predict([X_te], batch_size=1024, verbose=1) / folds.n_splits
#     print(preds[:10])
#     print(predictions[:10])
#
#     aucs.append(auc)
#     print("auc score: %.5f" % auc)
n_splits = 5
splits = list(KFold(n_splits=n_splits,random_state=2019).split(X_t,y))
# skf = StratifiedKFold(y, n_folds=n_splits, shuffle=True,random_state=2019)
for fold_ in range(n_splits):

    train_index,test_index = splits[fold_]
# for fold_, (train_index, test_index) in enumerate(skf):
    print("Fold :{}".format(fold_ + 1))
    backend.clear_session()
    cv_train_data, cv_train_label= X_t[train_index], y[train_index]
    cv_test_data, cv_test_label = X_t[test_index], y[test_index]

    model = model_gru_attn()
    model.fit(cv_train_data, cv_train_label>0.5,batch_size=BATCH_SIZE,epochs=30,validation_data=(cv_test_data,cv_test_label>0.5))

    oof_preds[test_index] += model.predict(cv_test_data)[:,0]
    predictions += model.predict(X_te)[:,0]

    # auc = metrics.roc_auc_score(cv_test_label, model.predict([cv_test_data], batch_size=1024, verbose=1)[:,0])
    # preds = model.predict([X_te], batch_size=1024, verbose=1) / n_splits
    # print(preds[:10])
    # print(predictions[:10])
    #
    # aucs.append(auc)
    # print("auc score: %.5f" % auc)
predictions /= n_splits
auc = metrics.roc_auc_score(y,oof_preds)
print('Mean auc', auc)
predictions = pd.DataFrame(predictions)
id = pd.DataFrame(np.arange(1, len(predictions) + 1))
data = pd.concat([id, predictions], axis=1)
data.to_csv('./data/{}_predictions.csv'.format(auc), header=['ID', 'Pred'], index=False)

# model.fit(X_t, y, batch_size=1280, epochs=3)
#
# y_test = model.predict([X_te], batch_size=1024, verbose=1)
# sample_submission = pd.read_csv(f'{path}{comp}sample_submission.csv')
# sample_submission[list_classes] = y_test
# sample_submission.to_csv('submission_textcnn.csv', index=False)