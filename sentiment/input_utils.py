import numpy as np
import re
import unicodedata

from itertools import accumulate, chain
from nltk import PorterStemmer

stemmer = PorterStemmer()

PADDING_IDX = 0
UNKNOWN_IDX = 1

def stem(text):
    """Runs NLTK PorterStemmer on the text and returns the stemmed version"""
    return " ".join([stemmer.stem(word) for word in text.split()])


def normalize(text, uncase=True):
    """Normalizes English text"""
    text = text.strip()
    if uncase:
        text = text.lower()
    # remove extra spaces
    text = re.sub(' +', ' ', text)
    # remove html tags
    text = re.sub(re.compile('<.*?>'), ' ', text)
    # remove twitter hastags, usernames, web addresses
    text = re.sub(r"#[\w\d]*|@[.]?[\w\d]*[\'\w*]*|https?:\/\/\S+\b|"
                  r"www\.(\w+\.)+\S*|", '', text)
    # convert accented characters to ASCII equivalents
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    # strip repeated chars (extra vals)
    text = re.sub(r'(.)\1+', r"\1\1", text)
    # separate punctuation from words and remove not included marks
    text = " ".join(re.findall(r"[\w']+|[\.?!,;:]", text))
    # remove underscores
    text = text.replace('_', ' ')
    # remove double quotes
    text = text.strip('\n').replace('\"', '')
    # remove single quotes
    text = text.replace("'", '')
    # remove extra spaces
    text = re.sub(' +', ' ', text)
    return text


def get_ids(tokens, gensim_model, max_seq_len):
    """Transform tokens to their corresponding ids in the gensim model vocabulary"""
    input_ids = [gensim_model.vocab[token].index if token in gensim_model.vocab else UNKNOWN_IDX for token in tokens]
    input_ids = input_ids + [PADDING_IDX] * (max_seq_len - len(input_ids))
    return np.array(input_ids)


def trim_text(text, max_seq_length):
    """Trims the text to the longest possible number of tokens <= max_seq_length that includes complete sentences
    Parameters
    ----------
    text: str
        the text to perform the trimming on
    max_seq_length: int
        the maximum number of tokens desired after trimming

    Returns
    -------
    str
        The trimmed text
    """
    trimmed_text = None
    valid_index = -1
    sentences = re.split("[\.,!?]", text)
    tokenized_sentences = [sentence.split() for sentence in sentences]
    accumulated_length = list(accumulate([len(sentence) for sentence in tokenized_sentences]))
    for i, ac_length in enumerate(accumulated_length[::-1]):
        if ac_length <= max_seq_length:
            trimmed_text = list(chain.from_iterable(tokenized_sentences[:len(tokenized_sentences) - i]))
            break

    if trimmed_text is None or len(trimmed_text) - max_seq_length >= 10:
        trimmed_text = tokenized_sentences[0][:max_seq_length]

    return trimmed_text


def trim_from_middle(text, max_seq_length):
    """Trims the text to max_seq_length number of tokens by extracting from both ends of the text
    Parameters
    ----------
    text: str
        the text to perform the trimming on
    max_seq_length: int
        the maximum number of tokens desired after trimming

    Returns
    -------
    str
        The trimmed text
    """
    begining_segment = trim_text(text, max_seq_length // 2)
    second_segment_length = min(max_seq_length, len(text.split())) - len(begining_segment)
    ending_segment = " ".join(trim_text(text[::-1], second_segment_length))[::-1].split()

    return begining_segment + ending_segment


def generate_balanced_batch(features, labels, batch_size,
                            shuffle, percentages=None):
    """Generates batches with the same class distribution as in percentages or balanced if None by oversampling miniori-
    ty classes
    Parameters
    ----------
    features: pandas.Series/pandas.DataFrame
        the feature column(S)
    labels: pandas.Series/pandas.DataFrame
        the labels column
    batch_size: int
        the number of samples per batch to generate
    shuffle: bool
        whether to shuffle the data before generating the batches or not
    percentages: list or dict
        a list/dict where the value of index C is the desired percentage of class C in the generated batches Pc
        where 0 < Pc < 1
    """
    classes = np.unique(labels)
    class_to_ids = {}
    class_pointers = {}

    if percentages is None:
        percentages = {iclass: 1.0 / len(classes) for iclass in classes}

    for iclass in classes:
        class_ids_list = labels[labels == iclass].index.to_list()
        class_to_ids[iclass] = class_ids_list
        class_pointers[iclass] = 0
        if shuffle:
            np.random.shuffle(class_to_ids[iclass])

    class_visited_flag = [False] * 3
    while True:
        if np.all(class_visited_flag):
            return
        batch_indices = []
        for iclass in classes:
            class_batch_size = 0
            if iclass == classes[-1]:
                class_batch_size = batch_size - len(batch_indices)
            else:
                class_batch_size = int(batch_size * percentages[iclass])

            current_loc = class_pointers[iclass]
            next_loc = current_loc + class_batch_size
            class_instances_num = len(class_to_ids[iclass])
            if next_loc <= class_instances_num:
                batch_indices += class_to_ids[iclass][current_loc:next_loc]
                class_pointers[iclass] = next_loc % class_instances_num
            else:
                batch_indices += class_to_ids[iclass][current_loc:next_loc]
                batch_indices += class_to_ids[iclass][0: next_loc % class_instances_num]
                class_pointers[iclass] = next_loc % class_instances_num
            if next_loc >= class_instances_num:
                class_visited_flag[iclass] = True
        yield features[batch_indices], labels[batch_indices]


def input_memmap_batch_generator(file, batch_size=1, shuffle_and_repeat=True):
    """Loads the data from a memap and returns a batch iterator over them
    Parameters
    ----------
    file: str
        the base_name of the files containing the data and it's meta
        where the data file is {base_name}.dat and the meta is {base_name}_meta.npy
    batch_size: int, default 1
        the number of samples per batch if the data is in 3D according to the meta
    shuffle_and_repeat: bool, default False
        A flag to indicate whether to shuffle and repeat the dataset after exhaustion indefinitely or not

    Returns
    -------
    iterator
        An iterator over the batches of the data
    tuple
        A tuple containing the shape of the data
    """
    data_file = "{}.dat".format(file)
    shape_file = "{}_meta.npy".format(file)
    data_shape = tuple(np.load(shape_file))

    def batch_generator(data_file, data_shape, batch_size=1, shuffle_and_repeat=True):
        f = np.memmap(data_file, dtype=np.int32, mode='r+',
                      shape=data_shape)

        while True:
            if len(data_shape) == 4:
                indexes = list(range(0, data_shape[0], 1))
                if shuffle_and_repeat:
                    np.random.shuffle(indexes)
                for idx in indexes:
                    yield f[idx, :, 0, :], f[idx, :, 1, 0], f[idx, :, 2, 0]
            elif len(data_shape) == 3:
                indexes = list(range(0, data_shape[0], batch_size))
                if shuffle_and_repeat:
                    np.random.shuffle(indexes)
                for idx in indexes:
                    end_idx = idx + batch_size
                    yield f[idx:end_idx, 0, :], f[idx:end_idx, 1, 0], f[idx:end_idx, 2, 0]

    return batch_generator(data_file, data_shape, batch_size, shuffle_and_repeat), data_shape
