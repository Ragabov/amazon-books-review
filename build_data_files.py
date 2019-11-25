import argparse
import sys
import gensim
import numpy as np
import pandas as pd
import logging

from math import ceil
from sklearn.model_selection import train_test_split
from sentiment.input_utils import trim_from_middle, get_ids, generate_balanced_batch, normalize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO, filename='building_data.log')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def group_scores(score):
    """Transforms the overall score to one of three classes"""
    score_to_class_map = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
    return score_to_class_map[score]


def transform_dataframe(df, gensim_model, max_seq_length, uncase=True):
    """Transforms the dataframe's data into compatible shapes and values of the DL model"""
    logging.info("Merging summary and reviewText columns")
    df["merged_text"] = df.apply(lambda x: normalize("{}. {}".format(x["summary"], x["reviewText"]), uncase), axis=1)
    df = df[["overall", "merged_text"]]
    df["overall"] = df["overall"].apply(group_scores)
    df["tokens"] = df["merged_text"].apply(trim_from_middle, args=(max_seq_length,))
    df["ids"] = df["tokens"].apply(get_ids, args=(gensim_model, max_seq_length))
    df["words_num"] = df["tokens"].apply(len)
    print(df.iloc[0])
    return df


def save_dataframe_to_memmap(df, file, max_seq_length):
    """Saves the dataframe data int oa memmap """
    data_shape = (len(df), 3, max_seq_length)
    data_file, meta_file = '{}.dat'.format(file), "{}_meta".format(file)
    f = np.memmap(data_file, dtype=np.int32, mode='w+',
                  shape=data_shape)

    f[:, 0, :] = np.array(df["ids"].to_list())
    f[:, 1, 0] = np.array(df["words_num"].to_list())
    f[:, 2, 0] = np.array(df["overall"].to_list())
    np.save(meta_file, np.array(data_shape))
    return data_file, meta_file


def save_balanced_batches_to_memmap(df, file, batch_size, max_seq_length, uncase=True):
    """Generates balanced batches from the dataframe's data and saves them into a memmap """
    df["merged_text"] = df.apply(lambda x: normalize("{}. {}".format(x["summary"], x["reviewText"]), uncase), axis=1)
    df["overall"] = df["overall"].apply(group_scores)
    batches_num = get_batches_num(df["overall"], batch_size)
    generator = generate_balanced_batch(df["merged_text"], df["overall"], batch_size, True)
    data_file, meta_file = '{}.dat'.format(file), "{}_meta".format(file)

    f = np.memmap(data_file, dtype=np.int32, mode='w+',
                  shape=(batches_num, batch_size, 3, max_seq_length))
    batch_idx = 0
    while True:
        try:
            text, classes = next(generator)
            classes = np.array(classes)
            # Running the tokenizer against the text and applying smart trimming
            tokens = text.apply(trim_from_middle, args=(max_seq_length,))
            # Generating ids from tokens
            ids = np.array(tokens.apply(get_ids, args=(gensim_model, max_seq_length,)).to_list())
            # Generating input lens
            lens = np.array(tokens.apply(len).to_list())

            f[batch_idx, :, 0, :] = ids
            f[batch_idx, :, 1, 0] = lens
            f[batch_idx, :, 2, 0] = classes

            if batch_idx % 1000 == 0:
                logging.info(
                    "Finished transforming {0:.2f}% batches of the training data".format(batch_idx / batches_num))
            batch_idx += 1
        except StopIteration:
            np.save(meta_file, np.array((batches_num, batch_size, 3, max_seq_length)))
            break


def get_batches_num(labels, batch_size, percentages=None):
    """Calculates the number of balanced batches after oversampling the miniority classes"""
    classes, counts = np.unique(labels, return_counts=True)
    max_class = np.argmax(counts)
    if percentages is None:
        percentages = [1.0 / len(classes) for iclass in classes]

    class_count_per_batch = [int(percentage * batch_size) for percentage in percentages]
    max_class_batch_count = int(class_count_per_batch[max_class] + (batch_size - np.sum(class_count_per_batch)))
    return ceil(counts[max_class] / max_class_batch_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Building data script""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('word2vec_file', type=str,
                        help="gensim compatible word2vec format file that contains the pre-trained embeddings")
    parser.add_argument('data_json', type=str,
                        help='The JSON file containing the data')
    parser.add_argument('train_file_base', type=str,
                        help='The base_file name to save the training data batches and meta')
    parser.add_argument('val_file_base', type=str,
                        help='The base_file name to save the validation data and meta')
    parser.add_argument('test_file_base', type=str,
                        help='The base_file name to save the testing data and meta')
    parser.add_argument('--uncase', type=bool, default=True,
                        help='Whether to uncase the text during normalization or not')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='The number of samples per training batch')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='The maximum number of tokens per sample text')

    args = parser.parse_args()

    logging.info("Loading word embeddings gensim model.")
    gensim_model = gensim.models.KeyedVectors.load_word2vec_format(args.word2vec_file)

    logging.info("Loading the dataset from {}".format(args.data_json))
    df = pd.read_json(args.data_json, lines=True)

    logging.info("Splitting the data into training, validation, testing")
    df_train, df_test = train_test_split(df, test_size=.3, stratify=df["overall"],
                                         random_state=12)
    df_train, df_valid = train_test_split(df_train, test_size=.125, stratify=df_train["overall"],
                                          random_state=12)

    logging.info("#Training Samples : {}, #Validation Samples : {}, #Testing Samples {}"
                 .format(len(df_train), len(df_valid), len(df_test)))
    logging.info("Transforming validation and testing datasets")

    df_valid = transform_dataframe(df_valid, gensim_model, args.max_seq_length, args.uncase)
    df_test = transform_dataframe(df_test, gensim_model, args.max_seq_length, args.uncase)
    dfile, mfile = save_dataframe_to_memmap(df_valid, args.val_file_base, args.max_seq_length)
    logging.info("Finished transforming and saving validation data to {} and it's meta to {} ".format(dfile, mfile))
    dfile, mfile = save_dataframe_to_memmap(df_test, args.test_file_base, args.max_seq_length)
    logging.info("Finished transforming and saving testing data to {} and it's meta to {} ".format(dfile, mfile))

    logging.info("Transforming and saving the training dataset balanced batches")
    save_balanced_batches_to_memmap(df_train, args.train_file_base, args.batch_size, args.max_seq_length, args.uncase)
    logging.info("Finished.")
