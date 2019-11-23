import argparse
import sys
import logging
import math
import numpy as np
import gensim

from sklearn.metrics import classification_report
from sentiment.model import SentModel
from sentiment.input_utils import input_memmap_batch_generator

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO, filename='testing_run_`.log')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Testing script""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('embeddings_file', type=str,
                        help="gensim compatible word2vec format file that contains the pre-trained embeddings")
    parser.add_argument('model_file', type=str,
                        help='The file to save to load the model from')
    parser.add_argument('test_file_base', type=str,
                        help='The base_file name to load the testing data and meta')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='the number of samples per testing batch')
    parser.add_argument('--num_hidden', type=int, default=100,
                        help='the number of hidden output cells by the bi-directional GRU layers')

    args = parser.parse_args()

    logging.info("Loading word embeddings matrix.")
    gensim_model = gensim.models.KeyedVectors.load_word2vec_format(args.embeddings_file)
    word_vectors = gensim_model.syn0

    test_batcher, testing_data_shape = input_memmap_batch_generator(args.test_file_base)

    max_seq_len = testing_data_shape[-1]
    steps_per_testing = math.ceil(testing_data_shape[0] / args.batch_size)

    logging.info("Instantiating Sentiment Model object.")
    sentiment_model = SentModel(word_embeddings_mat=word_vectors,
                                max_sentence_len=max_seq_len, num_hidden=args.num_hidden,
                                restore_file=args.model_file)

    train_batcher, testing_data_shape = input_memmap_batch_generator(args.test_file_base, args.batch_size)

    batches_num = math.ceil(testing_data_shape[0] / args.batch_size)

    logging.info("Model testing has started...")
    predicted, true = [], []
    for i in range(batches_num):
        ids, lens, labels = next(train_batcher)
        true += list(labels)
        pl = sentiment_model.infer(ids, lens)
        prediction_class = pl.argmax(axis=1)
        predicted += list(prediction_class)
        if i % 10 == 0:
            logging.info("Processed {:.2f}% of the testing dataset".format(i / batches_num))

    print(classification_report(true, predicted))
