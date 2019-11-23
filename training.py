import argparse
import logging
import sys
import math
import gensim
import numpy as np

from sentiment.model import SentModel
from sentiment.input_utils import input_memmap_batch_generator

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO, filename='training_run_1.log')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Training script""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('embeddings_file', type=str,
                        help="gensim compatible word2vec format file that contains the pre-trained embeddings")
    parser.add_argument('save_model_file', type=str,
                        help='The file to save the model to')
    parser.add_argument('train_file_base', type=str,
                        help='The base_file name to load the training data and meta')
    parser.add_argument('val_file_base', type=str,
                        help='The base_file name to load the validation data and meta')
    parser.add_argument('--batch_size', type=int, default=-1,
                        help='The number of samples per batch if input data is 3D (not pre-generated)')
    parser.add_argument('--lr', type=float, default=.001,
                        help='The learning rate used during training')
    parser.add_argument('--keep_prob', type=float, default=1,
                        help='The keep probability for the dropout layer wrapping the GRU layer')
    parser.add_argument('--num_hidden', type=int, default=100,
                        help='the number of hidden output cells by the bi-directional GRU layers')
    parser.add_argument('--epochs_num', type=int, default=5,
                        help='The number of training epochs')
    parser.add_argument('--checkpoints_num', type=int, default=1,
                        help='The number of checkpoints to save')
    parser.add_argument('--restore_file', type=str, default="",
                        help='The file to restore the weights from')

    args = parser.parse_args()

    logging.info("Loading word embeddings matrix.")
    gensim_model = gensim.models.KeyedVectors.load_word2vec_format(args.embeddings_file)
    word_vectors = gensim_model.syn0

    train_batcher, training_data_shape = input_memmap_batch_generator(args.train_file_base)
    if args.batch_size == -1:
        steps_per_training = training_data_shape[0]
        batch_size = training_data_shape[1]
        max_seq_len = training_data_shape[3]

    val_batcher, validation_data_shape = input_memmap_batch_generator(args.val_file_base)

    steps_per_validation = math.ceil(validation_data_shape[0] / batch_size)

    logging.info("Instantiating Sentiment Model object.")
    sentiment_model = SentModel(word_embeddings_mat=word_vectors,
                                max_sentence_len=max_seq_len,
                                learning_rate=args.lr, num_hidden=args.num_hidden,
                                restore_file=args.restore_file)
    logging.info("Model training has started...")
    sentiment_model.train(train_steps_per_epoch=steps_per_training,
                          val_steps_per_epoch=steps_per_validation,
                          keep_prob=args.keep_prob, train_batch_generator=train_batcher,
                          val_batch_generator=val_batcher, resume_training=bool(args.restore_file),
                          save_model_file=args.save_model_file,
                          epochs_num=args.epochs_num, n_checkpoints=args.checkpoints_num)

    logging.info("Model training has finished.")
