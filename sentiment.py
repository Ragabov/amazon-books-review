import argparse
import gensim
from flask import Flask, request, jsonify
from sentiment.interface import VotingEnsembleInterface, SentModelInterface

app = Flask('sentiment')
sentiment_interface = None
INT_TO_STR_CLASSES_MAP = {0: "negative",    1: "neutral", 2: "positive"}


@app.route("/classify", methods=['GET', 'POST', ])
def hello():
    utterance = request.args['utterance']
    predicted_class = sentiment_interface.run(utterance)
    return jsonify(
        detected_class=INT_TO_STR_CLASSES_MAP[predicted_class]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Run a flask application that serves the classifier""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(help='The type of the model to run the service with', dest='model_type')

    parser_traditional = subparsers.add_parser('traditional', help='Run a traditional ML classifier')
    parser_traditional.add_argument('model_file', type=str,
                        help="The path to the pickled model file")
    parser_traditional.add_argument('preprocessing_file', type=str,
                        help="The path to the pickled preprocessing pipeline file")

    parser_dl = subparsers.add_parser('dl', help='Run a DL classifier')
    parser_dl.add_argument('embeddings_file', type=str,
                        help="gensim compatible word2vec format file that contains the pre-trained embeddings")
    parser_dl.add_argument('model_file', type=str,
                        help='The file to save to load the model from')
    parser_dl.add_argument('uncase', type=bool, default=True,
                        help='Whether to uncase the text during normalization or not')
    parser_dl.add_argument('max_seq_len', type=int, default=128,
                        help='The maximum number of tokens per sample text')
    parser_dl.add_argument('num_hidden', type=int, default=100,
                        help='the number of hidden output cells by the bi-directional GRU layers')
    args = parser.parse_args()

    if args.model_type == "traditional":
        sentiment_interface = VotingEnsembleInterface(args.model_file, args.preprocessing_file)
    elif args.model_type == "dl":
        gensim_model = gensim.models.KeyedVectors.load_word2vec_format(args.embeddings_file)
        sentiment_interface = SentModelInterface(gensim_model, args.max_seq_len,
                                                 args.num_hidden, args.uncase, args.model_file)

    app.run()
