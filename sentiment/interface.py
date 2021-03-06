import numpy as np
import joblib

from sentiment.input_utils import normalize, trim_from_middle, get_ids, stem
from sentiment.model import SentModel


class GenericModelInterface:
    def run(self, utterance):
        raise NotImplementedError("{}.run() is not implemented".format(self.__class__.__name__))


class SentModelInterface(GenericModelInterface):
    def __init__(self, gensim_model, max_seq_len, num_hidden, uncase, saved_model_file):
        """"""
        self.gensim_model = gensim_model
        self.uncase = uncase
        self.max_seq_len = max_seq_len
        self.sentiment_model = SentModel(word_embeddings_mat=self.gensim_model.syn0,
                                         max_sentence_len=max_seq_len, num_hidden=num_hidden,
                                         restore_file=saved_model_file)

    def transform_utterance(self, utterance):
        utterance = normalize(utterance, self.uncase)
        utterance_tokens = trim_from_middle(utterance, self.max_seq_len)
        ids = get_ids(utterance_tokens, self.gensim_model, self.max_seq_len)
        tokens_num = len(utterance_tokens)

        return [ids], [tokens_num]

    def run(self, utterance):
        ids, lens = self.transform_utterance(utterance)
        probs = self.sentiment_model.infer(ids, lens)
        return np.argmax(probs)


class VotingEnsembleInterface(GenericModelInterface):
    def __init__(self, classifier_file, preprocessing_pipeline_file, max_seq_len):
        self.model = joblib.load(classifier_file)
        self.preprocesing_pipeline = joblib.load(preprocessing_pipeline_file)
        self.max_seq_len = max_seq_len if max_seq_len > 0 else None

    def run(self, utterance):
        stemmed_normalized_utterance = stem(normalize(utterance))
        if self.max_seq_len is not None:
            stemmed_normalized_utterance = trim_from_middle(stemmed_normalized_utterance, self.max_seq_len)
        inputs = self.preprocesing_pipeline.transform([stemmed_normalized_utterance])
        return self.model.predict(inputs)[0]
