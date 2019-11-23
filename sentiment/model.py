import functools
import logging
import tensorflow as tf


def define_scope(function):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope().
    """
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class SentModel:

    def __init__(self, word_embeddings_mat, max_sentence_len=42,
                 learning_rate=.001, num_hidden=200, restore_file=None):
        """Initializes the NLM model instance with the required variables

        Parameters
        ----------
        word_embeddings_mat : a float matrix of shape [word vocab size X word
            embeddings size] an embeddings matrix to be used by the model, if
            no pre-trained embeddings matrix already exist this should be
            passed a randomly initialized embeddings matrix
            
        max_sentence_len : int
            the maximum length of sentences accepted by the model

        learning_rate : float
            the learning rate to be used by the model for training

        num_hidden : int
            the number of hidden output cells by the bi-directional GRU layers

        restore_file : str
            the file name of a pre-trained model to load it's weights before
            using it for inference or resume training


        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            # initializing instance's attributes
            self.max_sentence_len = max_sentence_len
            self.learning_rate = learning_rate
            self.num_hidden = num_hidden
            self.word_embeddings_mat = word_embeddings_mat
            self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
            # initializing the tensorflow's model needed placeholders
            self.true_labels = tf.placeholder(tf.int32, shape=(None,))
            self.word_ids = tf.placeholder(tf.int32, shape=(None, None))
            self.sentence_lengths = tf.placeholder(tf.int32, shape=(None,))
            self.keep_prob = tf.placeholder(tf.float32)
            # building the rest of the model
            self.logits
            self.optimize
            # initializing the model's instance to be used and restoring the
            # weights from a file if passed with a value
            self.current_sess = tf.Session()
            init_g = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            if restore_file:
                saver = tf.train.Saver()
                saver.restore(self.current_sess, restore_file)
            else:
                self.current_sess.run(init_g)
                self.current_sess.run(init_l)

    @define_scope
    def logits(self):
        """ builds a subset of the model's graph that is responsible for the
            calculation of the output probabilities
        """
        batch_size = tf.shape(self.word_ids)[0]

        with tf.device('/cpu:0'), tf.name_scope('embedding_layer'):
            # EMBEDDINGS LAYER #
            # INPUT : ids of BATCH_SIZE X NUM_WORDS
            # OUTPUT : embeddings of BATCH_SIZE X NUM_WORDS X EMBEDDINGS_SIZE
            word_embeddings = tf.get_variable(name="Embeddings_mat_word",
                                              shape=self.word_embeddings_mat.shape,
                                              initializer=tf.constant_initializer(
                                                  self.word_embeddings_mat),
                                              trainable=False)

            final_word_embeddings = tf.nn.embedding_lookup(word_embeddings, self.word_ids)

        # create 2 GRUcells
        rnn_layers = [tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.GRUCell(
                size,
                kernel_initializer=tf.contrib.layers.xavier_initializer(False),
                bias_initializer=tf.zeros_initializer(),
            ),
            output_keep_prob=self.keep_prob)
            for size in [self.num_hidden] * 2]

        # create a bi-directional rnn  composed of 2 stacked-GRUcells for each direction

        outputs, state_fw, state_bw = tf.contrib.rnn.stack_bidirectional_rnn(
            cells_fw=rnn_layers,
            cells_bw=rnn_layers,
            inputs=tf.unstack(tf.transpose(final_word_embeddings,
                                           perm=[1, 0, 2]),
                              num=self.max_sentence_len),
            sequence_length=self.sentence_lengths,
            dtype=tf.float32)

        final_rnn_output = tf.concat([state_fw[-1], state_bw[-1]], -1)

        final_output = tf.layers.dense(
            final_rnn_output,
            100,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(
                mean=0, stddev=1 / self.num_hidden)
        )

        logits = tf.layers.dense(
            final_output,
            3,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(
                mean=0, stddev=2 / (self.num_hidden + 2))
        )

        self.probas = tf.nn.softmax(logits)
        prediction = tf.argmax(self.probas, 1, output_type=tf.int32)
        equality = tf.equal(prediction, self.true_labels)
        self.acc = tf.reduce_mean(tf.cast(equality, tf.float32))

        return logits

    @define_scope
    def optimize(self):
        """defines the loss and the optimization operation to update the
           model's parameters.
        """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.true_labels, logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var)
                      for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
        return train_op

    def train(self, train_steps_per_epoch, val_steps_per_epoch, train_batch_generator, epochs_num, keep_prob,
              val_batch_generator=None, resume_training=False, save_model_file=None,
              n_checkpoints=0):
        """trains the model with the given input
        Parameters
        ----------
        train_steps_per_epoch : int
            the numbers of batches to be trained per epoch
        val_steps_per_epoch: int
            the number of validation batches to consume during evaluation
        train_batch_generator : a generator
            The output of the generator must be a tuple of (ids, sentence_lens, labels)
            The generator is expected to loop over its data indefinitely.
            An epoch finishes when train_steps_per_epoch batches been seen by the
                model.
        epochs_num : int
            the number of epochs to train the model for
        keep_prob : float 0 <= keep_prob <= 1
            the keep probability for the dropout layer wrapping the GRU layer
        val_batch_generator : a generator
            same as batch_generator only it's now used to generate validation
                data
        resume_training : bool
            controls if the variables should be initialized before training or
                not
        save_model_file : str
            the file name to save the model in after finishing training
        n_checkpoints : int
            the number of lastest checkpoint to keep
        """
        with self.graph.as_default():
            init_g = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            if (not resume_training):
                self.current_sess.run(init_g)
                self.current_sess.run(init_l)
            saver = tf.train.Saver()

            best_loss = 9999999999
            best_acc = 0
            checkpoint_num = 0
            logging.info("############## STARTED TRAINING ##############")
            for i in range(epochs_num):
                avg_train_cost = 0
                for j in range(train_steps_per_epoch):
                    X_word, sentence_lens, y = next(
                        train_batch_generator)

                    _, train_cost = self.current_sess.run(
                        [self.optimize, self.loss],
                        feed_dict={self.true_labels: y,
                                   self.word_ids: X_word,
                                   self.sentence_lengths: sentence_lens,
                                   self.keep_prob: keep_prob})
                    avg_train_cost += train_cost
                    if j % 10 == 0:
                        logging.info("Train step #{0}, Train loss: {1}".format(j, train_cost))

                avg_train_cost = avg_train_cost / train_steps_per_epoch
                self.history['train_loss'] += [avg_train_cost]
                if (val_batch_generator):
                    avg_val_acc, avg_val_cost = 0, 0
                    for j in range(val_steps_per_epoch):
                        X_word_test, sentence_lens_test, y_test = next(val_batch_generator)
                        val_cost, val_acc = self.current_sess.run(
                            [self.loss, self.acc],
                            feed_dict={self.true_labels: y_test,
                                       self.word_ids: X_word_test,
                                       self.sentence_lengths: sentence_lens_test,
                                       self.keep_prob: 1})
                        avg_val_acc += val_acc
                        avg_val_cost += val_cost

                    avg_val_acc = avg_val_acc / val_steps_per_epoch
                    avg_val_cost = avg_val_cost / val_steps_per_epoch

                    self.history['val_loss'] += [avg_val_cost]
                    self.history['val_acc'] += [avg_val_acc]
                    if (n_checkpoints > 0 and (best_loss > avg_val_cost
                                               or best_acc < avg_val_acc)):
                        if best_acc < avg_val_acc:
                            best_acc = avg_val_acc

                        if best_loss > avg_val_cost:
                            best_loss = avg_val_cost

                        saver.save(
                            self.current_sess,
                            save_model_file + "--{}--.ckpt".format(
                                checkpoint_num % n_checkpoints))
                        checkpoint_num += 1

                    logging.info('Epoch #{} --- train_loss : {}, val_loss : {}, '
                                 'val_acc : {}'.format(
                        i + 1, avg_train_cost, avg_val_cost, avg_val_acc))
                else:
                    logging.info("Epoch #{} --- train_loss : {}".format(
                        i + 1, avg_train_cost))

            if (save_model_file):
                saver.save(self.current_sess, str(save_model_file) + ".ckpt")

    def infer(self, X_word, sentence_lens):
        """predicts the class of the input utterance if either it belongs to
            the presumed generative language model or not

        Parameters
        ----------
        X_word : 2d array of size [BATCH_SIZE, MAX_SENTENCE_LEN]
            contains all the training set transformed utterances input words
            
        sentence_lens : 1d array of size [BATCH_SIZE]
            contains the actual lengths of all the utterances before padding

        Returns
        -------
        bool array of shape [BATCH_SIZE , 2]
            the probabilities for each class for every utterance in the batch
        """
        with self.graph.as_default():
            word_probs = self.current_sess.run(
                self.probas,
                feed_dict={
                    self.word_ids: X_word,
                    self.sentence_lengths: sentence_lens,
                    self.keep_prob: 1
                })

            return word_probs

    def save(self, filename):
        """saves the current session model's parameters

        Parameters
        ----------
        filename : str
            the file to save the model's parameters in
        """
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.current_sess, str(filename) + ".ckpt")

    def load(self, filename):
        """loads the session model's parameters from a file

        Parameters
        ----------
        filename : str
            the file to load the model's parameters from
        """
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.current_sess, filename + ".ckpt")
