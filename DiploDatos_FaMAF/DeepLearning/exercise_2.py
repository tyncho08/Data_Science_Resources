"""Exercise 2"""
import argparse
import keras.backend as K
import pandas
import pickle
import keras
from keras.layers import Embedding, Average, Lambda, Input, Flatten
from keras.models import Sequential
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from utils import FilteredFastText
from keras import optimizers, regularizers
from keras.utils import plot_model
from keras.layers import Activation, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')


def read_args():
    parser = argparse.ArgumentParser(description='Exercise 2')
    parser.add_argument('--num_units', nargs='+', default=[100], type=int,
                        help='Number of hidden units of each hidden layer.')
    parser.add_argument('--dropout', nargs='+', default=[0.5], type=float,
                        help='Dropout ratio for every layer.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of instances in each batch.')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment, used in the filename'
                             'where the results are stored.')
    parser.add_argument('--embeddings_filename', type=str,
                        help='Name of the file with the embeddings.')
    args = parser.parse_args()
    assert len(args.num_units) == len(args.dropout)
    return args


def load_dataset():
    dataset = load_files('dataset/txt_sentoken', shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=42)
    print('Training samples {}, test_samples {}'.format(
        len(X_train), len(X_test)))
    return X_train, X_test, y_train, y_test


def transform_input(instances, mapping):
    """Replaces the words in instances with their index in mapping.
    Args:
        instances: a list of text instances.
        mapping: an dictionary from words to indices.
    Returns:
        A matrix with shape (n_instances, max_text_length)."""
    word_indices = []
    for instance in instances:
        word_indices.append([mapping[word.decode('utf-8')]
                             for word in instance.split()])
    # Check consistency
    assert len(instances[0].split()) == len(word_indices[0])
    # Pad the sequences to obtain a matrix instead of a list of lists.
    from keras.preprocessing.sequence import pad_sequences
    return pad_sequences(word_indices)


def main():
    args = read_args()
    X_train, X_test, y_train, y_test = load_dataset()

    y_test_orginal = y_test

    train_samples = len(X_train)
    test_samples = len(X_test)

    # Converting labels to categorical
    num_classes = 2
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Load the filtered FastText word vectors, using only the vocabulary in
    # the movie reviews dataset
    with open(args.embeddings_filename, 'rb') as model_file:
        filtered_fasttext = pickle.load(model_file)

    # The next thing to do is to choose how we are going to represent our
    # training matrix. Each review must be translated into a single vector.
    # This means we have to combine, somehow, the word vectors of each
    # word in the review. Some options are:
    #  - Take the average of all vectors.
    #  - Take the minimum and maximum value of each feature.
    # All these operations are vectorial and easier to compute using a GPU.
    # Then, it is better to put them inside the Keras model.

    # The Embedding layer will be quite handy in solving this problem for us.
    # To use this layer, the input to the network has to be the indices of the
    # words on the embedding matrix.
    X_train = transform_input(X_train, filtered_fasttext.word2index)
    X_test = transform_input(X_test, filtered_fasttext.word2index)

    print(X_train.shape)
    

    # The input is ready, start the model
    model = Sequential()
    model.add(Embedding(
        filtered_fasttext.wv.shape[0],  # Vocabulary size = 50920
        filtered_fasttext.wv.shape[1],  # Embedding size = 300
        weights=[filtered_fasttext.wv], # Word vectors
        trainable=False  # This indicates the word vectors must not be changed
                         # during training.
    ))
    # The output here has shape
    #     (batch_size (?), words_in_reviews (?), embedding_size)
    # To use a Dense layer, the input must have only 2 dimensions. We need to
    # create a single representation for each document, combining the word
    # embeddings of the words in the intance.
    # For this, we have to use a Tensorflow (K) operation directly.
    # The operation we need to do is to take the average of the embeddings
    # on the second dimension. We wrap this operation on a Lambda
    # layer to include it into the model.
    model.add(Lambda(lambda xin: K.mean(xin, axis=1), name='embedding_average'))

    # Now the output shape is (batch_size (?), embedding_size)

    # Finishing the Keras model
    model.add(Dense(300, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.1))
    model.add(Dense(300, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.1))
    model.add(Dense(250, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.1))
    model.add(Dense(120, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.1))
    model.add(Dense(80, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.1))
    model.add(Dense(40, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(2, activation='sigmoid'))

    # Printing Model Resume
    print(model.summary())

    # Compiling the Model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    plot_model(model, to_file="Modelos/Ex2_Fig_Model.png")

    # Fitting the Model
    #batch_size = 300
    epochs = 300
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.1, verbose=1)

    # List all data in history
    # Summarize history for accuracy
    fig = plt.figure()
    plt.subplot(121)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # Summarize history for loss
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    fig.savefig("Resultados/Ex2_Analisis_Overfitting_{}.png".format(args.experiment_name))

    # Evaluation of the Model
    predictions = model.predict(X_test)
    scores = model.evaluate(X_test, y_test)
    print('\n')
    print('Test Loss:', scores[0])
    print('Test Accuracy:', scores[1])

    f = open("Precision/Ex2_Loss_and_Accuracy_{}.txt".format(args.experiment_name), 'w')
    f.write("Training samples {}, test_samples {}.\n".format(train_samples, test_samples))
    f.write('Epochs: '+str(epochs)+'\n')
    f.write('Test Loss: '+str(scores[0])+'\n')
    f.write('Test Accuracy: '+str(scores[1])+'\n')
    f.close()

    # Saving the Model.
    model.save("Modelos/Ex2_Modelo_{}.h5".format(args.experiment_name))

    predictions_list = []
    for pred in predictions:
        predictions_list.append(np.argmax(pred))

    # Saving the Predictions:
    results = pandas.DataFrame(y_test_orginal, columns=['True_Label'])
    results.loc[:, 'Predicted'] = np.array(predictions_list)
    results.to_csv("Resultados/Ex2_Predicitions_{}.csv".format(args.experiment_name), index=False)


if __name__ == '__main__':
    main()
    K.clear_session()
