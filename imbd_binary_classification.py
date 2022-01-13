"""
This script uses deep learning libary to classify movie
reviews as either positive or negative.
"""
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb

from keras import models
from keras import layers


def vectorize_sequences(sequences, dimension=10000):
    """
    one hot encodes sequences. Input data is 2D array
    """
    results = np.zeros(shape=(len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        # set specific indices(in sequences) of results[i] to 1
        results[i, sequence] = 1
    return results

def plot_results_loss(hist_dict):
    """
    """
    loss_vals = hist_dict['loss']
    val_loss_values = hist_dict['val_loss']
    epochs = range(1, 21)
    plt.plot(epochs, loss_vals, 'bo', label='Training_loss')
    plt.plot(epochs, val_loss_values, 'b', label='validation_loss')
    plt.title('Training and Validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('training_and_validation_loss.png')

def plot_results_acc(hist_dict):
    """
    Parameters
    ------------
    hist_dict: Dictionary[str, ?]
        results of training model

    Returns
    ------------
    None
    """
    epochs = 20
    acc_vals = hist_dict['acc']
    validation_acc_vals = hist_dict['val_acc_vals']
    plt.plot(epochs, acc_vals, 'bo', label+"training accuracy")
    plt.plot(epochs, validation_acc_vals, 'b', label='validation accuracy')
    plt.title('Training and Validation Accurary')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('training_and_validation_accuracy.png')

def main(validation):
    """
    Parameters
    ----------------
    validation: Bool
        if True, model runs validation
        else model will fit using full train set and evaluate results
        on test data

    Returns
    -----------------
    None
    """
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    word_index = imdb.get_word_index()
    reverse_wi = {(value, key) for (key, value) in word_index.items()}

    # one-hot encode each review to a 10,000 length vector of 0 and 1
    X_train = vectorize_sequences(train_data)
    X_test = vectorize_sequences(test_data)
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    model = models.Sequential()
    model.add(layers.Dense(
        units=16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(units=16, activation='relu'))
    # sigmoid useful for probabilities
    model.add(layers.Dense(units=1, activation='sigmoid'))
    # binary crossentropy useful for loss function of
    # binary classification.
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    validation_size = 10000
    X_val = X_train[:validation_size]
    partial_x_train = X_train[validation_size:]
    y_val = y_train[:validation_size]
    partial_y_train = y_train[validation_size:]
    if validation:
        history = model.fit(x=partial_x_train,
                            y=partial_y_train,
                            batch_size = 512,
                            epochs=20,
                            validation_data=(X_val, y_val))
        history_dict = history.history
        print(history_dict)
        plot_results_loss(history_dict)
        plot_results_acc(history_dict)
        # inspecting the plots shows that the model begins to overfit
        # after four epochs
        # retrain model with four epochs and check how it performs on the
        # test data.
    else:
        model.fit(X_train, y_train, batch_size=512, epochs=4)
        results = model.evaluate(X_test, y_test)
        with open('results.txt', 'w') as fh:
            for result in results:
                  fh.write(f"{result}\n")

if __name__ == "__main__":
    main(validation=False)

