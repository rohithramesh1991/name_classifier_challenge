import fire
import zipfile
import os
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.metrics import Precision, Recall
from keras.optimizers import Adam
from keras.models import load_model
from keras.regularizers import l2
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ignore tensorflow warnings
import warnings
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#
def extract_ttl_files(data_folder: str):

    """ creates a path to the zip file.
    And extracts the zip file in the said path.
    """

    person_zip_path = os.path.join(data_folder, 'person.ttl.zip')
    name_zip_path = os.path.join(data_folder, 'name.ttl.zip')

    with zipfile.ZipFile(person_zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_folder)

    with zipfile.ZipFile(name_zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_folder)

    print(f"Extracted files to {data_folder}")

def load_data(in_folder: str):

    """
    Takes the folder where the zip file exists as input.
    Calls the extract_ttl_files function to unzip if not already unzipped and then reads the data.
    Also, Merges the name.ttl with person.ttl to obtain the y values (0 or 1).
    return a dataframe
    """

    # Paths to the.ttl files
    person_ttl_path = os.path.join(in_folder, 'person.ttl')
    name_ttl_path = os.path.join(in_folder, 'name.ttl')

    # If the .ttl files does not exist, they need to be extracted from the .zip files
    if not os.path.exists(person_ttl_path) or not os.path.exists(name_ttl_path):
        extract_ttl_files(in_folder)

    names_to_uris = {}
    people_uris = set()

    # Load the name.ttl file and add the names_to_uris dictionary
    with open(name_ttl_path, 'r', encoding='utf-8') as file:
        for line in file:
            start_index = line.find('"') + 1
            end_index = line.find('"@en', start_index)
            name = line[start_index:end_index]
            uri = line[line.find('<') + 1:line.find('>')]
            names_to_uris[name] = uri

    # Load the person.ttl file and add the people_uris set
    with open(person_ttl_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split()
            uri = parts[0].strip('<>')
            people_uris.add(uri)

    # Combine the data to create labels and dataset
    # If the URI associated with a name is in the set of people URIs, label it as 1, otherwise 0
    data = []
    labels = []
    for name, uri in names_to_uris.items():
        if uri in people_uris:
            label = 1
        else:
            label = 0
        data.append(name)
        labels.append(label)


    df = pd.DataFrame({
        'name': data,
        'label': labels
    })

    return df

# (2602762, 2)
# label
# 0    1895792
# 1     706970

def resample_dataset(df, label_column='label'):

    """
    Resamples the dataset to balance the classes by undersampling the majority class.
    Returns:
    - A pandas DataFrame with a balanced class distribution.
    - A sample pandas DataFrame for prediction.
    - remove this prediction sample from resampled df.
    """

    df_majority = df[df[label_column] == 0]
    df_minority = df[df[label_column] == 1]

    df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)

    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Resampled class distribution:\n", df_balanced[label_column].value_counts())

    df_predict_samples = sample_for_prediction(df_balanced)

    df_resampled = df_balanced.drop(df_predict_samples.index)

    print("Resampled class distribution after taking out predict sample:\n", df_resampled[label_column].value_counts())
    print("prediction data set sample class distribution:\n", df_predict_samples[label_column].value_counts())

    return df_resampled, df_predict_samples

# (1413940, 2)
# label
# 0    706970
# 1    706970

def sample_for_prediction(df, n_samples=20):
    """
    Randomly samples to use for prediction after training

    Returns:
    - A pandas DataFrame containing the random samples.
    """
    df_sampled = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    return df_sampled

def clean_name(name):

    """Remove numbers.
    Removes hyphens (-), en dashes (–), em dashes (—), parentheses
    and any other special characters that are not required.
    Retained apostrophe and diacritics"""

    name = re.sub(r'\d+', '', name)
    name = re.sub(r'[-–—(){}[\]]', ' ', name)
    name = re.sub(r'[<>:;,_\"!@#$%^&*=+|\\/?~`]', '', name)
    name = name.lower()
    name = re.sub(r'\s+', ' ', name)
    name = name.strip()
    return name

def tokenize_and_pad(df, text_col='cleaned_name'):
    """
    Tokenizes and pads the text data.
    Returns:
    - Tuple of padded sequences, vocabulary size, and maximum sequence length.
    """
    tokenizer = Tokenizer(char_level=True, oov_token='UNK')
    tokenizer.fit_on_texts(df[text_col])
    sequences = tokenizer.texts_to_sequences(df[text_col])

    # Calculate the 95th percentile of input sequence lengths
    max_seq_length = int(np.percentile([len(seq) for seq in sequences], 95))

    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')
    vocab_size = len(tokenizer.word_index) + 1

    return padded_sequences, vocab_size, max_seq_length, tokenizer


def split_data(data):
    """
    calls Tokenize function and then splits the dataset into training and test sets.
    Returns:
    - X_train, X_test, y_train, y_test , vocab size & max length
    """

    data.drop(['name'], axis=1, inplace=True)

    padded_sequences, vocab_size, max_seq_length, tokenizer = tokenize_and_pad(data)

    labels = data['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, vocab_size, max_seq_length, tokenizer

def evaluate_model(model, test_data):
    X_test, y_test = test_data
    evaluation = model.evaluate(X_test, y_test)
    print(f"Test Loss: {evaluation[0]}\nTest Accuracy: {evaluation[1]}\nPrecision: {evaluation[2]}\nRecall: {evaluation[3]}")
    return evaluation

def save_model(model, out_folder: str):

    """
    Saves the model weights to an output folder,
    create the folder if it does not present.
    """

    model_dir = os.path.join(out_folder, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, 'model.h5')

    model.save(model_path)
    print(f"Model saved to {model_path}")


def save_tokenizer_and_max_seq_length(tokenizer, max_seq_length, out_folder: str):
    """
    Saves the tokenizer and max sequence.
    """
    model_dir = os.path.join(out_folder, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    file_path = os.path.join(model_dir, 'tokenizer_and_max_seq_length.pkl')

    with open(file_path, 'wb') as file:
        pickle.dump({'tokenizer': tokenizer, 'max_seq_length': max_seq_length}, file)

    print(f"Tokenizer and max sequence length saved to {file_path}")


def save_predict_samples(df_predict_samples, out_folder: str):
    data_dir = os.path.join(out_folder, 'prediction')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Define the file path for saving
    file_path = os.path.join(data_dir, 'predict_input_samples.csv')

    # Save DataFrame
    df_predict_samples.to_csv(file_path, index=False)
    print(f"Predict samples saved to {file_path}")


def train(in_folder: str, out_folder: str) -> None:

    """
    Processes the data from the input folder, trains a model, and evaluates it.
    """

    df = load_data(in_folder)
    df_resampled, df_predict_samples = resample_dataset(df)
    df_resampled['cleaned_name'] = df_resampled['name'].apply(clean_name)
    X_train, X_test, y_train, y_test, vocab_size, max_seq_length, tokenizer = split_data(df_resampled)

    # Model definition
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=50, input_length=max_seq_length),
        Dropout(0.2),
        LSTM(64, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    # Model compilation
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy', Precision(), Recall()])

    # Model training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.001)
    ]

    model.fit(X_train, y_train, epochs=10, batch_size=512, validation_split=0.2, callbacks=callbacks)

    # Model evaluation
    test_data = (X_test, y_test)
    evaluate_model(model, test_data)

    #save model

    save_model(model, out_folder)
    save_tokenizer_and_max_seq_length(tokenizer, max_seq_length, './')
    save_predict_samples(df_predict_samples, './')
# #
# train('data','./')

## uncomment the below lines and comment the train function inorder to load saved model and predict on unseen sample ##
#
# def consolidated_load_and_predict(out_folder: str):
#     model_dir = os.path.join(out_folder, 'models')
#     data_dir = os.path.join(out_folder, 'prediction')
#
#     # Load the model
#     model_path = os.path.join(model_dir, 'model.h5')
#     model = load_model(model_path)
#
#     # Load the tokenizer and max_seq_length
#     tokenizer_max_seq_length_path = os.path.join(model_dir, 'tokenizer_and_max_seq_length.pkl')
#     with open(tokenizer_max_seq_length_path, 'rb') as file:
#         tokenizer_data = pickle.load(file)
#     tokenizer = tokenizer_data['tokenizer']
#     max_seq_length = tokenizer_data['max_seq_length']
#
#     # Load the prediction samples
#     predict_samples_path = os.path.join(data_dir, 'predict_input_samples.csv')
#     df_predict_samples = pd.read_csv(predict_samples_path)
#
#     # Preprocess the names
#     df_predict_samples['cleaned_name'] = df_predict_samples['name'].apply(clean_name)
#     sequences = tokenizer.texts_to_sequences(df_predict_samples['cleaned_name'])
#     padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')
#
#     # Predict
#     predictions = model.predict(padded_sequences)
#
#     # Convert predictions to binary using 0.5 as the threshold
#     predicted_labels = (predictions > 0.5).astype(int)
#
#     # Append predictions to the DataFrame
#     df_predict_samples['Predicted Label'] = predicted_labels
#
#     # Print predictions alongside actual labels for comparison
#     for index, row in df_predict_samples.iterrows():
#         print(f"Name: {row['name']}, Actual Label: {row['label']}, Predicted Label: {row['Predicted Label']}")
#
#     # Save the updated DataFrame with predictions
#     output_path = os.path.join(data_dir, 'prediction_results.csv')
#     df_predict_samples.to_csv(output_path, index=False)
#     print(f"Prediction results saved to {output_path}")
#
# consolidated_load_and_predict('./')

if __name__ == '__main__':
  fire.Fire(train)