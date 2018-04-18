import csv
import os
import shutil
import socket
import sys

#import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional, Concatenate, Flatten, regularizers
from keras.layers.embeddings import Embedding

csv.field_size_limit(sys.maxsize)
np.random.seed(1)
import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email_report(subject, txt):
    fromaddr = "rreporter33@gmail.com"
    toaddr = "etemad@dal.ca"
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = subject

    body = txt
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "A@12345678")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()


def work_folder():
    pid = os.getpid()
    hostname = socket.gethostname()
    directory = hostname + '_' + str(pid)
    ex_traj = directory + '/ex_traj'
    send_email_report(subject="Start Experiment",
                      txt="Start experiment on " + hostname + " Process id:" + str(pid) + " directory:" + directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(ex_traj):
        os.makedirs(ex_traj)
    return directory


def start_experiment():
    savefolder = work_folder()
    print('saving in : ' + savefolder)
    dir_src = os.getcwd()
    dir_dst = os.getcwd() + '/' + savefolder
    for filename in os.listdir(dir_src):
        if filename.endswith('.py'):
            shutil.copy(dir_src + '/' + filename, dir_dst)
            print(filename)
    return dir_dst, dir_src


def end_experiment(dir_dst, dir_src, backup_extentions=['.png', '.npy', '.h5', '.json', '.out', '.txt', '.csv'],txt=""):
    for ex in backup_extentions:
        for filename in os.listdir(dir_src):
            if filename.endswith(ex):
                shutil.copy(dir_src + '/' + filename, dir_dst)
                print(filename)
    send_email_report(subject="Finish Experiment",
                      txt="Finish experiment on " + dir_src + " saved in  directory:" + dir_dst+" Important result is:"+txt)


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        # Recall metric.
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        # Precision metric.
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))

def plot_results(history,txt):
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy '+txt)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    #plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss '+txt)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    #plt.show()


#       plt.show()

def pre_trained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["@"].shape[0]
    emb_matrix = np.zeros((vocab_len, emb_dim))
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    embedding_layer = Embedding(vocab_len, emb_dim)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer


def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            # print(i,j,w)
            X_indices[i, j] = word_to_index[w]
            j += 1
    return X_indices


def read_glove_vectors(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map
def tr_net_b2(input_shape_speed, word_to_vec_map_speed, word_to_index_speed, input_shape_bearing, word_to_vec_map_bearing, word_to_index_bearing, no_classes=11):

    speed = Input(shape=input_shape_speed, dtype=np.int32)
    embedding_layer = pre_trained_embedding_layer(word_to_vec_map_speed, word_to_index_speed)
    embeddings = embedding_layer(speed)
    X = Bidirectional(LSTM(128, return_sequences=True))(embeddings)
    X = Dropout(0.5)(X)
    X = Bidirectional(LSTM(128))(X)
    X = Dropout(0.5)(X)
    X = Dense(32)(X)
    X = Dropout(0.5)(X)
    X = Dense(16)(X)

    bearing = Input(shape=input_shape_bearing, dtype=np.int32)
    embedding_layer_bearing = pre_trained_embedding_layer(word_to_vec_map_bearing, word_to_index_bearing)
    embeddings_bearing = embedding_layer_bearing(bearing)
    B = Bidirectional(LSTM(128, return_sequences=True))(embeddings_bearing)
    B = Dropout(0.5)(B)
    B = Bidirectional(LSTM(128))(B)
    B = Dropout(0.5)(B)
    B = Dense(32)(B)
    B = Dropout(0.5)(B)
    B = Dense(16)(B) 

    X=Concatenate()([X,B])

    X = Dense(no_classes)(X)
    X = Activation('softmax')(X)
    model = Model(inputs=[speed,bearing],outputs= X)
    return model
def tr_net_b3(input_shape_speed, word_to_vec_map_speed, word_to_index_speed, input_shape_bearing, word_to_vec_map_bearing, word_to_index_bearing, no_classes=11):

    speed = Input(shape=input_shape_speed, dtype=np.int32)
    embedding_layer = pre_trained_embedding_layer(word_to_vec_map_speed, word_to_index_speed)
    embeddings = embedding_layer(speed)
    X = Bidirectional(LSTM(128, return_sequences=True))(embeddings)
    X = Dropout(0.5)(X)
    X = Bidirectional(LSTM(128))(X)
    X = Dropout(0.5)(X)
    X = Dense(32, kernel_regularizer=regularizers.l2(0.001),
                activity_regularizer=regularizers.l1(0.001))(X)
    X = Dropout(0.5)(X)
    X = Dense(16, kernel_regularizer=regularizers.l2(0.001),
                activity_regularizer=regularizers.l1(0.001))(X)

    bearing = Input(shape=input_shape_bearing, dtype=np.int32)
    embedding_layer_bearing = pre_trained_embedding_layer(word_to_vec_map_bearing, word_to_index_bearing)
    embeddings_bearing = embedding_layer_bearing(bearing)
    B = Bidirectional(LSTM(128, return_sequences=True))(embeddings_bearing)
    B = Dropout(0.5)(B)
    B = Bidirectional(LSTM(128))(B)
    B = Dropout(0.5)(B)
    B = Dense(32, kernel_regularizer=regularizers.l2(0.001),
                activity_regularizer=regularizers.l1(0.001))(B)
    B = Dropout(0.5)(B)
    B = Dense(16, kernel_regularizer=regularizers.l2(0.001),
                activity_regularizer=regularizers.l1(0.001))(B) 

    X=Concatenate()([X,B])

    X = Dense(no_classes)(X)
    X = Activation('softmax')(X)
    model = Model(inputs=[speed,bearing],outputs= X)
    return model


def tr_net_b(input_shape_speed, word_to_vec_map_speed, word_to_index_speed, input_shape_bearing, word_to_vec_map_bearing, word_to_index_bearing, no_classes=11):

    speed = Input(shape=input_shape_speed, dtype=np.int32)
    embedding_layer = pre_trained_embedding_layer(word_to_vec_map_speed, word_to_index_speed)
    embeddings = embedding_layer(speed)
    X = Bidirectional(LSTM(64, return_sequences=True))(embeddings)
    X = Dropout(0.5)(X)
    X = Bidirectional(LSTM(64))(X)
    X = Dropout(0.5)(X)
    X = Dense(16)(X)
    X = Dropout(0.5)(X)
    X = Dense(16)(X)
    X = Dropout(0.5)(X)
    X = Bidirectional(LSTM(64))(X)
    X = Dropout(0.5)(X)
    X = Dense(16)(X)
    X = Dropout(0.5)(X)
    X = Dense(16)(X)
    X = Dense(8, kernel_regularizer=regularizers.l2(0.001),     activity_regularizer=regularizers.l1(0.001))(X)


    bearing = Input(shape=input_shape_bearing, dtype=np.int32)
    embedding_layer_bearing = pre_trained_embedding_layer(word_to_vec_map_bearing, word_to_index_bearing)
    embeddings_bearing = embedding_layer_bearing(bearing)
    B = Bidirectional(LSTM(64, return_sequences=True))(embeddings_bearing)
    B = Dropout(0.5)(B)
    B = Bidirectional(LSTM(64))(B)
    B = Bidirectional(LSTM(64, return_sequences=True))(embeddings_bearing)
    B = Dropout(0.5)(B)
    B = Bidirectional(LSTM(64))(B)
    B = Dropout(0.5)(B)

    B = Dense(16)(B)
    B = Dropout(0.25)(B)
    B = Dense(16)(B)

    X=Concatenate()([X,B])

    X = Dense(no_classes, kernel_regularizer=regularizers.l2(0.01),     activity_regularizer=regularizers.l1(0.01))(X)

    B = Bidirectional(LSTM(512, return_sequences=True))(embeddings_bearing)
    B = Dropout(0.5)(B)
    B = Dense(16)(B)
    B = Dropout(0.5)(B)
    B = Dense(16)(B) 

    X=Concatenate()([X,B])

    X = Dense(no_classes, kernel_regularizer=regularizers.l2(0.001),     activity_regularizer=regularizers.l1(0.001))(X)
    X = Activation('softmax')(X)
    model = Model(inputs=[speed,bearing],outputs= X)
    return model

def tr_net_r(input_shape, word_to_vec_map, word_to_index, no_classes=11):
    sentence_indices = Input(shape=input_shape, dtype=np.int32)
    embedding_layer = pre_trained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)

    X = Bidirectional(LSTM(128, return_sequences=True))(embeddings)

    X = Dropout(0.5)(X)
    X = Bidirectional(LSTM(256))(X)
    X = Dropout(0.5)(X)
    X = Dense(no_classes, kernel_regularizer=regularizers.l2(0.01),     activity_regularizer=regularizers.l1(0.01))(X)
    X = Activation('softmax')(X)
    model = Model(sentence_indices, X)
    return model



def tr_net(input_shape, word_to_vec_map, word_to_index, no_classes=11):
    sentence_indices = Input(shape=input_shape, dtype=np.int32)
    embedding_layer = pre_trained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)

    X = Bidirectional(LSTM(512, return_sequences=True))(embeddings)

    X = Dropout(0.5)(X)
    X = Bidirectional(LSTM(512))(X)
    X = Dropout(0.5)(X)
    X = Dense(no_classes)(X)
    X = Activation('softmax')(X)
    model = Model(sentence_indices, X)
    return model


def read_csv(filename='train_speed.csv',
             valid_tran=['airplane', 'boat', 'taxi', 'subway', 'run', 'motorcycle', 'train', 'walk', 'bus', 'bike',
                         'car'], min_words=0):
    phrase = []
    trans_class = []
    dic = {}
    dic2 = {}
    i = 0
    for x in valid_tran:
        print(x + " : " + str(i))
        dic[x] = i
        dic2[i] = x
        i = i + 1

    with open(filename) as csvDataFile:
        csv_reader = csv.reader(csvDataFile, lineterminator='$\n')

        for row in csv_reader:
            l = len(row[0].split())
            print(l)
            if (row[1] in valid_tran) & (l > min_words):
                phrase.append(row[0])
                trans_class.append(dic[row[1]])

    X = np.asarray(phrase)
    Y = np.asarray(trans_class, dtype=int)

    return X, Y, dic, dic2

def read_csv_2(filenames=['train_speed.csv','train_bearing.csv'],
             valid_tran=['airplane', 'boat', 'taxi', 'subway', 'run', 'motorcycle', 'train', 'walk', 'bus', 'bike',
                         'car'], min_words=0):
    phrase = []
    phrase0 = []
    phrase1 = []
    trans_class = []
    dic = {}
    dic2 = {}
    i = 0
    for x in valid_tran:
        print(x + " : " + str(i))
        dic[x] = i
        dic2[i] = x
        i = i + 1

    with open(filenames[0]) as csvDataFile0,open(filenames[1]) as csvDataFile1:
        csv_reader0 = csv.reader(csvDataFile0, lineterminator='$\n')
        csv_reader1 = csv.reader(csvDataFile1, lineterminator='$\n')

        for row0,row1 in zip(csv_reader0,csv_reader1):
            l = len(row0[0].split())
            print(l)
            if (row0[1] in valid_tran) & (l > min_words):
                phrase.append([row0[0],row1[0]])
                phrase0.append(row0[0])
                phrase1.append(row1[0])
                trans_class.append(dic[row0[1]])

    X0 = np.asarray(phrase0)
    X1 = np.asarray(phrase1)
    X = np.asarray(phrase)
    Y = np.asarray(trans_class, dtype=int)

    return X, Y, dic, dic2


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y
