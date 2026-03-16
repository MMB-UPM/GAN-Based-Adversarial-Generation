"""
Module helper for training CriptoGANs
last version as for: 11/10/21
"""
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
import pickle

# For metrics
import numpy as np
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Tensorflow for modules
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
#from keras.utils.layer_utils import count_params

import distancias as dt
import os
import time


#####################################
#        Architecture modules       #
#####################################

# Simples


def get_num_params_from_model_path(model_path):
    model = tf.keras.models.load_model(model_path)

    return get_num_params(input_layers=model.input, output_layers=model.output)


def get_num_params(input_layers=None, output_layers=None):
    """
    Returns the number of parameters in a model
    """
    model = Model(inputs=input_layers, outputs=output_layers)
    trainable_params = int(sum(tf.keras.backend.count_params(p) for p in model.trainable_variables))
    non_trainable_params = int(sum(tf.keras.backend.count_params(p) for p in model.non_trainable_variables))

    #trainable_count = keras.count_params(model.trainable_weights)
    #non_trainable_count = count_params(model.non_trainable_weights)

    return trainable_params + non_trainable_params
    #return trainable_count + non_trainable_count


'''
def _get_num_params(input_layers=None, output_layers=None):
    """
    Returns the number of parameters in a model
    """
    model = Model(inputs=input_layers, outputs=output_layers)
    trainable_count = count_params(model.trainable_weights)
    non_trainable_count = count_params(model.non_trainable_weights)

    return trainable_count + non_trainable_count
'''



def build_discriminator_from_model_complexity(
    feature_dims,
    model_complexity,
    dense_units,
    batch_norm=False,
    p_dropout=0.,
    softmax=True,
    l2reg=0.0,
    alphaLRelu=0.15,
    num_class=3,
    archNN=None,
    es_WGAN=False
):
    """
    Builds a discriminator model with a random architecture that approximates the given number of parameters
    """

    l2 = tf.keras.regularizers.l2(l2=l2reg)
    ki = tf.keras.initializers.glorot_normal()

    input_layer = Input(shape=(feature_dims,))
    
    if archNN == None:
        num_params = 0
        while num_params < model_complexity:
            if num_params == 0:
                neurons_layer = dense_units[0]
                net = Dense(neurons_layer, kernel_regularizer=l2, kernel_initializer=ki)(input_layer)
            else:
                neurons_layer = np.random.choice(dense_units)
                net = Dense(neurons_layer, kernel_regularizer=l2, kernel_initializer=ki)(net)

            if p_dropout > 0:
                net = Dropout(p_dropout)(net)
                
            if batch_norm:
                net = BatchNormalization()(net)
         
            net = LeakyReLU(alpha=alphaLRelu)(net)
            num_params = get_num_params(input_layer, net)
    else:
        net=input_layer
        for i in range(len(archNN)):
            neurons_layer = archNN[i]
            net = Dense(neurons_layer, kernel_regularizer=l2, kernel_initializer=ki)(net)
            
            if p_dropout > 0:
                net = Dropout(p_dropout)(net)
            
            if batch_norm:
                net = BatchNormalization()(net)

            net = LeakyReLU(alpha=alphaLRelu)(net)
            
            
            
    if es_WGAN:
        output_layer = Dense(num_class, kernel_regularizer=l2, kernel_initializer=ki, activation="linear")(net)
    elif softmax:
        output_layer = Dense(num_class, kernel_regularizer=l2, kernel_initializer=ki, activation="softmax")(net)  # classifier one_hot_enc
    else:
        output_layer = Dense(1, activation="sigmoid")(net)  # classifier

    discriminator = tf.keras.Model(input_layer, output_layer, name="substitute_detector")
    discriminator.summary()

    return discriminator


def old_build_discriminator_from_model_accuracy(
    feature_dims,
    model_accuracy,
    dense_units,
    x_train,
    y_train,
    x_test,
    y_test,
    batch_norm=False,
    softmax=True,
    l2reg=0.0,
    alphaLRelu=0.15,
    num_class=3,
    archNN=None
):
    """
    Builds a discriminator model with a random architecture that approximates the given number of parameters
    """

    l2 = tf.keras.regularizers.l2(l2=l2reg)
    ki = tf.keras.initializers.glorot_normal()

    input_layer = Input(shape=(feature_dims,))
    accuracy = 0

    add_input_layer = True

    MAX_ITERATIONS = 10
    iteration = 0

    while accuracy < model_accuracy and iteration < MAX_ITERATIONS:
        if add_input_layer:
            neurons_layer = dense_units[0]
            net = Dense(neurons_layer, kernel_regularizer=l2, kernel_initializer=ki)(input_layer)

            add_input_layer = False
        else:
            neurons_layer = np.random.choice(dense_units)
            net = Dense(neurons_layer, kernel_regularizer=l2, kernel_initializer=ki)(output_layer)

        if batch_norm:
            net = BatchNormalization()(net)

        net = LeakyReLU(alpha=alphaLRelu)(net)

        if softmax:
            output_layer = Dense(num_class, kernel_regularizer=l2, kernel_initializer=ki, activation="softmax")(
                net
            )  # classifier ohe
        else:
            output_layer = Dense(1, activation="sigmoid")(net)  # classifier

        discriminator = tf.keras.Model(input_layer, output_layer, name="substitute_detector")
        discriminator.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        # convert to ohe
        y_train_ohe = np.zeros((y_train.shape[0], y_train.max() + 1))
        y_train_ohe[np.arange(y_train.shape[0]), y_train] = 1
        y_test_ohe = np.zeros((y_test.shape[0], y_test.max() + 1))
        y_test_ohe[np.arange(y_test.shape[0]), y_test] = 1
        # BalAccScore = BalancedAccuracyScore(x_test, y_test)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        discriminator.fit(
            x=x_train,
            y=y_train_ohe,
            batch_size=4096,
            epochs=10000,
            verbose=0,
            callbacks=[early_stopping],
            validation_data=(x_test, y_test_ohe),
        )

        y_pred_ohe = discriminator.predict(x_test)

        y_pred = np.argmax(y_pred_ohe, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        bal_accuracy = balanced_accuracy_score(y_test, y_pred)

        print("\nAccuracy of DISC: ", accuracy)
        print("Balanced Accuracy of DISC: ", bal_accuracy)
        disc_num_params = get_num_params(input_layer, output_layer)
        print("Number of parameters of DISC: ", disc_num_params)

        iteration += 1

    discriminator.summary()

    return discriminator


def build_generator_from_model_complexity(
    feature_dims,
    latent_dims,
    model_complexity,
    dense_units,
    p_dropout,
    activation="linear",
    separate_start=False,
    l2reg=0.0,
    custom_activation=None,
    batch_norm=True,
    alphaLRelu=0.15,
    archNN=None,
    solo_noise=False
):
    """
    Improved MalGAN

    Noise and example will both pass a layer of non linearities independently before
    concatenation.

    Then they will follow a backbone of n_layers blocks and finally return the generated input.

    Attributes:
    - feature_dims: int: number of dimensions of feature input
    - latent_dims: int: number of dimensions of latent noise input
    - n_layers: int: depth of the model
    - dense_units: array: units of dense layers, len value hast to be >= len(n_layers)+1
    - p_dropout: array: probabilities of dropout layers, len value has to be >= len(n_layers)+1
    - activation: string: type of activation: tanh, sigmoid, etc.
    - maximum: bool: If return maximum of example input or model by maximum values.

    """

    l2 = tf.keras.regularizers.l2(l2=l2reg)
    ki = tf.keras.initializers.glorot_normal()

    example = Input(shape=(feature_dims,))
    noise = Input(shape=(latent_dims,))
    
    if solo_noise:
        entrada=noise
    else:
        entrada = concatenate([noise, example])

    entrada = Flatten()(entrada)
    x = entrada

    if archNN == None:
        
        num_params = 0

        while num_params < model_complexity:
            neurons = np.random.choice(dense_units)
            x = Dense(neurons, kernel_regularizer=l2, kernel_initializer=ki)(x)

            if p_dropout > 0:
                x = Dropout(p_dropout)(x)

            if batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU(alpha=alphaLRelu)(x)

            num_params = get_num_params(input_layers=[example, noise], output_layers=[x])
            print("Number of parameters of GEN: ", num_params)
    else:
        
        for i in range(len(archNN)):
            neurons_layer = archNN[i]
            x = Dense(neurons_layer, kernel_regularizer=l2, kernel_initializer=ki)(x)

            if p_dropout > 0:
                net = Dropout(p_dropout)(x)

            if batch_norm:
                bn=BatchNormalization()
                bns.append(bn)
                net = bn(x)

            net = LeakyReLU(alpha=alphaLRelu)(x)

    last_layer = x
    ll = []
    bns=[]
    if custom_activation != None:
        
        for i in range(feature_dims):
            print(
                "Adding Smirnov Transformation function as the activation of the last layer of the generator:",
                i,
                custom_activation[i],
            )
            l = Dense(1)(last_layer)
            
            bn=BatchNormalization(scale=False, center=False,momentum=0.2)
            bns.append(bn)          
            #l = BatchNormalization(scale=False, center=False)(l)
            l = bn(l)
            l = tf.keras.layers.Lambda(custom_activation[i])(l)
            ll.append(l)
        t_smirnov = tf.keras.layers.concatenate(ll)
        t_smirnov = tf.keras.layers.Reshape((feature_dims,))(
            t_smirnov
        )  # "t_smirnov" is the tensor of the last layer of the generator.
        x = t_smirnov

    else:
        x = Dense(feature_dims, activation=activation)(last_layer)

    if solo_noise:
        generator = tf.keras.Model(noise, x, name="generator")
    else:
        generator = tf.keras.Model([noise,example], x, name="generator")
        
    generator.summary()
    return generator, bns


def build_discriminator_simple(
    feature_dim, n_layers, dense_units, batch_norm=False, softmax=True, l2reg=0.0, alphaLRelu=0.15, num_class=3
):
    """
    Simplest model possible.
    - feature_dims: int (len of example vector dimension)
    - dense_units: array (dimension of dense layer)
    """
    assert len(dense_units) == n_layers, "Filters need to have same dimension than n_layers."

    l2 = tf.keras.regularizers.l2(l2=l2reg)
    ki = tf.keras.initializers.glorot_normal()

    input = Input(shape=(feature_dim,))
    x = input
    # x = Flatten()(x)

    for dim in range(n_layers):
        x = Dense(dense_units[dim], kernel_regularizer=l2, kernel_initializer=ki)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alphaLRelu)(x)

    # x = Dropout(0.2)(x)
    # x = Dense(1)(x)
    if softmax:
        x = Dense(num_class, kernel_regularizer=l2, kernel_initializer=ki, activation="softmax")(x)  # classifier ohe
    else:
        x = Dense(1, activation="sigmoid")(x)  # classifier

    discriminator = tf.keras.Model(input, x, name="substitute_detector")
    discriminator.summary()
    return discriminator


# Complex
def build_generator_2(
    feature_dims,
    latent_dims,
    n_layers,
    dense_units,
    p_dropout,
    activation="linear",
    separate_start=False,
    l2reg=0.0,
    custom_activation=None,
    batch_norm=True,
    alphaLRelu=0.15,
    solo_noise=False
):
    """
    Improved MalGAN

    Noise and example will both pass a layer of non linearities independently before
    concatenation.

    Then they will follow a backbone of n_layers blocks and finally return the generated input.

    Attributes:
    - feature_dims: int: number of dimensions of feature input
    - latent_dims: int: number of dimensions of latent noise input
    - n_layers: int: depth of the model
    - dense_units: array: units of dense layers, len value hast to be >= len(n_layers)+1
    - p_dropout: array: probabilities of dropout layers, len value has to be >= len(n_layers)+1
    - activation: string: type of activation: tanh, sigmoid, etc.
    - maximum: bool: If return maximum of example input or model by maximum values.

    """
    assert (
        len(dense_units) + 1 >= n_layers
    ), f"Filters need to have a dimension >= len(n_layers) + 1. \n - n_layers: {n_layers} \n - dense_units lenght: {len(dense_units)}"
    assert (
        len(p_dropout) + 1 >= n_layers
    ), f"Filters need to have a dimension >= len(n_layers) + 1. \n - n_layers: {n_layers} \n - p_dropout lenght: {len(p_dropout)}"

    l2 = tf.keras.regularizers.l2(l2=l2reg)
    ki = tf.keras.initializers.glorot_normal()

    example = Input(shape=(feature_dims,))
    # example_= Dense (latent_dims) (example)
    noise = Input(shape=(latent_dims,))
    """
    if latent_dims != feature_dims:
        noise= Dense (feature_dims) (noise)
        #noise= LeakyReLU(alpha=alphaLRelu)(noise)
        #noise = BatchNormalization(scale=False, center=False)(noise)
        #noise = PReLU()(noise)
    noise_d= Dense (feature_dims) (noise)
    #noise_bn = BatchNormalization()(noise)
    entrada= add([noise_d, example])
    """

    if solo_noise:
        entrada=noise
    else:
        entrada = concatenate([noise, example])
        
    entrada = Flatten()(entrada)
    x = entrada
    # example = Flatten()(example)

    for dim in range(0, n_layers):
        x = Dense(dense_units[dim], kernel_regularizer=l2, kernel_initializer=ki)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alphaLRelu)(x)
        # x = PReLU()(x)

        # x = Dropout(p_dropout[dim])(x)

    # x = Dropout(p_dropout[-1])(x)
    last_layer = x
    ll = []
    if custom_activation != None:
        for i in range(feature_dims):
            print("Smirnov Transformation Activation:", i, custom_activation[i])
            l = Dense(1)(last_layer)
            l = BatchNormalization(scale=False, center=False)(l)
            l = tf.keras.layers.Lambda(custom_activation[i])(l)
            ll.append(l)
        t_smirnov = tf.keras.layers.concatenate(ll)
        t_smirnov = tf.keras.layers.Reshape((feature_dims,))(t_smirnov)
        # "t_smirnov" is the tensor of the last layer of the generator.
        x = t_smirnov

    else:
        x = Dense(feature_dims)(x)
    # x = Activation(activation=activation)(x)

    if solo_noise:
        generator = tf.keras.Model(noise, x, name="generator")
    else:
        generator = tf.keras.Model([noise,example], x, name="generator")
        
    generator.summary()
    return generator


#####################################
#          Model functions          #
#####################################


def add_model_regularizer_loss(model):
    loss = 0
    for l in model.layers:
        if hasattr(l, "layers") and l.layers:  # the layer itself is a model
            loss += add_model_regularizer_loss(l)
        if hasattr(l, "kernel_regularizer") and l.kernel_regularizer:
            loss += l.kernel_regularizer(l.kernel)
        if hasattr(l, "bias_regularizer") and l.bias_regularizer:
            loss += l.bias_regularizer(l.bias)
    return loss


def build_model(classify_algo, params={}):
    """
    Try first DT & LR, MLP, RF
    """
    if classify_algo == "RF":
        # model = RandomForestClassifier(n_estimators=1000)
        print("hola RF")
        n_estimators = params["n_estimators"] if "n_estimators" in params.keys() else 100
        max_depth = params["max_depth"] if "max_depth" in params.keys() else None
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, max_depth=max_depth, verbose=1)

    elif classify_algo == "GB":
        model = GradientBoostingClassifier()
    elif classify_algo == "AB":
        model = AdaBoostClassifier()
    elif classify_algo == "LR":
        model = LogisticRegression(solver="lbfgs")
    elif classify_algo == "DT":
        model = DecisionTreeClassifier()
    elif classify_algo == "MLP":
        model = MLPClassifier(hidden_layer_sizes=(64,))
    elif classify_algo == "SVM":
        model = SVC()
    elif classify_algo == "BNB":
        model = BernoulliNB()
    elif classify_algo == "KNN":
        model = KNeighborsClassifier(n_jobs=10)
    elif classify_algo == "VOTE":
        clf1 = RandomForestClassifier(n_estimators=1000)
        clf2 = GradientBoostingClassifier()
        clf3 = AdaBoostClassifier()
        clf4 = LogisticRegression(solver="lbfgs")
        clf5 = DecisionTreeClassifier()
        clf6 = MLPClassifier(hidden_layer_sizes=(64,))
        clf7 = SVC()
        model = VotingClassifier(
            estimators=[
                ("RF", clf1),
                ("GB", clf2),
                ("AB", clf3),
                ("LR", clf4),
                ("DT", clf5),
                ("MLP", clf6),
                ("SVM", clf7),
            ],
            voting="soft",
        )
    else:
        raise NotImplementedError("Error! No such classify algorithm")
    return model


def train(ds_dir, algo, output_dir):
    # Features filter
    f_UPC = [3, 24, 51, 53]

    # Dataset Paths
    ds_train = ds_dir / "train"
    ds_test = ds_dir / "test"

    # Load train data
    x_train = np.load(ds_train / "x_train.npy")[:, f_UPC]
    y_train = np.load(ds_train / "y_train.npy")

    # Load test data
    x_test = np.load(ds_test / "x_test.npy")[:, f_UPC]
    y_test = np.load(ds_test / "y_test.npy")

    # Train
    model = build_model(algo)
    model.fit(x_train, y_train)

    # Evaluate
    print(model.score(x_test, y_test))
    y_pred = model.predict(x_test)
    print(confusion_matrix(y_test, y_pred, [1, 0]))
    print("f1 score:", f1_score(y_test, y_pred))

    pickle.dump(model, open(output_dir / str("blackbox_model_" + str(algo) + ".txt" + ".pkl"), "wb"))


def bb_confusion(algo, model_dir, samples, real_labels):
    """
    Returns a confusion matrix of bb algorithm for samples and real labels
    """

    # Predictions
    y_pred = predict(samples, algo, model_dir)

    # Confusion matrix
    return confusion_matrix(real_labels, y_pred, [1, 0])


# model=None

# def predict(batch, algo, bb_dir):

#     # Predict
#     #model = pickle.load(open(bb_dir/str('blackbox_model_'+str(algo)+'.txt'+'.pkl'), 'rb'))
#     #if model == None:
#     model = pickle.load(open(bb_dir/str('blackbox_model_'+str(algo)+'.pkl'), 'rb'))
#     y_pred = model.predict(batch)

#     # Print
#     """
#     for i, result in enumerate(y_pred):
#         if result == 0:
#             print(i, ' is clean traffic')
#         elif result == 1:
#             print(i, ' is cripto attack')
#         else:
#             print(result)
#     """
#     return y_pred


import inspect


def get_gpu_memory_usage():
    return
    """Use nvidia-smi to get the gpu memory usage"""
    # nvidia-smi | grep MiB | grep -v MiB | awk '{print $3}'
    memory_usage = os.popen("nvidia-smi | grep MiB | awk '{print $8}'").read()
    memory_usage_list = memory_usage.split("\n")

    # get the one corresponding to the python process
    processes = os.popen("nvidia-smi | grep MiB | awk '{print $7}'").read()
    processes_list = processes.split("\n")

    python_process_index = processes_list.index("python")
    memory_usage = memory_usage_list[python_process_index]

    # get the peak memory usage
    memory_usage_list = []
    memory_usage_list.append(memory_usage)

    sorted_memory_usage_list = sorted(memory_usage_list)
    peak_memory_usage = sorted_memory_usage_list[-1]

    print(peak_memory_usage)


last_bb_model=None
last_bb_model_path=""

def predict(batch, algo, bb_model_path, batch_size=1024, in_tape_gradient=False):
    global last_bb_model
    global last_bb_model_path
    
    #print("CALLING BB PREDICT WITH BATCH SIZE:", batch_size)
    #curframe = inspect.currentframe()
    #calframe = inspect.getouterframes(curframe, 2)

    #print("caller name:", calframe[1][3])
    if algo == "RF":
        if bb_model_path != last_bb_model_path:
            bb_model = pickle.load(open(bb_model_path, "rb"))
            last_bb_model=bb_model
            last_bb_model_path=bb_mode_path
            tf.print ("load model:",bb_model_path,"last model:",last_bb_model_path)
        else:
            bb_model=last_bb_model
        y_pred = bb_model.predict(batch)
    
    elif algo == "NN":
        #print("GPU memory usage (before)")
        #get_gpu_memory_usage()
        if bb_model_path != last_bb_model_path:
            bb_model = tf.keras.models.load_model(bb_model_path)
            last_bb_model=bb_model
            last_bb_model_path=bb_model_path
            tf.print ("load model:",bb_model_path,"last model:",last_bb_model_path)
        else:
            bb_model=last_bb_model
        
        if in_tape_gradient:
            y_pred = bb_model(batch, training=False)
        else:
            y_pred = bb_model.predict(batch, batch_size=batch_size)
        y_pred = np.argmax(y_pred, axis=1)

        #print("GPU memory usage (after)")
        #get_gpu_memory_usage()

    return y_pred

last_bb_model_tr=None
last_bb_model_path_tr=""

def predict_training(batch, bb_model_path, batch_size=1024):
    global last_bb_model_tr
    global last_bb_model_path_tr
    #print("CALLING BB PREDICT WITH BATCH SIZE:", batch_size)
    #curframe = inspect.currentframe()
    #calframe = inspect.getouterframes(curframe, 2)

    #print("caller name:", calframe[1][3])

    #print("GPU memory usage (before)")
    #get_gpu_memory_usage()
    if bb_model_path != last_bb_model_path_tr :
        bb_model = tf.keras.models.load_model(bb_model_path)
        last_bb_model_path_tr= bb_model_path
        last_bb_model_tr= bb_model
        tf.print ("load model:",bb_model_path,"last model:",last_bb_model_path)
    else:
        bb_model= last_bb_model_tr
        
    y_pred = bb_model(batch, training=False)

    #print("GPU memory usage (after)")
    #get_gpu_memory_usage()

    return y_pred


#####################################
#        Sampling functions         #
#####################################

# OBSOLETO, se usa muestras.py


def _sample_examples(ds_dir, batch_size, cl=0, test=False, GAN_training=False):
    """
    Sample examples of class cl=(0,1) : (normal_traffic, cripto_traffic)
    from dataset in ds_dir, with batch_size given and subset test (test=True)
    or train (test=False)
    """
    # Dataset Paths
    # ds_train = ds_dir/"train"
    # ds_test = ds_dir/"test"
    ds_train = ds_dir
    ds_test = ds_dir
    if GAN_training:
        ddd
    else:
        path_x = ds_train / "x_train.npy" if not test else ds_test / "x_test.npy"
        path_y = ds_train / "y_train.npy" if not test else ds_test / "y_test.npy"

    x = np.load(path_x)
    y = np.load(path_y)
    # print ("media total:",np.mean(x,axis=0))
    # print ("std total:",np.std(x,axis=0))

    # Actual sampling
    indeces = np.where(y == cl)[0]
    # print ("Num total elemin class:",cl,len(indeces))
    idx = np.random.choice(indeces, batch_size, replace=False)

    return x[idx]


def _sample_benign_examples(ds_dir, batch_size, test=False):
    """
    Benign examples are those labelled with 0.
    """
    # Dataset Paths
    # ds_train = ds_dir/"train"
    # ds_test = ds_dir/"test"
    ds_train = ds_dir
    ds_test = ds_dir

    if not test:
        # Load train data
        path_x = ds_train / "x_train.npy"
        path_y = ds_train / "y_train.npy"
    else:
        # Load test data
        x_test = np.load(ds_test / "x_test.npy")
        y_test = np.load(ds_test / "y_test.npy")

    x = np.load(path_x)
    y = np.load(path_y)

    # Actual sampling
    benign_indeces = np.where(y == 0)[0]
    idx = np.random.choice(benign_indeces, batch_size)

    return x[idx]


def _sample_malign_examples(ds_dir, batch_size, test=False):
    """
    Malign exampes are those labelled with 1.
    """
    # Dataset Paths
    # ds_train = ds_dir/"train"
    # ds_test = ds_dir/"test"
    ds_train = ds_dir
    ds_test = ds_dir

    if not test:
        # Load train data
        path_x = ds_train / "x_train.npy"
        path_y = ds_train / "y_train.npy"
    else:
        # Load test data
        x_test = np.load(ds_test / "x_test.npy")
        y_test = np.load(ds_test / "y_test.npy")

    # Load test data
    # x_test = np.load(ds_test/"x_test.npy")
    # y_test = np.load(ds_test/"y_test.npy")

    # Actual sampling
    malign_indeces = np.where(y_train == 1)[0]
    idx = np.random.choice(malign_indeces, batch_size)

    return x_train[idx]


#####################################
#         Metrics functions         #
#####################################


def compute_histogram(sample, xmin, xmax, npoints=10000):
    x = np.linspace(xmin, xmax, npoints)
    try:
        kernel = stats.gaussian_kde(sample)
        return kernel(x), True
    except Exception as e:
        print("Exception:", e)
        return None, False


def my_distance_jensenshannon(sample1, sample2, npoints=10000):
    xmin = min(min(sample1), min(sample2))
    xmax = max(max(sample1), max(sample2))

    histo1, ok1 = compute_histogram(sample1, xmin, xmax, npoints)
    histo2, ok2 = compute_histogram(sample2, xmin, xmax, npoints)

    return distance.jensenshannon(histo1, histo2, 2.0) if ok1 and ok2 else np.inf


def my_distance_wasserstein(sample1, sample2, npoints=10000):
    xmin = min(min(sample1), min(sample2))
    xmax = max(max(sample1), max(sample2))

    # histo1,ok1 = compute_histogram(sample1, xmin, xmax, npoints)
    # histo2,ok2 = compute_histogram(sample2, xmin, xmax, npoints)
    # return wasserstein_distance(histo1, histo2) if ok1 and ok2 else np.inf
    return wasserstein_distance(sample1, sample2)  # if ok1 and ok2 else np.inf


def get_centroid(s1):
    """
    Calculates the sum of each component of a vector and return a new
    one with the same dimension as the elements of the original list
    """
    assert s1 != []
    assert len(s1) != 0
    return np.add.reduce(s1) / len(s1)


def euclidean_distance(sample1, sample2,debug=False, dist_alb=True, tipo_distancia="normal", 
                       stochastic=False, inf=0.0, sup=1.0):
    assert len(sample1) == len(sample2)
    #inf=0
    #sup=0.85
    return dt.get_dist_WK(sample1, sample2,debug=debug,dist_alb=dist_alb, 
                          tipo_distancia=tipo_distancia, stochastic=stochastic, inf=inf, sup=sup, 
                          )


def old_euclidean_distance(sample1, sample2):
    assert len(sample1) == len(sample2)
    c1 = get_centroid(sample1)
    c2 = get_centroid(sample2)
    return np.linalg.norm(c1 - c2)


def measure_samples(samples, dist_type="WS"):
    """
    Returns [Jensen Shannon, Wasserstain] distance vector of given samples
    """
    assert type(samples) is type([]), "samples need to be a list"
    assert len(samples) == 2, "samples must contain 2 items"
    (s1, s2) = samples
    assert len(s1) == len(s2), "samples must be same length"
    if dist_type == "WS":
        distance = my_distance_wasserstein(s1, s2, len(s1))
    else:
        distance = my_distance_jensenshannon(s1, s2, len(s1))
    return distance


def compute_metrics(rb, rm, distance="WS", scaler=None, msg=None, debug=False,dist_alb=True,
                    tipo_distancia="normal",stochastic=False, inf=0.0, sup=1.0, 
                    **kwargs):
    """
    Measure metrics for samples of more than one dimension.
    Returns n distances, one for each fature.
    distance :  WS JS or EU for Wasserstein, Jensen Shannon, or Euclidean
    """
    #tf.print ("compute_metrics",distance)
    #if msg != None:
    #    tf.print (msg)
    assert len(rb) == len(rm)
    assert len(rb[0]) == len(rm[0])
    dist_list = []  # 1xn
    if scaler:
        rb = scaler.inverse_transform(rb)
        rm = scaler.inverse_transform(rm)
        pause()
        # Se supone que medimos las distancias de las distribuciones normalizadas/escaladas
        
    if distance == "WS":
        n_kwargs={}
        for k,v in kwargs.items():
            if "ws_" in k:
                n_kwargs[k]=v
        #print ("compute metrics, ws:",n_kwargs)
        dist= dt.wasser (rb,rm,**n_kwargs) #*len(rb)
        # old wasserstein
        '''
        for f in range(len(rb[0])):
            a = rb[:, f]
            b = rm[:, f]
            w = measure_samples([a, b], dist_type="WS")
            dist_list.append(w)
        dist = dist_list
        '''
    elif distance == "WS-SH":
        n_kwargs={}
        for k,v in kwargs.items():
            if "sh_" in k:
                n_kwargs[k]=v
        #print ("compute metrics, ws-sh:",n_kwargs)
        t1=time.time()
        dist= dt.wasser_aprox (rb,rm,**n_kwargs) *len(rb)
        t2=time.time()
        t_sh=t2-t1
        '''
        t1=time.time()
        dist2= dt.wasser_aprox_tf (rb,rm,**n_kwargs) *len(rb)
        t2=time.time()
        t_sh_tf=t2-t1
        print (f"compute_metrics: diff sh-sh_tf: {dist-dist2} \n tiempos: t_sh:{t_sh}, tsh_tf:{t_sh_tf}")
        #print (f"sh-ws: dist con cpu:{dist} dist con tensores:{dist2}")
        if np.abs(dist-dist2)> 0.01:
            print ("mucha diferencia")
            print (f"sh-ws: dist con cpu:{dist} dist con tensores:{dist2}")
            pause()
        
        '''
        
    elif distance == "JS":
        for f in range(len(rb[0])):
            a = rb[:, f]
            b = rm[:, f]
            j = measure_samples([a, b], dist_type="JS")
            dist_list.append(j)
        dist = np.sum(dist_list)
        
    elif distance == "EU":
        dist = euclidean_distance(rb, rm, debug=debug, dist_alb=dist_alb, tipo_distancia=tipo_distancia, 
                                  stochastic=stochastic, inf=inf, sup=sup)/sup # para normalizar el valor al tamaño de la muestra
    else:
        tf.print ("Distancia descnonocida: ", distance)
        pause()
        
    return dist


"""
distances[i, m_d_type[m], :] = (
                        mc.compute_metrics(rb1, rb2, m),0  ## 11   
                        mc.compute_metrics(rm1, rm2, m),1  ## 22
                        mc.compute_metrics(gm1, gm2, m),2  ## 33
                        mc.compute_metrics(rb1, rm1, m),3  ## 12
                        mc.compute_metrics(gm1, rb1, m),4  ## 13
                        mc.compute_metrics(gm1, rm1, m),5  ## 23
                        #
                        mc.compute_metrics(rm2, rb2, m),6  ## 21
                        mc.compute_metrics(rb2, gm2, m),7  ## 31
                        mc.compute_metrics(rm2, gm2, m),8  ## 32
                    )

"""


def _confeccionate_matrix(distances, metrics_type, std=False):
    values = np.std(distances[:, metrics_type, :], axis=0) if std else np.mean(distances[:, metrics_type, :], axis=0)
    f1 = np.round([values[0], values[3], values[4]], decimals=3)
    f2 = np.round([values[6], values[1], values[5]], decimals=3)
    f3 = np.round([values[7], values[8], values[2]], decimals=3)
    distance_matrix = np.matrix(f"{f1};{f2};{f3}")
    return distance_matrix


def confeccionate_matrix(distances, metrics_type, std=False):
    # values = np.std(distances[:,metrics_type,:], axis=0) if std else np.mean(distances[:,metrics_type,:], axis=0)
    values = distances[0, metrics_type, :]
    assert distances.shape[0] == 1
    f1 = np.round([values[0], values[3], values[4]], decimals=3)
    f2 = np.round([values[6], values[1], values[5]], decimals=3)
    f3 = np.round([values[7], values[8], values[2]], decimals=3)
    distance_matrix = np.matrix(f"{f1};{f2};{f3}")
    return distance_matrix


# ---------
# OBSO
# ---------


def old_get_closest_dist(sample, group):
    """
    Will return the least distance from a sample to group.
    Helper function for custom loss
    """
    dists = [tf.math.reduce_euclidean_norm(tf.math.subtract(e, sample)) for e in group]
    return np.min(dists)


def _get_closest_dist(sample, group):
    """
    Will return the least distance from a sample to group.
    Helper function for custom loss
    """
    minimo = np.inf
    # dists = [tf.math.reduce_euclidean_norm(tf.math.subtract(e, sample)) for e in group]
    for e in group:
        d = tf.math.reduce_euclidean_norm(tf.math.subtract(e, sample))
        if d < minimo:
            minimo = d
    # tf.print ("minimo:",d,type(d))
    return minimo


def old_build_generator_2(
    feature_dims, latent_dims, n_layers, dense_units, p_dropout, activation="linear", separate_start=False
):
    """
    Improved MalGAN

    Noise and example will both pass a layer of non linearities independently before
    concatenation.

    Then they will follow a backbone of n_layers blocks and finally return the generated input.

    Attributes:
    - feature_dims: int: number of dimensions of feature input
    - latent_dims: int: number of dimensions of latent noise input
    - n_layers: int: depth of the model
    - dense_units: array: units of dense layers, len value hast to be >= len(n_layers)+1
    - p_dropout: array: probabilities of dropout layers, len value has to be >= len(n_layers)+1
    - activation: string: type of activation: tanh, sigmoid, etc.
    - maximum: bool: If return maximum of example input or model by maximum values.

    """
    assert (
        len(dense_units) + 1 >= n_layers
    ), f"Filters need to have a dimension >= len(n_layers) + 1. \n - n_layers: {n_layers} \n - dense_units lenght: {len(dense_units)}"
    assert (
        len(p_dropout) + 1 >= n_layers
    ), f"Filters need to have a dimension >= len(n_layers) + 1. \n - n_layers: {n_layers} \n - p_dropout lenght: {len(p_dropout)}"

    example = Input(shape=(feature_dims,))
    noise = Input(shape=(latent_dims,))
    # example = Flatten()(example)
    # separate_start = False
    if separate_start:
        # Processing noise this way could give us XAI by making
        # the network to learn those activations that are relevant
        # with respect to the variations a vector needs
        # and then extracting these weights
        # noise = Dense(dense_units[0])(noise)
        # noise = PReLU()(noise)
        # noise = Dense(dense_units[0])(noise)
        # noise = BatchNormalization()(noise)
        # noise = PReLU()(noise)
        # noise = Dropout(p_dropout[0])(noise)
        example = Dense(dense_units[0])(example)
        example = PReLU()(example)
        example = Dense(dense_units[0])(example)
        example = BatchNormalization()(example)
        example = PReLU()(example)
        example = Dropout(p_dropout[0])(example)

    x = Concatenate(axis=1)([example, noise])

    for dim in range(1 if separate_start else 0, n_layers):
        x = Dense(dense_units[dim])(x)
        x = PReLU()(x)
        x = Dense(dense_units[dim])(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        # x = Dropout(p_dropout[dim])(x)

    # x = Dropout(p_dropout[-1])(x)
    x = Dense(feature_dims)(x)
    x = Activation(activation=activation)(x)

    generator = tf.keras.Model([example, noise], x, name="generator")
    generator.summary()
    return generator


def _old_build_generator_simple(feature_dims, latent_dims, n_layers, dense_units, batch_norm=False):
    """
    Simplest model possible.
    - feature_dims: int (len of example vector dimension)
    - latent_dims: int (len of noise dimension)
    - dense_units: array (dimension of dense layer)
    """
    assert len(dense_units) >= n_layers, "Filters need to have same or greater dimension than n_layers."
    example = Input(shape=(feature_dims,))
    noise = Input(shape=(latent_dims,))
    x = Concatenate(axis=1)([example, noise])
    x = example  # 4xbs
    for dim in range(n_layers):
        x = Dense(dense_units[dim])(x)
        if batch_norm:
            x = BatchNormalization()(x)
    x = Dense(feature_dims)(x)
    x = Activation(activation="linear")(x)  # bc data is scaled
    generator = tf.keras.Model([example, noise], x, name="generator")
    generator.summary()
    return generator
