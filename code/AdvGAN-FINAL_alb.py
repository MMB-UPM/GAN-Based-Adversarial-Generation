#!/usr/bin/env python

import tensorflow as tf
import argparse



# Parse arguments

argparser = argparse.ArgumentParser(description="Script to train CryptoGAN")
argparser.add_argument("--exp", type=str, default="0", help="Experiment number")
argparser.add_argument("--dataset", type=str, default="ctu", help="Dataset to use")
argparser.add_argument("--modeltype", type=str, default="rf", help="Model type to use")
argparser.add_argument("--modelsize", type=str, default="small", help="Model size to use")
argparser.add_argument("--combid", type=str, default="000", help="Combination ID")
argparser.add_argument("--epochs", type=int, default=500, help="Epochs to run")
#argparser.add_argument("--gen_semilla", type=int, default=0, help="Epochs to run")
argparser.add_argument("--archNN", type=str, default="y", help="Arquitectura")
argparser.add_argument("--model_gan", type=str, default="advgan", help="advgan/malgan")
argparser.add_argument("--gpu_id", type=str, default="0", help="[0..4]")
argparser.add_argument("--es_wgan", type=str, default="gan", help="gan/wgan")
#argparser.add_argument("--gp_wgan", type=float, default="10", help="5/10/20.5")


args = argparser.parse_args()

EXP_NUM = args.exp
dataset = args.dataset
BB_MODEL_TYPE = args.modeltype.upper()
BB_SIZE = args.modelsize
trial_id = args.combid
EPOCHS= args.epochs
archNN= args.archNN
model_gan=args.model_gan
gpu_id=int(args.gpu_id)
es_wgan= args.es_wgan == "wgan"
#gp_wgan= args.gp_wgan 

WGAN=es_wgan
#GP_WGAN=gp_wgan

CUDA_VISIBLE_DEVICES = 1
gpus = tf.config.experimental.list_physical_devices("GPU")
print ("GPUS visibles:",gpus)

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print ("Logical GPUS visibles despues de fijar gpu_id:",logical_gpus)
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        

tf.print("Experiment number:", EXP_NUM)
tf.print("Dataset:", dataset)
tf.print("Model type:", BB_MODEL_TYPE)
tf.print("Model size:", BB_SIZE)
tf.print("Trial ID:", trial_id)
tf.print ("EPOCHS:", EPOCHS)
tf.print ("archNN:",archNN)
tf.print ("model_gan:",model_gan)
tf.print ("gpu_id:",gpu_id)
tf.print ("es_wgan:",es_wgan,args.es_wgan)

import distancias as dist
import model_constructor_alb as mc
import muestras as mu
#import advgan_alb as ag
import advgan_alb_solo_noise as ag
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

import matplotlib.pyplot as plt
import time
import smirnov_activation as sa
import json
from numba import cuda
import os
import pickle
from tensorflow.keras.utils import plot_model
from pathlib import Path
import yaml

selected_features_datasets = {
    "syn": [0, 1, 2, 3],
    "ctu": [0, 3, 4, 5, 6, 7, 8],
    "crypto": [0, 1, 2, 3],
    "adult": [0, 1, 2, 3, 4, 5, 7, 8, 9],
}
selected_features = selected_features_datasets[dataset]
FEAT_DIM = len(selected_features)

scaler_datasets = {
    "syn": "MaxMin",
    "ctu": "MaxMin", 
    "crypto": "MaxMin",
    "adult": "Standard",
}

tipo_scaler= scaler_datasets[dataset]




# DEFINE MODEL SIZE
BB_NOISE_TYPE = "uniform"

# BLACK BOX MODELS DEFINITION
BB_MODEL_COD = f"{BB_MODEL_TYPE}_f-" + str(selected_features) + f"size-{BB_SIZE}_noise-{BB_NOISE_TYPE}_{dataset}"

BB_MODEL_EXT = "h5" if BB_MODEL_TYPE == "NN" else "pkl"
DISTILLED_BB_MODEL_EXT = "h5"

EXPS = {
    "0": {"ACT_SMIRNOV_G": False, "DIST_G": False, "BETA": 0.5},
    "1": {"ACT_SMIRNOV_G": True, "DIST_G": False, "BETA": 0.5},
    "2": {"ACT_SMIRNOV_G": False, "DIST_G": True, "BETA": 0.0},
    "3": {"ACT_SMIRNOV_G": True, "DIST_G": True, "BETA": 0.5},
}

EXP = EXPS[EXP_NUM]
EXP_NAME = "EXP_{exp_num}_{model_gan}_{dataset}_{bb_model_type}_{size}".format(
    exp_num=EXP_NUM, model_gan=model_gan, dataset=dataset, bb_model_type=BB_MODEL_TYPE, size=BB_SIZE,
)


OUT_DIR = Path(f"./{model_gan}_{dataset}_output/")
DS_DIR = Path(f"./dataset/{dataset}/")
DATASET_INFO_DIR = Path(f"./dataset_info/")

BB_TRAIN_DATA = Path(f"./bb_data/{BB_MODEL_TYPE}/{BB_MODEL_TYPE}_BB_{dataset}_{BB_SIZE}_train_dataset.npy")
BB_TRAIN_LABELS = Path(f"./bb_data/{BB_MODEL_TYPE}/{BB_MODEL_TYPE}_BB_{dataset}_{BB_SIZE}_train_labels.npy")
BB_TEST_DATA = Path(f"./bb_data/{BB_MODEL_TYPE}/{BB_MODEL_TYPE}_BB_{dataset}_{BB_SIZE}_test_dataset.npy")
BB_TEST_LABELS = Path(f"./bb_data/{BB_MODEL_TYPE}/{BB_MODEL_TYPE}_BB_{dataset}_{BB_SIZE}_test_labels.npy")

BB_MODEL_PATH = Path(f"./bb_models/{BB_MODEL_TYPE}/{BB_MODEL_TYPE}_BB_{BB_SIZE}_{dataset}.{BB_MODEL_EXT}")

DISTILLED_BB_MODEL_PATH = Path(
    f"./bb_models/{BB_MODEL_TYPE}/{BB_MODEL_TYPE}_BB_{BB_SIZE}_{dataset}_distilled.{DISTILLED_BB_MODEL_EXT}"
)
BB_MODEL_TEST_RESULTS = Path(
    f"./bb_test_results/{BB_MODEL_TYPE}/{BB_MODEL_TYPE}_BB_{dataset}_{BB_SIZE}_test_results.json"
)

NN_BB_MODEL_PATH = Path(f"./bb_models/NN/NN_BB_{BB_SIZE}_{dataset}.h5")


#SCALER_PATH = DS_DIR / f"scaler_attacker_{dataset}.pkl"
    
SCALER_PATH = DS_DIR / f"./scaler_{dataset}_{tipo_scaler}.pkl"



# Create directories
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/GAN_training_times/").mkdir(parents=True, exist_ok=True)
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/GAN_models/").mkdir(parents=True, exist_ok=True)
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/evaluation/bb_hits").mkdir(parents=True, exist_ok=True)
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/evaluation/losses").mkdir(parents=True, exist_ok=True)
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/evaluation/distances").mkdir(parents=True, exist_ok=True)
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/evaluation/distance_matrix").mkdir(parents=True, exist_ok=True)
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/evaluation/other_metrics").mkdir(parents=True, exist_ok=True)
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/plots/bb_hits").mkdir(parents=True, exist_ok=True)
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/plots/bb_distances").mkdir(parents=True, exist_ok=True)
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/plots/gen_loss").mkdir(parents=True, exist_ok=True)
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/plots/disc_loss").mkdir(parents=True, exist_ok=True)
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/plots/eu_gen_mal_dist_training").mkdir(parents=True, exist_ok=True)
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/plots/wa_gen_mal_dist_training").mkdir(parents=True, exist_ok=True)
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/plots/eu_ben_mal_dist_training").mkdir(parents=True, exist_ok=True)
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/plots/wa_ben_mal_dist_training").mkdir(parents=True, exist_ok=True)
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/plots/eu_dist").mkdir(parents=True, exist_ok=True)
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/plots/wa_dist").mkdir(parents=True, exist_ok=True)
Path(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/model_plots").mkdir(parents=True, exist_ok=True)



# GENERATOR LATENT DIM
LATENT_DIM = FEAT_DIM

# DISCRIMINATOR EXTRA TRAINING STEPS
DISC_EXTRA_STEPS = 1

# SMIRNOV ACTIVATION FUNCTION
ACT_SMIRNOV_G = EXP["ACT_SMIRNOV_G"]

# BATCH SIZE
BS = 512

# DISCRIMINATOR
DISC_DENSE_UNITS = {"small": [128, 256, 512], "large": [512, 1024,]}
#conf_small
archNN_DISC= [50,500,50] if archNN == "y" else None
#conf_large
#archNN_DISC= [50,500,500,50] if archNN == "y" else None
DISC_BATCH_NORM = False #True
alphaLRelu_D = 0.15
NUM_CLASS_DISC = 2
L2_REG_DISC = 0.001
disc_lr = 0.001
DISC_DROPOUT = 0.

# GENERATOR
#gen_layers = [50, 500, 1000, 500, 50]
GEN_DENSE_UNITS = {"small": [128, 256, 512], "large": [512, 1024,]}
# crypto
#archNN_GEN= [50, 500, 1000, 500, 50] if archNN == "y" else None
#CTU
archNN_GEN= [50, 500, 4000, 500, 50] if archNN == "y" else None
#Para exp0
#archNN_GEN= [50, 100, 400, 100, 50] if archNN == "y" else None
GEN_DROPOUT = 0.
GEN_BATCH_NORM = False
alphaLRelu_G = 0.15
L2_REG_GEN = 0.001
gen_lr = 0.001

#EPOCHS = 500
DISC_WARMUP_EPOCHS = 0 # 1 # 5

NOISE_NUM = 1000000000000000000000000000

########################################
# ALBERTO
########################################
SAMPLE_RATIO_SMIRNOV_DATASET=1
RATIO_COMPLEXITY_GEN_DISC=2


#----------

# AdvGAN
# beta = EXP["BETA"]
alpha= 0 # 1.0
beta = 0 # 1.0 #0.5

print("AdvGAN Beta:", beta)


# Para ser Determinista
#SEMILLA_FIJA=True

GEN_SEMILLA=True

if not GEN_SEMILLA :
    SEMILLA=1
    tf.random.set_seed(SEMILLA)
    tf.keras.utils.set_random_seed(some_seed)
    #tf.config.experimental.enable_op_determinism()
    np.random.seed(SEMILLA)
    random.seed(SEMILLA)


# Clustering
#with open("clustering_only_malign.json", "r") as f:
#    clustering_data_only_malign = json.loads(f.read())

#with open("clustering.json", "r") as f:
#    clustering_data = json.loads(f.read())

#clustering_k_only_malign = clustering_data_only_malign[dataset]
#clustering_k = clustering_data[dataset]

# Muestras
tf.print ("creo muestras")

    
mu.muestras = mu.Muestras(dir=f"{DS_DIR}/", lista_features=selected_features,RATIO_STD_CORTE=None,tipo_scaler=tipo_scaler,scaler_path=SCALER_PATH,dataset_features=selected_features)

# ------------ Combinations -------------

with open(f"Combinations/Combination_{trial_id}.yaml", "r") as outfile:
    combination = yaml.load(outfile, Loader=yaml.FullLoader)

combination_dict = dict(combination)

    
if archNN == "y" and "DISC_NN" in combination_dict.keys():
    archNN_DISC= combination_dict["DISC_NN"]
else:
    pause()
    
if archNN == "y" and "GEN_NN" in combination_dict.keys():
    archNN_GEN= combination_dict["GEN_NN"]
else:
    pause()

tf.print ("archNN_DISC",archNN_DISC)
tf.print ("archNN_GEN",archNN_GEN)
#pause ()
#model.DISC_NN= combination_dict["DISC_NN"]
#tf.print ("XX:",combination_dict["DISC_NN"],":")
#tf.print ("YY:",archNN_DISC,":",archNN_DISC[0])
#tf.print (combination_dict)
#pause ()

if es_wgan:
    print ("GP_WGAN",combination_dict["GP_WGAN"])
else: 
    print ("No hay WGAN")
    combination_dict["GP_WGAN"]=0.0
    
#pause()

# -------------------------


if ACT_SMIRNOV_G:
    tf.print ("Activando Smirnov ..")
    # Sample malign data to build the Smirnov Activation that we will later add to the generator
    #_X_train = mu.muestras.sample_examples(batch_size=0, tipo_dataset=mu.MUESTRA_MG, cl=mu.MALIGN)
    _X_train = mu.muestras.sample_examples(batch_size=0, class_label=mu.MALIGN)

    tf.print("Shape _X_train: ", _X_train.shape)

    #sample_size_ratio = 0.2  # 1 = 100%
    sample_size_ratio = SAMPLE_RATIO_SMIRNOV_DATASET  
    
    if sample_size_ratio < 1:
        sample_size = int(sample_size_ratio * _X_train.shape[0])
        indices = np.random.choice(_X_train.shape[0], sample_size, replace=True)
        XX_train = _X_train[indices]
    else:
        XX_train = _X_train

    print("Shape XX_train: ", XX_train.shape)
    
    if "SMIRNOV_POINTS" in combination_dict.keys():
        puntos_spline=combination_dict["SMIRNOV_POINTS"]
    else:
        puntos_spline=2000
        
    tf.print ("SMIRNOV_POINTS:",puntos_spline)
                         

    sa = sa.SmirnovActivation(XX_train.shape,puntos_spline=puntos_spline)
    sa.create(XX_train)

d_softmax = model_gan == "malgan"


if BB_MODEL_TYPE == "NN":
    bb_model_num_params = mc.get_num_params_from_model_path(NN_BB_MODEL_PATH)
    print(f"Number of parameters in BB model: {bb_model_num_params}")

    discriminator = mc.build_discriminator_from_model_complexity(
        FEAT_DIM,
        model_complexity=bb_model_num_params,
        dense_units=DISC_DENSE_UNITS[BB_SIZE],
        batch_norm=DISC_BATCH_NORM,
        p_dropout=DISC_DROPOUT,
        softmax=d_softmax,
        l2reg=L2_REG_DISC,
        alphaLRelu=alphaLRelu_D,
        num_class=NUM_CLASS_DISC,
        archNN=archNN_DISC,
        es_WGAN=WGAN
    )
elif BB_MODEL_TYPE == "RF":
    bb_model_num_params = mc.get_num_params_from_model_path(NN_BB_MODEL_PATH)
    print(f"Number of parameters in BB model: {bb_model_num_params}")

    discriminator = mc.build_discriminator_from_model_complexity(
        FEAT_DIM,
        model_complexity=bb_model_num_params,
        dense_units=DISC_DENSE_UNITS[BB_SIZE],
        batch_norm=DISC_BATCH_NORM,
        p_dropout=DISC_DROPOUT,
        softmax=d_softmax,
        l2reg=L2_REG_DISC,
        alphaLRelu=alphaLRelu_D,
        num_class=NUM_CLASS_DISC,
        archNN=archNN_DISC,
    )

disc_model_num_params = mc.get_num_params(discriminator.inputs, discriminator.outputs)
print(f"Number of parameters in discriminator: {disc_model_num_params}")

custom_activation_function = sa.custom_fs if ACT_SMIRNOV_G else None

generator,bns  = mc.build_generator_from_model_complexity(
    FEAT_DIM,
    LATENT_DIM,
    model_complexity=RATIO_COMPLEXITY_GEN_DISC*disc_model_num_params,
    dense_units=GEN_DENSE_UNITS[BB_SIZE],
    p_dropout=GEN_DROPOUT,
    l2reg=L2_REG_GEN,
    custom_activation=custom_activation_function,
    batch_norm=GEN_BATCH_NORM,
    alphaLRelu=alphaLRelu_G,
    archNN=archNN_GEN,
    solo_noise=True,
)

# Chequeo de GAN/WGAN para no meter la pata
if not WGAN:
    if "GP_WGAN" in combination_dict:
        if combination_dict["GP_WGAN"] != 0:
            tf.print (f'error. GAN con GP_WAN:{combination_dict["GP_WGAN"]}')
            pause ()
        else:
            tf.print (f'GAN estandar con GP_WAN 0: {combination_dict["GP_WGAN"]}')
    else:
        tf.print ("GAN estandar. No hay GP_WAN")
else: #WGAN
    if combination_dict["GP_WGAN"] == 0:
        tf.print ("WARNiNG. es WGAN pero GP_WAN es 0")
    else:
        tf.print (f'WGAN con GP_WAN >0: {combination_dict["GP_WGAN"]}')
    

    
# Create model
model = ag.AdvGAN(
    discriminator,
    generator,
    dataset=dataset,
    exp_name=EXP_NAME,
    trial_id=trial_id,
    latent_dim=LATENT_DIM,
    feature_dims=FEAT_DIM,
    discriminator_extra_steps=DISC_EXTRA_STEPS,
    black_box_model=BB_MODEL_TYPE,
    ds_dir=DS_DIR,
    output_dir=OUT_DIR,
    bb_model_path=BB_MODEL_PATH,
    distilled_bb_model_path=DISTILLED_BB_MODEL_PATH,
    debug2=False,
    alfa=alpha,
    beta=beta,
    scaler_path=SCALER_PATH,
    noise_num=NOISE_NUM,
    #clustering_k_only_malign=clustering_k_only_malign,
    #clustering_k=clustering_k,
    selected_features=selected_features,
    bns=bns,
    model_gan=model_gan,
    es_WGAN=WGAN,
    gp_weight=combination_dict["GP_WGAN"]
)

tf.print ("fin create ag.AdvGAN")

optimizer_d_adam_params = {"learning_rate": disc_lr, "beta_1": 0.5, "beta_2": 0.9}
optimizer_g_adam_params = {"learning_rate": gen_lr, "beta_1": 0.5, "beta_2": 0.9}

optimizer_d_adam = tf.keras.optimizers.Adam(**optimizer_d_adam_params)
optimizer_g_adam = tf.keras.optimizers.Adam(**optimizer_g_adam_params)

optimizer_d = optimizer_d_adam
optimizer_g = optimizer_g_adam


# Define la función de pérdida del generador
def wgan_generator_loss(fake_logits):
    cfd=tf.reduce_mean(fake_logits)
    tf.print ("g_loss, mean critic_fake_data:",cfd)
    return -cfd

# Define la función de pérdida del crítico
def wgan_critic_loss(real_logits=None, fake_logits=None):
    crd=tf.reduce_mean(real_logits)
    cfd=tf.reduce_mean(fake_logits)
    tf.print ("d_loss, mean critic_real_data:",crd)
    tf.print ("d_loss, mean critic_fake_data:",cfd)
    return  cfd - crd


'''
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss

# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)

'''

if WGAN:
    loss_d= wgan_critic_loss
    loss_g= wgan_generator_loss
else:
    loss_d = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss_g = tf.keras.losses.BinaryCrossentropy(from_logits=False)

model.compile(optimizer_d, optimizer_g, loss_d, loss_g)

model.it = 0
model.save()

# Set model dataset
ds_train = model.ds_dir
ds_test = model.ds_dir

# # Define test dataset path
# path_x = ds_train / "x_test.npy"
# path_y = ds_train / "y_test.npy"

# # Load the data
# x = np.load(path_x)
# x = x[:, selected_features]
# y = np.load(path_y)

# # Create random noise
# x_random_vectors_test = np.random.normal(0, 25, size=(61000, FEAT_DIM))
# y_random_vectors_test = np.linspace(2, 2, num=61000)
# print("Random noise shape: ", x_random_vectors_test.shape)

# # Add random noise to the test set
# _xx = np.concatenate([x, x_random_vectors_test], axis=0)
# _yy = np.concatenate([y, y_random_vectors_test], axis=0)

# # Try to predict the test data with the (non-trained) model just to prove that it is correctly processed
# y_pred = mc.predict(_xx, BB_MODEL_TYPE, BB_MODEL_PATH)

# # Print confusion matrix
# m = confusion_matrix(_yy, y_pred, labels=[0, 1, 2])
# print(m)

# # Configure testing dataset of the model
# model._xx = _xx
# model._yy = _yy


model.RATIO_LOSS_G = combination_dict["RATIO_LOSS_G"]
model.RATIO_REG_G = combination_dict["RATIO_REG_G"]

model.beta=combination_dict["BETA_SAMPLE_DISTANCE"]
model.alfa=combination_dict["ALPHA_DISTILLED_LOSS"]

model.RATIO_DIST_G = combination_dict["RATIO_DIST_G"]
if not EXP["DIST_G"] :
    assert combination_dict["RATIO_DIST_G"] == 0, f"debe ser experimento 2XX/3XX. RATIO_DIST_G:{model.RATIO_DIST_G}"
    

model.umbral_dist_alb= combination_dict["UMBRAL_DIST_ALB"] # >0 : filtra por distancia alb, 0: distancia euclidea
model.dist_alb= model.umbral_dist_alb >0 #False #True #se puede poner distancia euclidea
# Si pongo distancia alb (RATIO_DIST_G>0), entonces tengo que tener algo de distancia_alb, sino seria toda euclidea y seria vainilla "0xx"
tf.print ("HOLA:",(model.RATIO_DIST_G>0) , model.dist_alb ,"\n")
assert (model.RATIO_DIST_G>0) == model.dist_alb 

if "STOCHASTIC" in combination_dict.keys():
    model.stochastic= combination_dict["STOCHASTIC"]
else:
    model.stochastic= False
tf.print ("STOCHASTIC:",model.stochastic)

if "TIPO_DISTANCIA" in combination_dict.keys():
    model.tipo_distancia= combination_dict["TIPO_DISTANCIA"]
else:
    model.tipo_distancia= "normal" # euclidea sin elevar al cuadrado
tf.print ("TIPO_DISTANCIA:",model.tipo_distancia)

model.RATIO_LOSS_D = combination_dict["RATIO_LOSS_D"]
model.RATIO_REG_D = combination_dict["RATIO_REG_D"]


print("RATIO_DIST_G:", model.RATIO_DIST_G)
print("RATIO_LOSS_G:", model.RATIO_LOSS_G)
print ("BETA_SAMPLE_DISTANCE",model.beta)
print ("ALPHA_DISTILLED_LOSS",model.alfa)
print("RATIO_REG_G:", model.RATIO_REG_G)
tf.print (f"UMBRAL_DIST_ALB: {model.umbral_dist_alb} dist_alb: {model.dist_alb}") 

print("RATIO_LOSS_D:", model.RATIO_LOSS_D)
print("RATIO_REG_D:", model.RATIO_REG_D)

if DISC_WARMUP_EPOCHS >0 :
    time_disc_warmup_start = time.time()
    model.train(epochs=DISC_WARMUP_EPOCHS, batch_size=BS, train_gen=False)
    time_disc_warmup_end = time.time()

    time_disc_warmup = time_disc_warmup_end - time_disc_warmup_start
else:
    time_disc_warmup=0


#### -------- TRAIN 
time_gan_training_start = time.time()
model.train(epochs=EPOCHS, batch_size=BS, train_gen=True)
time_gan_training_end = time.time()
#### -------- END TRAIN

time_gan_training = time_gan_training_end - time_gan_training_start

# Save training times
training_times = {
    "EXP_NAME": EXP_NAME,
    "trial_id": trial_id,
    "time_disc_warmup": time_disc_warmup,
    "time_gan_training": time_gan_training,
    "RATIO_LOSS_D": model.RATIO_LOSS_D,
    "RATIO_REG_D": model.RATIO_REG_D,
    "RATIO_LOSS_G": model.RATIO_LOSS_G,
    "RATIO_REG_G": model.RATIO_REG_G,
    "RATIO_DIST_G": model.RATIO_DIST_G,
}

with open(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/GAN_training_times/{EXP_NAME}_tn-{trial_id}.json", "w") as f:
    json.dump(training_times, f)

# Save model
discriminator.save(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/GAN_models/disc_{EXP_NAME}_tn-{trial_id}.h5")
generator.save(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/GAN_models/gen_{EXP_NAME}_tn-{trial_id}.h5")

# PLOT HITS
x = model.lista_bh
y = model.lista_mh

np.save(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/evaluation/bb_hits/lista_bh.npy", model.lista_bh)
np.save(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/evaluation/bb_hits/lista_mh.npy", model.lista_mh)

plt.plot(x, label="bh")
plt.plot(y, label="mh")
plt.legend(loc="upper left")

plt.savefig(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/plots/bb_hits/{EXP_NAME}_test_hits_tn-{trial_id}.png")

# PLOT LOSS
gen_loss = []
disc_loss = []

for dic in model.history:
    gen_loss.append(dic["g_loss"])
    disc_loss.append(dic["d_loss"])

# GENERATOR
plt.plot(gen_loss)
plt.savefig(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/plots/gen_loss/{EXP_NAME}_test_gen_loss_tn-{trial_id}.png")
print("Training gen loss (last iteration): ", gen_loss[-1])

# DISCRIMINATOR
plt.plot(disc_loss)
plt.savefig(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/plots/disc_loss/{EXP_NAME}_test_disc_loss_tn-{trial_id}.png")
print("Training disc loss (last iteration): ", disc_loss[-1])

np.save(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/evaluation/losses/gen_loss.npy", gen_loss)
np.save(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/evaluation/losses/disc_loss.npy", disc_loss)

pause()

#################
#################
#################

# NO SE EJECUTA ------------------------------------


# SAVE DISTANCES MATRICES
with open(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/evaluation/distance_matrix/distance_matrix.pkl", "wb") as fp:
    pickle.dump(model.metrics_dict["distance_matrix"], fp)

with open(f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/evaluation/other_metrics/other_metrics.pkl", "wb") as fp:
    pickle.dump(model.metrics_dict, fp)

# PLOT DISTANCES MATRICES
matrices_wa = []
matrices_eu = []

for k, v in model.metrics_dict["distance_matrix"].items():
    if v != {}:
        for i in range(1, len(v)):
            if k == "WS":  # Wasserstein Distance
                matrices_wa.append(v[f"{i:>5}".replace(" ", "0")])
            elif k == "EU":  # Euclidean Distance
                matrices_eu.append(v[f"{i:>5}".replace(" ", "0")])

# Euclidean Distance
gen_mal_dist = []

for m in matrices_eu:
    gen_mal_dist.append(m[1, 2])

plt.plot(gen_mal_dist)
plt.legend(["eu"])
plt.title("gen-mal distance during training")
plt.xlabel("epoch%20")
plt.ylabel("value")
plt.savefig(
    f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/plots/eu_gen_mal_dist_training/{EXP_NAME}_training_eu_gen_mal_dist_tn-{trial_id}.png"
)

# Wasserstein Distance
gen_mal_dist = []

for m in matrices_wa:
    gen_mal_dist.append(m[1, 2])

plt.plot(gen_mal_dist)
plt.legend(["wass"])
plt.title("gen-mal distance during training")
plt.xlabel("epoch%20")
plt.ylabel("value")
plt.savefig(
    f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/plots/wa_gen_mal_dist_training/{EXP_NAME}_training_wa_gen_mal_dist_tn-{trial_id}.png"
)

# Euclidean Distance

real_dist_eu = []

for m in matrices_eu:
    real_dist_eu.append(m[1, 0])

plt.plot(real_dist_eu)
plt.legend(["eu"])
plt.title("ben-mal distance during training")
plt.xlabel("epoch%20")
plt.ylabel("value")
plt.savefig(
    f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/plots/eu_ben_mal_dist_training/{EXP_NAME}_training_eu_ben_mal_dist_tn-{trial_id}.png"
)

# Wasserstein Distance

real_dist_wa = []

for m in matrices_wa:
    real_dist_wa.append(m[1, 0])

plt.legend(["wass"])
plt.xlabel("epoch%20")
plt.ylabel("value")
plt.title("ben-mal distance during training")
plt.plot(real_dist_wa, label="wass")
plt.savefig(
    f"./{model_gan}_{dataset}_output/{EXP_NAME}/{trial_id}/plots/wa_ben_mal_dist_training/{EXP_NAME}_training_wa_ben_mal_dist_tn-{trial_id}.png"
)

# Euclidean Distance

bb = []
mm = []
gg = []

for m in matrices_eu:
    bb.append(m[0, 0])
    mm.append(m[1, 1])
    gg.append(m[2, 2])

plt.plot(bb)
plt.plot(mm)
plt.plot(gg)
plt.legend("bmg")
plt.title("euclidean")
plt.xlabel("epoch%20")
plt.ylabel("value")

bm = []
bg = []
mg = []

for m in matrices_eu:
    bm.append(m[0, 1])
    bg.append(m[0, 2])
    mg.append(m[1, 2])

plt.plot(bm)
plt.plot(bg)
plt.plot(mg)
plt.legend(["bm", "bg", "mg"])
plt.title("eu distances")
plt.xlabel("epoch%20")
plt.ylabel("value")
plt.savefig(
    f"./{dataset}_output/{EXP_NAME}/{trial_id}/plots/eu_dist/{EXP_NAME}_eu_dist_tn-{trial_id}_with_intra_distances.png"
)

plt.plot(bm)
plt.plot(bg)
plt.plot(mg)
plt.legend(["bm", "bg", "mg"])
plt.title("eu distances")
plt.xlabel("epoch%20")
plt.ylabel("value")
plt.savefig(f"./{dataset}_output/{EXP_NAME}/{trial_id}/plots/eu_dist/{EXP_NAME}_eu_dist_tn-{trial_id}.png")

# Wasserstein Distance

bb = []
mm = []
gg = []

for m in matrices_wa:
    bb.append(m[0, 0])
    mm.append(m[1, 1])
    gg.append(m[2, 2])

plt.plot(bb)
plt.plot(mm)
plt.plot(gg)
plt.legend("bmg")
plt.title("wass")
plt.xlabel("epoch%20")
plt.ylabel("value")

# Wasserstein Distance
bm = []
bg = []
mg = []

for m in matrices_wa:
    bm.append(m[0, 1])
    bg.append(m[0, 2])
    mg.append(m[1, 2])

plt.plot(bm)
plt.plot(bg)
plt.plot(mg)
plt.legend(["bm", "bg", "mg"])
plt.title("wass distances")
plt.xlabel("epoch%20")
plt.ylabel("value")
plt.savefig(
    f"./{dataset}_output/{EXP_NAME}/{trial_id}/plots/wa_dist/{EXP_NAME}_wa_dist_tn-{trial_id}_with_intra_distances.png"
)

plt.plot(bm)
plt.plot(bg)
plt.plot(mg)
plt.legend(["bm", "bg", "mg"])
plt.title("wass distances")
plt.xlabel("epoch%20")
plt.ylabel("value")
plt.savefig(f"./{dataset}_output/{EXP_NAME}/{trial_id}/plots/wa_dist/{EXP_NAME}_wa_dist_tn-{trial_id}.png")

# save file marking combination as done

file_id = f"EXP_{EXP_NUM}_{dataset}_{BB_MODEL_TYPE}_{BB_SIZE}_{trial_id}"
folder_dir = f"./{dataset}_output/EXP_{EXP_NUM}_{dataset}_{BB_MODEL_TYPE}_{BB_SIZE}/{trial_id}"
file_name = f"{folder_dir}/{file_id}.txt"

with open(file_name, "w") as f:
    f.write("done")
