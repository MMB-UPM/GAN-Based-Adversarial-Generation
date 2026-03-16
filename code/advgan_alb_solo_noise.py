import distancias as dist
import model_constructor_alb as mc
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import muestras as mu
import collections
import math
import threading
import seaborn as sns
from clustering import KMeansHelper
import pandas as pd


# DEFAULTS
OUTPUT_DIR = Path("./output/")

# BLACK_BOX_DIR = Path("./black_box_models/test/")
BLACK_BOX_DIR = Path("./")
FEATURE_DIMS = 11
DISCRIMINATOR_EXTRA_STEPS = 1

METRICS_SAMPLING_LENGTH = 512 # Para dist_alb en modo full no stochastic 1000

# BLACK_BOX_MODEL = "RF_3"

###BLACK_BOX_MODEL = "RF_ALB_3_v2_CTU"
###DS_DIR = Path("./dataset/new_ctu/")

BENIGN_LABEL = 0
MALIGN_LABEL = 1
RUIDO_LABEL = 2

FAKE_LABEL = 0
REAL_LABEL = 1


#Para hacer cubitos
BENIGN = 0
MALIGN = 1

FACTOR_RND=0.002
selec_ETIQUETA=MALIGN
#FACTOR_ESCALA= 16 if selec_ETIQUETA==0 else 6
#FACTOR_ESCALA= 16 if selec_ETIQUETA==0 else 10

# ------------------------
dataset_uso= "escala_0-1" #"crypto"
# ------------------------

if dataset_uso == "escala_0-1":
    #FACTOR_ESCALA= 16 if selec_ETIQUETA==0 else 10
    #crypto etiqueta MALIGN
    #FACTOR_ESCALA=2 #10
    FACTOR_ESCALA=5 #10 #10
    FR=0.02
else:
    print ("Se supone que usamos en todos Escala 0-1. Si se usa otra escala cambiar aqui")
    pause()

'''
elif dataset_uso == "ctu":
    FACTOR_ESCALA=1
else:
    print ("NO se ha definido ningun dataset. No hay escala posible")
    pause()
'''

print (f"FACTOR_ESCALA por defecto {FACTOR_ESCALA}, Factor random FR={FR}, para dataset_uso: {dataset_uso}")
dist.set_ESCALA (FE=FACTOR_ESCALA,FR=FR)


# random_examples BB
# uniform dist
#noise_std = 25
#noise_mean = 0

NUM_CLASSES = 2


def sum_gradient(lista_grads):
    tot = 0
    for e in lista_grads:
        ee = e.numpy()
        # print (ee.shape)
        tot += np.sum(ee)
    return tot


# CLASS DEFINITION
class AdvGAN(tf.keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        dataset,
        exp_name=None,
        trial_id=None,
        feature_dims=FEATURE_DIMS,
        discriminator_extra_steps=DISCRIMINATOR_EXTRA_STEPS,
        metrics_sampling=METRICS_SAMPLING_LENGTH,
        black_box_model=None,
        ds_dir=None,
        output_dir=OUTPUT_DIR,
        bb_model_path=BLACK_BOX_DIR,
        distilled_bb_model_path=None,
        alfa=0.5,
        beta=0.5,
        debug=False,
        debug2=False,
        scaler_path=None,
        noise_num=10,
        clustering_k_only_malign=None,
        clustering_k=None,
        selected_features=[],
        dist_alb=True,
        stochastic=True,
        tipo_distancia= "NORMAL", # "CUADRADOS"
        bns=[],
        model_gan="advgan",
        es_WGAN=False,
        gp_weight=10.0
    ):
        super(AdvGAN, self).__init__()
        
        # Experiment name
        self.exp_name = exp_name if exp_name else f"EXP_{np.random.randint(99999):5.0f}"

        # Seed
        # self.seed = np.random.randint(999999)

        # Output Paths: metrics, models, ds_dir
        if trial_id is not None:
            self.trial_id = trial_id
            self.output_dir = output_dir / self.exp_name / str(self.trial_id)
        else:
            self.trial_id = None
            self.output_dir = output_dir / self.exp_name

        self.metrics_dir = self.output_dir / "metrics"
        self.models_dir = self.output_dir / "./GAN_models"
        self.bb_model_path = bb_model_path
        self.distilled_bb_model_path = distilled_bb_model_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.ds_dir = ds_dir
        self.distances_dir = self.output_dir / "distances"
        self.distances_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir = self.output_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.preds_dir = self.output_dir / "preds"
        self.preds_dir.mkdir(parents=True, exist_ok=True)
        self.bb_hits_plots_dir = self.output_dir / "plots" / "bb_hits"
        self.bb_hits_plots_dir.mkdir(parents=True, exist_ok=True)
        self.distances_plots_dir = self.output_dir / "plots" / "distances"
        self.distances_plots_dir.mkdir(parents=True, exist_ok=True)

        # Models
        self.model_gan=model_gan
       
        self.discriminator = discriminator
        self.generator = generator
        self.black_box_model = black_box_model
        self.bns=bns #batch normalization layer para ver las medias y varianzas

        self.gp_weight= gp_weight
        self.WGAN= es_WGAN
        # Shaping the generator
        #example = Input(shape=(feature_dims,))
        #noise = Input(shape=(latent_dim,))
        #input_tensor = [noise, example]
        #generator_output_tensor = generator(input_tensor)
        #self.generator = tf.keras.Model(input_tensor, generator_output_tensor)
        

        # Models config
        self.d_steps = discriminator_extra_steps
        self.latent_dim = latent_dim
        self.feature_dims = feature_dims

        # Metrics
        # NOTE: Maybe to store everything in a dictionary
        self.metrics_dict = {}
        self.metrics_dict["distance_matrix"] = {}
        self.metrics_dict["distance_matrix"]["WS"] = {}
        self.metrics_dict["distance_matrix"]["JS"] = {}
        self.metrics_dict["distance_matrix"]["EU"] = {}
        self.metrics_dict["distance_matrix"]["WS-SH"] = {}
        #
        self.metrics_dict["distance_matrix_filter"] = {}
        self.metrics_dict["distance_matrix_filter"]["WS"] = {}
        self.metrics_dict["distance_matrix_filter"]["JS"] = {}
        self.metrics_dict["distance_matrix_filter"]["EU"] = {}
        self.metrics_dict["distance_matrix_filter"]["WS-SH"] = {}
        
        #Distancia de benigno real a maligno filtrado por BB
        self.metrics_dict["rb_rm-f"]={}
        self.metrics_dict["rb_rm-f"]["EU"]={}
        
        #
        self.metrics_dict["CM"] = {}
        self.metrics_dict["CM"]["bb_class_malign"] = {}
        self.metrics_dict["CM"]["d_class_malign"] = {}
        self.metrics_dict["clustering_mse"] = {}
        self.metrics_dict["BB_hits"] = {}
        self.metrics_dict["BB_hits_real"] = {}

        self.history = []
        self.k = metrics_sampling
        self.bb_class_malign = []
        self.d_class_malign = []

        # Training config
        self.tipo_distancia= tipo_distancia # "cuadrados" # o raiz cuadrada "normal"
        
        self.DIST_LOSS = True
        self.REG_LOSS = True

        self.RATIO_REG = 0.0
        self.RATIO_DIST = 0.0
        self.RATIO_G_LOSS = 1.0

        self.debug = debug
        self.debug2 = debug2
        
        self.debugALB=True
        self.debug=False
        self.debug2=debug2

        # self.get_gpu_memory_usage()

        self.alfa = alfa
        self.beta = beta

        '''
        self.scaler_path = scaler_path

        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        '''
        

        self.noise_num = noise_num

        self.clustering_k_only_malign = clustering_k_only_malign
        self.clustering_k = clustering_k

        self.dataset = dataset
        self.feats= selected_features
        
        # Funcion distancia_alb
        self.dist_alb=dist_alb
        self.stochastic=stochastic
        self.stochastic_train=stochastic
        self.tipo_distancia= "normal" # "cuadrados"
        self.tipo_distancia_train= "normal" # "cuadrados"
        self.inf_train=0.0
        self.sup_train=1 #0.90  # 1.0
        self.inf=0.0
        self.sup=1.0
        
        
        #Gen
        #self.max_rnd= 0.00000000000000000000000001
        #self.min_rnd=-self.max_rnd #poniamos [-1..1]
        
        self.max_rnd= 1.0
        self.min_rnd=-self.max_rnd #poniamos [-1..1]
        
      
        #self.min_rnd=-1.0 #poniamos [-1..1]
        #self.max_rnd=1.0
        tf.print (f"Umbral random noise. min_rnd:{self.min_rnd} max_rnd:{self.max_rnd}")
        
        #ADV_GAN modificada para meter solo ruido enla entrada
        self.solo_noise=True 
        tf.print ("solo_noise:", self.solo_noise)

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(AdvGAN, self).compile()

        # Optimizers
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        # Loss
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

        # Compute reference metrics
        # self.compute_reference_metrics()

    def set_sampling_size(self, k):
        """
        Change sampling size of the sampling functions
        """
        self.k = k

    def set_ratio_reg(self, ratio):
        self.RATIO_REG = ratio

    def set_ratio_dist(self, ratio):
        self.RATIO_DIST = ratio

    def set_ratio_g_loss(self, ratio):
        self.RATIO_G_LOSS = ratio

    def sample_generator(self, num_samples, training=True):
        """
        Call the generator to generate a batch of adversarial samples
        """
        
        assert num_samples <= 2048
        
        # Todas las muestras se generan con los datos del dataset MG
        # random_latent_vectors = tf.random.normal(shape=(num_samples, self.latent_dim))
        random_latent_vectors = tf.random.uniform(shape=(num_samples, self.latent_dim), 
                                                  minval=self.min_rnd, maxval=self.max_rnd)
        malign_inputs = mu.muestras.sample_examples(num_samples, class_label=MALIGN_LABEL)

        #if num_samples > 2048:
        #    print("WARNING: Sampling more than 2048 samples")
        
        
        #training=True
        if self.solo_noise:
            adversarials= self.generator(random_latent_vectors, training=training)
        else:
            adversarials= self.generator((random_latent_vectors, malign_inputs), training=training)
    
        return adversarials

    def sample_generator_x(self, num_samples, training=True):
        """
        Call the generator to generate a batch of adversarial samples
        """
        assert num_samples <= 2048
        
        # random_latent_vectors = tf.random.normal(shape=(num_samples,self.latent_dim))
        random_latent_vectors = tf.random.uniform(shape=(num_samples, self.latent_dim), 
                                                  minval=self.min_rnd, maxval=self.max_rnd)
        # Todas las muestras se generan con los datos del dataset MG
        malign_inputs = mu.muestras.sample_examples(num_samples, class_label=MALIGN_LABEL)
        
        
        #training=True
        if self.solo_noise:
            adversarials= self.generator(random_latent_vectors, training=training)
        else:
            adversarials= self.generator((random_latent_vectors, malign_inputs), training=training)
            
        
        return adversarials, random_latent_vectors, malign_inputs,
    
    def get_malign_filtered (self,num_muestras=None,label_filter_BB=BENIGN_LABEL,ratio_ev_malign=5,vueltas=0):
        
        tf.print ("get_malign_filtered ratio:",ratio_ev_malign)
        if num_muestras != None:
            num_samples= num_muestras
        else:
            num_samples=self.k
        #print ("ratio_ev_malign",ratio_ev_malign)
        
        num_muestras=num_samples
        
        tf.print ("pido muestras:",num_samples*ratio_ev_malign)
        rm1_eu = mu.muestras.sample_examples(num_samples*ratio_ev_malign, class_label=MALIGN_LABEL)
        
        #Get preds Black Box
        '''
        var = self.scaler.var_[self.feats]
        std = np.sqrt(var)
        mean = self.scaler.mean_[self.feats]
        non_scaled_maligns= rm1_eu*std + mean
        '''
        non_scaled_maligns =  mu.muestras.inverse_normalise (rm1_eu, tensor=False)
        
        bb_rm1_logits = mc.predict(non_scaled_maligns, self.black_box_model, self.bb_model_path)
        rm1_eu_bn = rm1_eu[bb_rm1_logits == label_filter_BB]
        
        if rm1_eu_bn.shape[0] == 0:
            tf.print ("error en get_malign_filtered. muestras malignas que pasan BB son 0.")
            #ratio = ratio_ev_malign
        else:
            ratio= num_samples/rm1_eu_bn.shape[0]
            
            if (1.0/ratio) < 0.001:
                tf.print ("error en get_malign_filtered. ratio muestras malignas por debajo de 1%", (1.0/ratio), ratio)
            else:
                if rm1_eu_bn.shape[0] < num_samples:
                    tf.print ("error: no tengo suficientes muestras malignas para comparar",rm1_eu_bn.shape[0] ,  num_samples,vueltas)
                    nuevo_ratio=int(ratio)+1
                    tf.print ("nuevo ratio:",nuevo_ratio)
                    if vueltas < 2:
                        tf.print ("llamo a get_malign_filtered. vueltas:",vueltas, "nuevo ratio:", nuevo_ratio, 
                                  "ratio_ev_malign*nuevo_ratio", ratio_ev_malign*nuevo_ratio)
                        rm1_eu_bn= self.get_malign_filtered (num_muestras=num_samples, 
                                                                 ratio_ev_malign=ratio_ev_malign*nuevo_ratio, 
                                                                 vueltas=vueltas+1)
                    else:
                        tf.print ("NO HAY SUFICIENTES MUESTRAS MALIGNAS que PASEN filtro BB.")
                        tf.print ("Num muestras devueltas:",rm1_eu_bn.shape[0])
                else:
                    rm1_eu_bn= rm1_eu_bn[:num_samples]
            
        tf.print ("return rm1_eu_bn",rm1_eu_bn.shape)
        return rm1_eu_bn
            
    def distance_benign_to_malign_filtered (self,num_muestras=None,ratio_ev_malign=5,vueltas=0):
        
        if num_muestras != None:
            num_samples= num_muestras
        else:
            num_samples=self.k
        #print ("ratio_ev_malign",ratio_ev_malign)
        
        rb1_eu = mu.muestras.sample_examples(num_samples, class_label=BENIGN_LABEL)
        
               
        rm1_eu = mu.muestras.sample_examples(num_samples*ratio_ev_malign, class_label=MALIGN_LABEL)
        
        #Get preds Black Box
        non_scaled_maligns =  mu.muestras.inverse_normalise (rm1_eu, tensor=False)
        bb_rm1_logits = mc.predict(non_scaled_maligns, self.black_box_model, self.bb_model_path)
        rm1_eu_bn = rm1_eu[bb_rm1_logits == BENIGN_LABEL]
        
        
        self.metrics_dict["rb_rm-f"]["EU"][f"{self.it:>05}"] = 0.0 
        
        if rm1_eu_bn.shape[0] == 0:
            tf.print ("error en distance_benign_to_malign_filtered. Muestras malignas que pasan BB son 0.")
            #ratio = ratio_ev_malign
        else:
            ratio= num_samples/rm1_eu_bn.shape[0]
            
            #tf.print (f"self.metrics_dict['rb_rm-f']['EU']['{self.it:>05}']:",0.0)

            if (1.0/ratio) < 0.001:
                tf.print ("error en distance_benign_to_malign_filtered. ratio muestras malignas por debajo de 1%.", (1.0/ratio), ratio)
            else:
                if rm1_eu_bn.shape[0] < num_samples:
                    tf.print ("error: no tengo suficientes muestras malignas para comparar",rm1_eu_bn.shape[0] ,  num_samples,vueltas)
                    nuevo_ratio=int(ratio)+1
                    if vueltas < 2:
                        tf.print ("llamo a distance_benign_to_malign_filtered. vueltas:",vueltas, "nuevo ratio:", nuevo_ratio, 
                                  "ratio_ev_malign*nuevo_ratio", ratio_ev_malign*nuevo_ratio)
                        self.distance_benign_to_malign_filtered (num_muestras=num_muestras, 
                                                                 ratio_ev_malign=ratio_ev_malign*nuevo_ratio, 
                                                                 vueltas=vueltas+1)
                    else:
                        tf.print ("NO HAY SUFICIENTES MUESTRAS MALIGNAS que PASEN filtro BB.")
                        tf.print ("Num muestras devueltas:",rm1_eu_bn.shape[0])            

                else:
                    rm1_eu_bn= rm1_eu_bn[:num_samples]
                    tf.print ("distance_benign_to_malign_filtered. num_muestras:",rm1_eu_bn.shape)
                    v=mc.compute_metrics(rb1_eu, rm1_eu_bn, "EU", msg="rb1, rm1",debug=False, 
                                         tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
                    #v2=mc.compute_metrics(rm1_eu_bn,rb1_eu,  "EU", msg="rb1, rm1",debug=False)
                    tf.print (f"self.metrics_dict['rb_rm-f']['EU']['{self.it:>05}']:",v)
                    self.metrics_dict["rb_rm-f"]["EU"][f"{self.it:>05}"] = v  
            

      
    def debug_dst (self):
        '''
        tf.print ("******* Iniciales ")
        #mf-gf
        d1=mc.compute_metrics(gm1_eu, rm1_eu, m, msg="gf-mf",debug=True),  ## 32
        d2=mc.compute_metrics(rm1_eu, gm1_eu, m, msg="mf-gf",debug=True),  ## 23
        tf.print (">>>distancia (mf-gf) ojo solo en filtrada:",d1,d2)
        #m-gf
        rm1_eu_x = mu.muestras.sample_examples(NUM_SAMPLES_WA, class_label=MALIGN_LABEL)
        d1=mc.compute_metrics(gm1_eu, rm1_eu_x, m, msg="gf-m",debug=True),  ## 32
        d2=mc.compute_metrics(rm1_eu_x, gm1_eu, m, msg="m-gf",debug=True),  ## 23
        tf.print (">>>distancia (m-gf) ojo solo en filtrada:",d1,d2)    
        
        '''
        tf.print ("******* debug_dst Iniciales ")
        #
        gfb = self.get_adv_filter_x (1000,label=BENIGN_LABEL)
        if gfb != None:
            tf.print ("pedidos gfb 1000, devueltos:",gfb.shape[0])
            gfb=gfb[:1000] 
            assert gfb.shape[0] == 1000, "No hay muestras suficientes gfb (1000):"
        else:
            tf.print ("NO hay gfb")
        
        gfm = self.get_adv_filter_x (1000,label=MALIGN_LABEL)
        if gfm != None:
            tf.print ("pedidos gfm 1000, devueltos:",gfm.shape[0])
            gfm=gfm[:1000]
            assert gfm.shape[0] == 1000, "No hay muestras suficientes gfm (1000):"
        else: 
            tf.print ("NO hay gfm")
        
        g=self.sample_generator(1000)
        
        
        m="EU"
        rmf= self.get_malign_filtered (num_muestras=1000,label_filter_BB=BENIGN_LABEL)            
        #
        if gfb != None:
            d1=mc.compute_metrics(gfb, rmf, m, msg="gfb-mfb",debug=True,
                                  tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
            tf.print ("gfb-mfb:",d1)
        #
        if gfm != None:
            d1=mc.compute_metrics(gfm, rmf, m, msg="gfm-mfb",debug=True,
                                  tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
            tf.print ("gfm-mfb:",d1)
        #
        d1=mc.compute_metrics(g, rmf, m, msg="g-mfb",debug=True,
                              tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
        tf.print ("g-mfb:",d1)

        tf.print ("----------------")

        rmf= self.get_malign_filtered (num_muestras=1000,label_filter_BB=MALIGN_LABEL)
        #
        if gfb != None:
            d1=mc.compute_metrics(gfb, rmf, m, msg="gfb-mfm",debug=True,
                                  tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
            tf.print ("gfb-mfm:",d1)
        #
        if gfm != None:
            d1=mc.compute_metrics(gfm, rmf, m, msg="gfm-mfm",debug=True,
                                  tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
            tf.print ("gfm-mfm:",d1)
        #
        d1=mc.compute_metrics(g, rmf, m, msg="g-mfm",debug=True,
                              tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
        tf.print ("g-mfm:",d1)

        tf.print ("----------------")

        rm= mu.muestras.sample_examples(1000, class_label=MALIGN_LABEL)
        #
        if gfb != None:
            d1=mc.compute_metrics(gfb, rm, m, msg="gfb-m",debug=True,
                                  tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
            tf.print ("gfb-m:",d1)
        #
        if gfm != None:
            d1=mc.compute_metrics(gfm, rm, m, msg="gfm-m",debug=True,
                                  tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
            tf.print ("gfm-m:",d1)
        #
        d1=mc.compute_metrics(g, rm, m, msg="g-m",debug=True,
                              tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
        tf.print ("g-m:",d1)

        tf.print ("----------------")

        #---
        rmf1= self.get_malign_filtered (num_muestras=1000,label_filter_BB=BENIGN_LABEL)    
        rmf2= self.get_malign_filtered (num_muestras=1000,label_filter_BB=BENIGN_LABEL)
        d1=mc.compute_metrics(rmf1, rmf2, m, msg="mfb-mfb",debug=True,
                              tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
        tf.print ("mfb-mfb:",d1)
        #
        rmf1= self.get_malign_filtered (num_muestras=1000,label_filter_BB=MALIGN_LABEL)    
        rmf2= self.get_malign_filtered (num_muestras=1000,label_filter_BB=MALIGN_LABEL)
        d1=mc.compute_metrics(rmf1, rmf2, m, msg="mfm-mfm",debug=True,
                              tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
        tf.print ("mfm-mfm:",d1)
        #
        rmf1= self.get_malign_filtered (num_muestras=1000,label_filter_BB=BENIGN_LABEL)     
        rmf2= self.get_malign_filtered (num_muestras=1000,label_filter_BB=MALIGN_LABEL)
        d1=mc.compute_metrics(rmf1, rmf2, m, msg="mfb-mfm",debug=True,
                              tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
        tf.print ("mfb-mfm:",d1)
        #
        rm1= mu.muestras.sample_examples(1000, class_label=MALIGN_LABEL)
        rm2= mu.muestras.sample_examples(1000, class_label=MALIGN_LABEL)
        d1=mc.compute_metrics(rm1, rm2, m, msg="m-m",debug=True,
                              tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
        tf.print ("m-m:",d1)

        tf.print ("----------------")

        #----
        gfb1 = self.get_adv_filter_x (1000,label=BENIGN_LABEL)
        gfb2= self.get_adv_filter_x (1000,label=BENIGN_LABEL)
        if gfb1 != None and gfb2 != None:
            d1=mc.compute_metrics(gfb1, gfb2, m, msg="gfb-gfb",debug=True,
                                  tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
            tf.print ("gfb-gfb:",d1)
        #
        gfm1 = self.get_adv_filter_x (1000,label=MALIGN_LABEL)
        gfm2= self.get_adv_filter_x (1000,label=MALIGN_LABEL)
        if gfm1 != None and gfm2 != None:
            d1=mc.compute_metrics(gfm1, gfm2, m, msg="gfm-gfm",debug=True,
                                  tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
            tf.print ("gfm-gfm:",d1)
        #
        gfb1 = self.get_adv_filter_x (1000,label=BENIGN_LABEL)
        gfm2= self.get_adv_filter_x (1000,label=MALIGN_LABEL)
        if gfb1 != None and gfm2 != None:
            d1=mc.compute_metrics(gfb1, gfm2, m, msg="gfb-gfm",debug=True,
                                  tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
            tf.print ("gfb-gfm:",d1)
        #
        g1=self.sample_generator(num_samples=1000, training=True)
        g2=self.sample_generator(num_samples=1000, training=True)
        d1=mc.compute_metrics(g1, g2, m, msg="g-g",debug=True,
                              tipo_distancia=self.tipo_distancia, stochastic=self.stochastic)
        tf.print ("g-g:",d1)

        tf.print ("----------------")


    def distance_matrix(self, test=False, training=False, gm1=None, gm2=None, act_metrics=True, num_muestras=None, distance_matrix="distance_matrix",filtra_malign=False):
        """
        Calculates a square simmetric matrix with the distances considered.
        Saves it both in the object and in the file structure.
        """
        tf.print("DISTANCE_MATRIX: rb1-rb1, rm1-rm2, gm1-gm2, .. rb1-rm1, gm1-rb1, gm1-rm1")
        tf.print("using dataset test:", test)
        # metrics_it = 20
        metrics_it = 1
        # metrics_types = ["WS", "JS", "EU"]
        # m_d_type = {"WS": 0, "JS": 1, "EU": 2}
        metrics_types = ["WS", "EU", "WS-SH"]
        m_d_type = {"WS": 0, "EU": 1, "WS-SH":2}

        
        
        
        #NUM_SAMPLES_WA = 500
        
        # OJO_DEBUG
        # ---------
        # self.k=15
        # ---------
        
        if num_muestras != None:
            NUM_SAMPLES_WA= num_muestras
        else:
            NUM_SAMPLES_WA=self.k
        
        #NUM_SAMPLES_WA=512
        
        if gm1 != None:
            gm1=gm1[:NUM_SAMPLES_WA]
        if gm2 != None:
            gm2=gm2[:NUM_SAMPLES_WA]
        
        tf.print ("NUM_SAMPLES_WA:",NUM_SAMPLES_WA)

        distances = np.ndarray(shape=(metrics_it, len(metrics_types), 9), dtype=float)

        if self.k > 1000:
            tf.print("WARNING: Sampling more than 1000 samples")

        tstart = time.time()
        for i in range(metrics_it):
            # start = time.time()
            # Sample examples
            #tf.print("ronda Metric: ", i)
            #print("Gen samples rb1, rb2, rm1, rm2")

            gm1_eu = gm1
            gm2_eu = gm2

            gm1_wa = gm1
            gm2_wa = gm2
            
            
            
            if not filtra_malign and (gm1_eu == None):
                gm1_eu = self.sample_generator(num_samples=NUM_SAMPLES_WA, training=True)
                tf.print ("distance_matrix generando muestras gm1_eu")
            if not filtra_malign and (gm2_eu == None):
                gm2_eu = self.sample_generator(num_samples=NUM_SAMPLES_WA, training=True)
                tf.print ("distance_matrix generando muestras gm2_eu")
            
            tf.print ("tipos gm1 y gm2")
            tf.print (gm1_eu.shape, type(gm1_eu))
            tf.print (gm2_eu.shape, type(gm2_eu))
            gm1_eu=np.array(gm1_eu)
            gm2_eu=np.array(gm2_eu)
            gm1_wa=gm1_eu
            gm2_wa=gm2_eu
            tf.print (gm1_eu.shape, type(gm1_eu))
            tf.print (gm2_eu.shape, type(gm2_eu))
            
            rb1_eu = mu.muestras.sample_examples(NUM_SAMPLES_WA, class_label=BENIGN_LABEL)
            rb2_eu = mu.muestras.sample_examples(NUM_SAMPLES_WA, class_label=BENIGN_LABEL)
            if not filtra_malign:
                tf.print ("get_malign SIN filtrar rm_eu")
                rm1_eu = mu.muestras.sample_examples(NUM_SAMPLES_WA, class_label=MALIGN_LABEL)
                rm2_eu = mu.muestras.sample_examples(NUM_SAMPLES_WA, class_label=MALIGN_LABEL)
            else:
                tf.print ("get_malign_filtered rm_eu")
                rm1_eu= self.get_malign_filtered (num_muestras=NUM_SAMPLES_WA)
                tf.print ("shape rm1_eu",rm1_eu.shape)
                rm2_eu= self.get_malign_filtered (num_muestras=NUM_SAMPLES_WA)
                tf.print ("shape rm2_eu",rm2_eu.shape)

            '''
            if gm1_wa == None:
                gm1_wa = self.sample_generator(num_samples=NUM_SAMPLES_WA, training=training)
            else:
                gm1_wa = gm1_wa[:NUM_SAMPLES_WA]
            if gm2_wa == None:
                gm2_wa = self.sample_generator(num_samples=NUM_SAMPLES_WA, training=training)
            else:
                gm2_wa = gm2_wa[:NUM_SAMPLES_WA]
            
            '''
            
            rb1_wa= rb1_eu
            rb2_wa= rb2_eu
            rm1_wa= rm1_eu
            rm2_wa= rm2_eu
            hay_m= ((rm1_eu.shape[0] == rb1_eu.shape[0]) and (rm2_eu.shape[0] == rb1_eu.shape[0]) )
            hay_g= ((gm1_eu.shape[0] == rb1_eu.shape[0]) and (gm2_eu.shape[0] == rb1_eu.shape[0]) )
                
            
            '''
            rb1_wa = mu.muestras.sample_examples(NUM_SAMPLES_WA, class_label=BENIGN_LABEL)
            rb2_wa = mu.muestras.sample_examples(NUM_SAMPLES_WA, class_label=BENIGN_LABEL)
            if not filtra_malign:
                rm1_wa = mu.muestras.sample_examples(NUM_SAMPLES_WA, class_label=MALIGN_LABEL)
                rm2_wa = mu.muestras.sample_examples(NUM_SAMPLES_WA, class_label=MALIGN_LABEL)
            else:
                tf.print ("get_malign_filtered rm_wa")
                rm1_wa= self.get_malign_filtered (num_muestras=NUM_SAMPLES_WA)
                rm2_wa= self.get_malign_filtered (num_muestras=NUM_SAMPLES_WA)
            '''
            

            # Compute distances
            for m in metrics_types:
                tf.print("compute metric:", m)

                if m == "EU":
                    tipo_dist=self.tipo_distancia
                    
                    # Debug
                    if self.epoch > 20000:
                        self.debug_dst()
                                        
                    tf.print ("******* Matriz ")
                    
                    time_distances_euclidean_start = time.time()
                    # OJo luego se ordenan bien con mc.confeccionate_matrix(distances, m_d_type["EU"])
                    distances[i, m_d_type[m], :] = (
                        mc.compute_metrics(rb1_eu, rb2_eu, m, msg="rb1, rb2",dist_alb=True, 
                                           tipo_distancia=tipo_dist, stochastic=self.stochastic),  ## 11
                        mc.compute_metrics(rm1_eu, rm2_eu, m, msg="rm1, rm2",dist_alb=True, 
                                           tipo_distancia=tipo_dist, stochastic=self.stochastic) if hay_m else 0., ## 22
                        mc.compute_metrics(gm1_eu, gm2_eu, m, msg="gm1, gm2",dist_alb=True, 
                                           tipo_distancia=tipo_dist, stochastic=self.stochastic) if hay_g else 0., ## 33
                        
                        mc.compute_metrics(rb1_eu, rm1_eu, m, msg="rb1, rm1",dist_alb=True, 
                                           tipo_distancia=tipo_dist, stochastic=self.stochastic) if hay_m else 0.,  ## 12
                        mc.compute_metrics(rb1_eu, gm1_eu, m, msg="gm1, rb1",dist_alb=True, 
                                           tipo_distancia=tipo_dist, stochastic=self.stochastic) if hay_g else 0., ## 13
                        mc.compute_metrics(rm1_eu, gm1_eu, m, msg="gm1, rm1",dist_alb=True, 
                                           tipo_distancia=tipo_dist, stochastic=self.stochastic) if hay_m else 0., ## 23
                        
                        mc.compute_metrics(rm2_eu, rb2_eu, m, msg="rm2, rb2",dist_alb=True, 
                                           tipo_distancia=tipo_dist,stochastic=self.stochastic) if hay_m else 0., ## 21
                        mc.compute_metrics(gm2_eu, rb2_eu, m, msg="rb2, gm2",dist_alb=True, 
                                           tipo_distancia=tipo_dist, stochastic=self.stochastic) if hay_g else 0., ## 31
                        mc.compute_metrics(gm2_eu, rm2_eu,  m, msg="rm2, gm2",dist_alb=True, 
                                           tipo_distancia=tipo_dist, stochastic=self.stochastic) if hay_m else 0. ## 32
                    )
                    
                else:
                    # m -> metric
                    time_distances_wa_start = time.time()
                    distances[i, m_d_type[m], :] = (
                        mc.compute_metrics(rb1_wa, rb2_wa, m),  ## 11
                        mc.compute_metrics(rm1_wa, rm2_wa, m) if hay_m else 0.,  ## 22
                        mc.compute_metrics(gm1_wa, gm2_wa, m) if hay_g else 0.,  ## 33
                        mc.compute_metrics(rb1_wa, rm1_wa, m) if hay_m else 0.,  ## 12
                        mc.compute_metrics(rb1_wa, gm1_wa, m) if hay_g else 0.,  ## 13
                        mc.compute_metrics(rm1_wa, gm1_wa, m) if hay_m else 0.,  ## 23
                        mc.compute_metrics(rm2_wa, rb2_wa, m) if hay_m else 0.,  ## 21
                        mc.compute_metrics(gm2_wa, rb2_wa, m) if hay_g else 0.,  ## 31
                        mc.compute_metrics(gm2_wa, rm2_wa, m) if hay_m else 0.,  ## 32
                    )
                    '''
                    distances[i, m_d_type[m], :] = (
                        np.mean(mc.compute_metrics(rb1_wa, rb2_wa, m)),  ## 11
                        np.mean(mc.compute_metrics(rm1_wa, rm2_wa, m)),  ## 22
                        np.mean(mc.compute_metrics(gm1_wa, gm2_wa, m)),  ## 33
                        np.mean(mc.compute_metrics(rb1_wa, rm1_wa, m)),  ## 12
                        np.mean(mc.compute_metrics(gm1_wa, rb1_wa, m)),  ## 31
                        np.mean(mc.compute_metrics(gm1_wa, rm1_wa, m)),  ## 32
                        np.mean(mc.compute_metrics(rm2_wa, rb2_wa, m)),  ## 21
                        np.mean(mc.compute_metrics(rb2_wa, gm2_wa, m)),  ## 31
                        np.mean(mc.compute_metrics(rm2_wa, gm2_wa, m)),  ## 23
                    )
                    
                    '''
                 
                distances[i, m_d_type[m], :]= [f"{e:.2f}" for e in distances[i, m_d_type[m], :]]
                tf.print ("distances[i, m_d_type[m], :]\n",distances[i, m_d_type[m], :])

        end = time.time() - tstart
        tf.print(f"Total metrics loop time: {end}")

        distance_matrix_ws = mc.confeccionate_matrix(distances, m_d_type["WS"])
        # distance_matrix_js = mc.confeccionate_matrix(distances, m_d_type["JS"])
        tf.print ("distancia antes:",distances[0, m_d_type["EU"], -3:] )
        distance_matrix_eu = mc.confeccionate_matrix(distances, m_d_type["EU"])
        tf.print ("distancia despues:",distance_matrix_eu)
        distance_matrix_ws_sh = mc.confeccionate_matrix(distances, m_d_type["WS-SH"])

        if act_metrics:
            self.metrics_dict[distance_matrix]["WS"][f"{self.it:>05}"] = distance_matrix_ws
            self.metrics_dict[distance_matrix]["WS-SH"][f"{self.it:>05}"] = distance_matrix_ws_sh
            self.metrics_dict[distance_matrix]["EU"][f"{self.it:>05}"] = distance_matrix_eu
        
            tf.print (f"Actualizado self.metrics_dict[{distance_matrix}]\n",self.metrics_dict[distance_matrix]["EU"][f"{self.it:>05}"])

        # self.bb_hits()

        tf.print("Distance matrix WS:\n", str(distance_matrix_ws))
        # tf.print("Distance matrix JS:\n",str(distance_matrix_js))
        tf.print("Distance matrix EU:\n", str(distance_matrix_eu))
        tf.print("Distance matrix WS-SH:\n", str(distance_matrix_ws_sh))

        return distance_matrix_eu, distance_matrix_ws
    
    def bb_hits(self, training=False):
        tf.print("-------\nbb_hits sobre TESTING \n-------")
        training=True
        # xx=mu.muestras.dataset[mu.MUESTRA_TEST]["x"]

        # var = self.scaler.var_
        # std = np.sqrt(var)
        # mean = self.scaler.mean_

        # # convert to tensor
        # var = tf.convert_to_tensor(var, dtype=tf.float32)
        # mean = tf.convert_to_tensor(mean, dtype=tf.float32)

        # non_scaled_xx = tf.add(
        #     tf.cast(tf.multiply(self._xx, std), tf.float32), tf.cast(mean, tf.float32)
        # )  # REVIEW: de donde sale self._xx?

        # y_pred = mc.predict(non_scaled_xx, self.black_box_model, self.bb_model_path)
        # m = confusion_matrix(self._yy, y_pred, labels=[0, 1, 2])
        # print("Matriz con datos reales + ruido. Dataset de testing\n", m)

        # Generate sample of size k with generator:
        tf.print("ahora probamos blackbox y el discriminador con datos sinteticos (malignos retocados por generadora)")
        gm, random_latent_vectors, malign_inputs = self.sample_generator_x(num_samples=self.k, training=training)

        '''
        var = self.scaler.var_[self.feats]
        std = np.sqrt(var)
        mean = self.scaler.mean_[self.feats]

        # convert to tensor
        var = tf.convert_to_tensor(var, dtype=tf.float32)
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)

        non_scaled_gm = tf.add(tf.cast(tf.multiply(gm, std), tf.float32), tf.cast(mean, tf.float32))
        '''
        
        non_scaled_gm= mu.muestras.inverse_normalise (gm,tensor=True)
        tf.print ("las shapes",gm.shape,non_scaled_gm.shape)

        # Save generated samples
        np.save(self.samples_dir / f"gm_it_{self.it:>05}.npy", gm)
        np.save(self.samples_dir / f"random_latent_vectors_it_{self.it:>05}.npy", random_latent_vectors)
        np.save(self.samples_dir / f"malign_inputs_it_{self.it:>05}.npy", malign_inputs)

        # gm=tf.add(random_latent_vectors*0.0001,malign_inputs)
        # gm=malign_inputs
        if self.debug:
            print("Salida del generador gm con training a :", training, "\n", gm)

        # Calculate BB confusion matrix
        y_pred = mc.predict(non_scaled_gm, self.black_box_model, self.bb_model_path)

        np.save(self.preds_dir / f"bb_preds_it_{self.it:>05}.npy", y_pred)
        # print ("y_pred",y_pred)
        # print ("y_label",[MALIGN_LABEL] * self.k)
        cm = confusion_matrix([MALIGN_LABEL] * self.k, y_pred, labels=[0, 1])
        tf.print("Black Box confusion matrix: input gm")
        tf.print(cm)
        
        ######
        for tr in [True,False]:
            malign_inputs = mu.muestras.sample_examples(self.k, class_label=MALIGN_LABEL)  # [:, f_UPC]
            # random_latent_vectors = tf.random.normal(shape=(malign_inputs.shape[0], self.latent_dim))
            random_latent_vectors = tf.random.uniform(
                shape=(malign_inputs.shape[0], self.latent_dim), minval=self.min_rnd, maxval=self.max_rnd
            )
            if self.solo_noise:
                adversarials_examples = self.generator(random_latent_vectors, training=tr)
            else:                
                adversarials_examples = self.generator([random_latent_vectors,malign_inputs], training=tr)

            # adversarials_labels_cat = [MALIGN_LABEL]*len(adversarials_examples)

            # Unscale data
            
            non_scaled_adversarials_examples = mu.muestras.inverse_normalise (adversarials_examples,tensor=True)
            '''
            var = self.scaler.var_[self.feats]
            std = np.sqrt(var)
            mean = self.scaler.mean_[self.feats]

            # convert to tensor
            var = tf.convert_to_tensor(var, dtype=tf.float32)
            mean = tf.convert_to_tensor(mean, dtype=tf.float32)

            non_scaled_adversarials_examples = tf.add(
                tf.cast(tf.multiply(adversarials_examples, std), tf.float32), tf.cast(mean, tf.float32)
            )
            
            '''

            ## Ojo, son malignos, asi que los podriamos etiquetar directamente como malignos
            # Labels are obtained from the BlackBox model (we do not know the labels)
            #adversarials_labels_cat = mc.predict(
            #    non_scaled_adversarials_examples, self.black_box_model, self.bb_model_path,
            #).astype(int)
            adversarials_labels_cat = mc.predict(
                non_scaled_adversarials_examples, self.black_box_model, self.bb_model_path,
            )

            tf.print("MALIGN_ADV", adversarials_examples.shape[0], "GEn training=",tr)
            labels_ok=np.array ([MALIGN_LABEL]*adversarials_examples.shape[0])
            m = confusion_matrix(labels_ok, adversarials_labels_cat, labels=[0, 1])
            tf.print ("Confusion de valores generados GM sobre BB\n",m)
        
        #######
        
        self.metrics_dict["CM"]["bb_class_malign"][f"{self.it:>05}"] = cm[MALIGN_LABEL]
        # tf.print(str(np.matrix(f'{cm[0]};{cm[1]};{cm[2]}')))
        # self.bb_class_malign.append(cm[1])

        # Contar los BB hits (Malignos que pasan como benignos en el BB)
        bh = np.sum(y_pred == ([BENIGN_LABEL] * self.k))
        mh = np.sum(y_pred == ([MALIGN_LABEL] * self.k))
        nh = np.sum(y_pred == ([RUIDO_LABEL] * self.k))
        ratio_bh = round(bh / self.k, 4)
        ratio_mh = round(mh / self.k, 4)
        ratio_nh = round(nh / self.k, 4)
        tf.print("Ratios BB: **ratio_bh**,ratio_mh,ratio_nh:", "**",ratio_bh, "**",ratio_mh, ratio_nh)
        self.metrics_dict["BB_hits"][f"{self.it:>05}"] = ratio_bh
        
        #--------------
        # Calcular ratio sobre datos reales malignos (baseline)
        malign_data = mu.muestras.sample_examples(self.k, class_label=MALIGN_LABEL)
        
        '''
        var = self.scaler.var_[self.feats]
        std = np.sqrt(var)
        mean = self.scaler.mean_[self.feats]

        # convert to tensor
        var = tf.convert_to_tensor(var, dtype=tf.float32)
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)

        non_scaled_gm = malign_data*std+mean
        '''
        
        non_scaled_m = mu.muestras.inverse_normalise (malign_data,tensor=False)
        
        tf.print ("las shapes",malign_data.shape,non_scaled_gm.shape)

        # Calculate BB confusion matrix
        y_pred = mc.predict(non_scaled_m, self.black_box_model, self.bb_model_path)
        bh = np.sum(y_pred == ([BENIGN_LABEL] * self.k))
        ratio_bh = round(bh / self.k, 4)
        self.metrics_dict["BB_hits_real"][f"{self.it:>05}"] = ratio_bh
        tf.print ("MALIGNOS que pasan BB", f"self.metrics_dict['BB_hits_real'][f'{self.it:>05}']",ratio_bh)
        #--------------
        
        # Calculate discriminator confusion matrix
        # Con predict
        inferences = self.discriminator.predict(non_scaled_gm)
        y_pred = np.argmax(inferences, axis=1)

        np.save(self.preds_dir / f"disc_preds_it_{self.it:>05}.npy", y_pred)
        cm = confusion_matrix([MALIGN_LABEL] * self.k, y_pred, labels=[0, 1])
        tf.print("Discriminator confusion matrix con predict: input gm")
        tf.print(cm)

        # self.d_class_malign.append(cm[1])
        self.metrics_dict["CM"]["d_class_malign"][f"{self.it:>05}"] = cm[MALIGN_LABEL]
        #with open(self.metrics_dir / f"metrics_dict_it_{self.it:>05}.plk", "wb") as f:
        #    pickle.dump(self.metrics_dict, f)

        return ratio_bh, ratio_mh, ratio_nh

        """
        inferences = self.discriminator(gm, training=True)
        y_pred= np.argmax(inferences,axis=1)    
        cm = confusion_matrix([MALIGN_LABEL] * self.k, y_pred, labels=[0, 1, 2])
        print("Discriminator confusion matrix  con constructor y training=True: input gm")
        print (cm)
        
        
        
        #tf.print(str(np.matrix(f'{cm[0]};{cm[1]};{cm[2]}')))
        # Con Training=False
        inferences = self.discriminator(gm, training=False)
        y_pred= np.argmax(inferences,axis=1)    
        cm = confusion_matrix([MALIGN_LABEL] * self.k, y_pred, labels=[0, 1, 2])
        print("Discriminator confusion matrix  con constructor y training=False: input gm")
        print (cm)
        # Con predict
        inferences = self.discriminator.predict (gm)
        y_pred= np.argmax(inferences,axis=1)    
        cm = confusion_matrix([MALIGN_LABEL] * self.k, y_pred, labels=[0, 1, 2])
        print("Discriminator confusion matrix con predict: input gm")
        print (cm)
        
        
        #Otra prueba igual a train
        print ("pruebo lo mismo pero usando el codigo del train_step de la generadora con Training a True")
        print ("uso random_latent_vectors,malign_inputs iniciales")
        self.gen_test_cm ("bbhits check como en train",
                          random_latent_vectors,malign_inputs,"3")
        
        batch_size=self.k
        #random_latent_vectors = tf.random.normal(shape=(batch_size,self.latent_dim))
        #malign_inputs = mc.sample_examples(self.ds_dir,int(np.floor(batch_size)),cl=MALIGN_LABEL)
        gm,random_latent_vectors,malign_inputs  = self.sample_generator_x (num_samples=self.k) #No uso las gm, training=X da igual
        print ("genero nuevos random_latent_vectors,malign_inputs")
        self.gen_test_cm ("bbhits check como en train",random_latent_vectors,malign_inputs,"4")
        
        
        #self.d_class_malign.append(cm[1])
        self.metrics_dict["CM"]["d_class_malign"][
            f"{self.it:>05}"] = cm[MALIGN_LABEL]
        """

        return ratio_bh, ratio_mh, ratio_nh

    

    def evaluate_clustering(self):
        print(f"Clustering with real benign and malign data")

        # Inter-MSE distances
        benign_data = mu.muestras.sample_examples(self.k, class_label=BENIGN_LABEL)
        malign_data = mu.muestras.sample_examples(self.k, class_label=MALIGN_LABEL)
        generated_malign_data, _random_latent_vectors, _malign_inputs = self.sample_generator_x(
            num_samples=self.k, training=False
        )

        kmeans_real = KMeansHelper(
            num_clusters=self.clustering_k, classes="benign_malign", dataset=self.dataset, output_dir=self.ds_dir
        )

        kmeans_real.load()
        kmeans_real.report(sort_cluster_centers=False)

        num_samples_per_cluster_generated = kmeans_real.calculate_num_samples_per_clusters_with_outliers(
            generated_malign_data, real_data=False
        )
        num_samples_per_cluster_benign = kmeans_real.calculate_num_samples_per_clusters_with_outliers(benign_data)
        num_samples_per_cluster_malign = kmeans_real.calculate_num_samples_per_clusters_with_outliers(malign_data)

        mse_bm = np.mean((num_samples_per_cluster_benign - num_samples_per_cluster_malign) ** 2)
        mse_bg = np.mean((num_samples_per_cluster_benign - num_samples_per_cluster_generated) ** 2)
        mse_mg = np.mean((num_samples_per_cluster_malign - num_samples_per_cluster_generated) ** 2)

        print(f"Evaluation Mean Squared Error (Clustering BM): {mse_bm}")
        print(f"Evaluation Mean Squared Error (Clustering BG): {mse_bg}")
        print(f"Evaluation Mean Squared Error (Clustering MG): {mse_mg}")

        # Intra-MSE distances
        benign_data_2 = mu.muestras.sample_examples(self.k, class_label=BENIGN_LABEL)
        malign_data_2 = mu.muestras.sample_examples(self.k, class_label=MALIGN_LABEL)
        generated_malign_data_2, _random_latent_vectors, _malign_inputs = self.sample_generator_x(
            num_samples=self.k, training=False
        )

        num_samples_per_cluster_benign_2 = kmeans_real.calculate_num_samples_per_clusters_with_outliers(benign_data_2)
        num_samples_per_cluster_malign_2 = kmeans_real.calculate_num_samples_per_clusters_with_outliers(malign_data_2)
        num_samples_per_cluster_generated_2 = kmeans_real.calculate_num_samples_per_clusters_with_outliers(
            generated_malign_data_2
        )

        mse_bb = np.mean((num_samples_per_cluster_benign - num_samples_per_cluster_benign_2) ** 2)
        mse_mm = np.mean((num_samples_per_cluster_malign - num_samples_per_cluster_malign_2) ** 2)
        mse_gg = np.mean((num_samples_per_cluster_generated - num_samples_per_cluster_generated_2) ** 2)

        print(f"Evaluation Mean Squared Error (Clustering BB): {mse_bb}")
        print(f"Evaluation Mean Squared Error (Clustering MM): {mse_mm}")
        print(f"Evaluation Mean Squared Error (Clustering GG): {mse_gg}")

        # create distance matrix (3x3)
        first_row = [mse_bb, mse_bm, mse_bg]
        second_row = [mse_bm, mse_mm, mse_mg]
        third_row = [mse_bg, mse_mg, mse_gg]

        mse_matrix = np.matrix([first_row, second_row, third_row])

        assert mse_matrix.shape == (3, 3)

        # save the Mean Squared Errors in self.metrics_dict
        self.metrics_dict["clustering_mse"][f"{self.it:>05}"] = mse_matrix

    def get_adv_filter_x (self, batch_size, label=BENIGN_LABEL):
        
        # Devuelve exactamente batch_size filtrados por el BB
        
        tot=batch_size
        acc=0
        lista=[]
        ratio_t=0
        veces=0
        while (tot-acc) > 0:
            pendiente=min(tot-acc,1000)
            gf,gf_nonscaled,ratio = self.get_adv_filter_bb_x (batch_size,label=label)
            
            if ratio == 0 :  #assert ratio >0, "ERROR. get_adv_filter_x, ratio 0 en get_adv_filter_bb_x "
                return None
                
            ratio_t+= ratio
            veces+=1
            lista.append (gf)
            acc+= gf.shape[0]
        gf_tot=tf.concat (lista, axis=0)
        gf_tot=gf_tot[:tot]
        tf.print ("gf_tot:",gf_tot.shape, "ratio:",round(ratio_t/veces,3))
        return gf_tot
        
    def get_adv_filter_bb_x (self, batch_size, label=BENIGN_LABEL):
        
        # Pides batch_size y te devuelve x filtrados por el BB.
        # Se generan a golpe de mini_batch para no saturar la memoria de la GPU
        
        # random_latent_vectors = tf.random.normal(shape=(batch_size,self.latent_dim))
        #tf.print ("get_adv_filter_bb me piden:",batch_size)
        mini_batch=1024
        training=True
        tot=batch_size
        adversarial_examples=[]
        while tot >0:
            bs=min(tot,mini_batch)
            #random_latent_vectors = tf.random.uniform(shape=(bs, self.latent_dim), minval=-1, maxval=1)
            #malign_inputs = mu.muestras.sample_examples(bs, class_label=MALIGN_LABEL)
            #advs=self.generator((random_latent_vectors, malign_inputs), training=training)
            gm, random_latent_vectors, malign_inputs = self.sample_generator_x(num_samples=bs, training=training)
            advs=gm
            #tf.print ("advs:",type(advs),advs.shape)
            adversarial_examples.append(advs)
            tot-=bs
            
        adversarial_examples = tf.concat (adversarial_examples, axis=0)
        #tf.print ("advs:",type(adversarial_examples),adversarial_examples.shape)
        assert adversarial_examples.shape[0] == batch_size
        assert adversarial_examples.shape[1] == self.latent_dim
        
        '''
        var = self.scaler.var_[self.feats]
        std = np.sqrt(var)
        mean = self.scaler.mean_[self.feats]

        # convert to tensor
        var = tf.convert_to_tensor(var, dtype=tf.float32)
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)

        non_scaled_adversarial_examples = tf.add(
            tf.cast(tf.multiply(adversarial_examples, std), tf.float32), tf.cast(mean, tf.float32)
        )
        '''
        
        non_scaled_adversarial_examples= mu.muestras.inverse_normalise (adversarial_examples,tensor=True)

        bb_adv_logits = mc.predict(non_scaled_adversarial_examples, self.black_box_model, self.bb_model_path)
        # print ("Generacion de malignos_adv por generadora. Resultado en BB-RF bb_adv_logits",bb_adv_logits)
        non_scaled_adv_examples_bn = non_scaled_adversarial_examples[bb_adv_logits == label]
        adv_examples_bn = adversarial_examples[bb_adv_logits == label]
        lab= "BENIGN" if label == BENIGN_LABEL else "MALIGN"
        #tf.print(f"obtengo filtradas por BB con etiq:{lab} tot examples: {len(bb_adv_logits)}, filtrados {len(adv_examples_bn)}")
        ratio= len(adv_examples_bn)/ len(bb_adv_logits)
        #tf.print (f"get_adv_filter_bb_x me piden:{batch_size} obtengo filtradas:{len(adv_examples_bn)} con ratio:{ratio}")
        return adv_examples_bn,non_scaled_adv_examples_bn,ratio
    
    def get_adv_filter_bb (self, batch_size, label=BENIGN_LABEL):
        adv_examples_bn,non_scaled_adv_examples_bn,ratio=  self.get_adv_filter_bb_x (batch_size, label=label)
        return adv_examples_bn,non_scaled_adv_examples_bn

    def distances_bn(self,act_metrics=False,num_muestras=None,distance_matrix="distance_matrix_filter"):
        
        if num_muestras != None:
            num_samples= num_muestras
        else:
            num_samples= self.k
            
        #num_samples = self.k
        
        # Calcula ratio
        gm1,_ = self.get_adv_filter_bb(num_samples)
        gm2,_ = self.get_adv_filter_bb(num_samples)
        ratio1 = round(len(gm1) / num_samples, 4)
        ratio2 = round(len(gm2) / num_samples, 4)
        dist_null = np.reshape([-1] * (3 * 3), (3, 3))
        tf.print("DISTANCES_BN. ratio mal_adv que pasan el BB como benignos:", ratio1, ratio2)
        r = (ratio1 + ratio2) / 2
        if r > 0.01:
            gm1,_ = self.get_adv_filter_bb(int(num_samples * 1.5 / r))
            gm2,_ = self.get_adv_filter_bb(int(num_samples * 1.5 / r))
            if len(gm1) < self.k or len(gm2) < self.k:
                tf.print("distance_bn Error, no se pueden generar las muestras necesarias", self.k, "solo hay:", len(gm1), len(gm2))
                distance_matrix_eu, distance_matrix_ws = (dist_null, dist_null)
            else:
                gm1 = gm1[: self.k]
                gm2 = gm2[: self.k]
                distance_matrix_eu, distance_matrix_ws = self.distance_matrix(
                    test=True, training=True, gm1=gm1, gm2=gm2, act_metrics=act_metrics, distance_matrix=distance_matrix, filtra_malign=True
                )
            """
            # Saca todas las necesarias para la matriz
            gm1=self.get_num_samples_adv_filter (num_samples)
            gm2=self.get_num_samples_adv_filter (num_samples)
            assert len(gm1) == len(gm2) and len(gm1) == self.k
            self.distance_matrix(test=True,training=False,gm1=gm1,gm2=gm2)
            """
        else:
            tf.print("distance_bn Error, no se pueden generar las muestras necesarias", self.k, "solo hay:", len(gm1), len(gm2))
            distance_matrix_eu, distance_matrix_ws = (dist_null, dist_null)

        return distance_matrix_eu, distance_matrix_ws

    def get_num_samples_adv_filter(self, num_muestras):
        lista = []
        tot = 0
        while tot < num_muestras:
            malign_adv_filter_bb = self.get_adv_filter_bb(num_muestras)
            lista.append(malign_adv_filter_bb)
            tot += malign_adv_filter_bb.shape[0]

        lista = np.concatenate(lista, axis=0)
        return lista[:num_muestras]
    
    
    def plot_histograms_new(self,df_real,df_sm, alpha={"real":None,"sm":None}, name = None,x_axis=None,\
                            bins=[None,None],hist=True,kde=True,rango=None,rango_bins=[],N_BINS=100):

        '''
          if np.std(df_real) >0.00001:
                bins = 50 if len(np.histogram_bin_edges(df_real, bins="fd")) > 50 else "fd"
            elif np.std(xx[:,j]) >0.00001:

        '''
        if len(rango_bins) == 0:
            n_bins=N_BINS
            print ("NUM bins:",n_bins)
        else:
            n_bins=rango_bins

        suggested_n_bins=len(np.histogram_bin_edges(df_real, bins="fd"))
        #print ("suggested bins:",suggested_n_bins)
        #n_bins = n_bins if suggested_n_bins > n_bins else suggested_n_bins
        bins=[n_bins,n_bins]


        df_real0_clip = apl_per (df_real, alpha["real"]) if alpha["real"] != None else df_real
        #df_lin0_clip = apl_per (df_lin, alpha["lin"]) if alpha["lin"] != None else df_lin
        df_sm0_clip = apl_per (df_sm, alpha["sm"]) if alpha["sm"] != None else df_sm
        
        #tf.print ("plot_histograms_new shapes",df_real0_clip.shape,df_sm0_clip.shape)
        if rango == None:
            rango= (np.min([df_real0_clip,df_sm0_clip]),np.max([df_real0_clip,df_sm0_clip]))
            if "MaxMin" in str(mu.muestras.scaler_path):
                rango=(np.min([0.,rango[0]]),np.max([1.,rango[1]]))
                
            tf.print ("Rango distplot:",rango)

        #plt.figure(figsize=(30, 10))
        #print ("Longs:", len(df_real0_clip),len(df_lin0_clip),len(df_sm0_clip))
        #print ("Longs:", len(df_real0_clip),len(df_sm0_clip))

        #sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
        sns.set(rc={"figure.dpi":100, 'savefig.dpi':100})
        sns.set_style('darkgrid')

        #plt.legend(labels=["Real","Linear", "Custom"])
        plt.legend(labels=[name])
        if x_axis!= None:
            plt.xlim(x_axis)
        plt.xlabel('Values')

        #if not name is None:
        #    plt.savefig(name + '.svg')
        
      
        for hist,kde in [[True,False],[False,True]]:
            
            #hist,kde=[True,False]
            #data1, kde=True, hist=True, kde_kws={"clip":(0,10)}, hist_kws={"range":(0,10)}
            if hist or np.std(df_real0_clip) > 0.00001:
                #sns.distplot(df_real0_clip, hist=hist,kde=kde,bins=bins[0],kde_kws={"lw": 1.5} ,hist_kws={"range":range})
                if hist:
                    #plt.figure(figsize=(30, 10))
                    plt.figure(figsize=(18, 6))
                    
                    datos={ "Synthetic":df_sm0_clip, "Real":df_real0_clip}
                    #sns.displot(data=datos, x="Real", color="orange", kind="hist" , kde=False, binrange=rango, bins=bins[0])
                    #sns.displot(data=datos, x="Synthetic", color="blue", kind="hist" , kde=False, binrange=rango, bins=bins[0])
                    sns.displot(data=pd.DataFrame(datos), kind="hist" , kde=False, binrange=rango, bins=bins[0], height=6, aspect=1.3, color=[ "orange","blue"], legend=False)
                    plt.legend(["Real","Synthetic"])
                    
                else:
                    plt.figure(figsize=(9, 6))
                    sns.kdeplot(df_real0_clip, label="Real", fill=False,clip=rango, color="orange")  # shade=True rellena el área bajo la curva KDE
                    sns.kdeplot(df_sm0_clip, label="Synthetic", fill=False,clip=rango, color="blue")
                    plt.legend(["Real","Synthetic"])
                    
                    


                # Ajustar los límites del eje x si se necesita
                #plt.xlim(rango)
                '''
                if hist:
                    sns.displot(df_real0_clip, bins=bins[0], kde=False, height=6, aspect=1.5)
                else:
                    sns.displot(df_real0_clip, kde=True, height=6, aspect=1.5)
                '''
            else: 
                print ("no se puede pintar valor REAL")
            #sns.distplot(df_lin0_clip,hist=hist,kde=kde,bins=bins[1],kde_kws={"lw": 1.5, "color":"r"},hist_kws={"color": "r"})
            if hist or np.std(df_sm0_clip) > 0.00001:
                #sns.distplot(df_sm0_clip, hist=hist,kde=kde,bins=bins[0],kde_kws={"lw": 1.5} ,hist_kws={"range":range})
                #sns.displot(df_sm0_clip, kde=kde, bins=bins[0], height=6, aspect=1.1, color="blue")
                '''
                if hist:
                    sns.displot(df_sm0_clip, bins=bins[0], kde=False, height=6, aspect=1.5)
                else:
                    sns.displot(df_sm0_clip, kde=True, height=6, aspect=1.5)
                '''
            else: 
                print ("no se puede pintar valor SYN")
                
            plt.xlim(rango)
            
            
            
            #if hist or np.std(df_sm0_clip) > 0.00001:
            #    sns.distplot(df_sm0_clip,hist=hist,kde=kde,bins=bins[1],kde_kws={"lw": 1.5})
            #else: 
            #    print ("no se puede pintar valor SYNTH")

            #plt.legend(labels=["Real","Linear", "Custom"])
            #plt.legend(labels=["Real", "Synth"])
            if x_axis!= None:
                plt.xlim(x_axis)
            #plt.xlabel('Values')

            #if not name is None:
            #    plt.savefig(name + '.svg')

            #plt.show()
            
            tipo= "hist" if hist else "kde"
            plt.savefig(f"plots_hists_debug/{self.exp_name}/{self.trial_id}/{self.it}/x_hist_feat-{name}_{tipo}_it-{self.it}.png")
            plt.close()
            
            
    def compute_histograms(self, adversarial_examples, malign_inputs, benign_inputs, l2_list):
        NBINS=100
        tf.print ("Generando histogramas: ",f"plots_hists_debug/{self.exp_name}/{self.trial_id}/{self.it}")
        for feature in range(adversarial_examples.shape[1]):
            Path(f"plots_hists_debug/{self.exp_name}/{self.trial_id}/{self.it}").mkdir(parents=True, exist_ok=True)

            # plot the histogram of the feature of adversarial examples and malign examples side by side
            
            
            

            colors = sns.color_palette("tab10")
            fig, ax = plt.subplots(3, 1, figsize=(10, 10))

            datos=np.array(adversarial_examples)[:, feature]
            if "MaxMin" in str(mu.muestras.scaler_path):
                rango=(0,1)
                rango= ( np.min([np.min(datos),0.]) , np.max([np.max(datos),1.]) )
            else:
                rango= None   
                
            if np.std(datos) != 0:
                ax[0].hist(
                    np.array(adversarial_examples)[:, feature],
                    bins=NBINS,
                    label=f"adversarial_examples (feature {feature})",
                    color=colors[0], range=rango
                )
            else:
                tf.print ("Warning: compute_histogram, datos generados stdev==0, generadora -> datos fijos a un valor")
                      
            
            ax[1].hist(malign_inputs[:, feature], bins=NBINS, label=f"malign_inputs (feature {feature})", color=colors[1], range=rango)
            ax[2].hist(benign_inputs[:, feature], bins=NBINS, label=f"benign_inputs (feature {feature})", color=colors[2], range=rango)

            fig.legend()
            fig.suptitle(f"Feature {feature} histogram")
            fig.tight_layout()
            #tf.print ("Generando histograma:",
            #       f"plots_hists_debug/{self.exp_name}/{self.trial_id}/{self.it}/hist_feat-{feature}_it-{self.it}.png")
            fig.savefig(
                f"plots_hists_debug/{self.exp_name}/{self.trial_id}/{self.it}/hist_feat-{feature}_it-{self.it}.png"
            )
            plt.close()
            
            # Plotear juntas
            #tf.print ("ploteando features juntas con histograma y kde")
            malign_col=malign_inputs[:, feature]
            #val_max=np.percentile(malign_col,100)
            #ids=malign_col<=val_max
            ids=range(malign_inputs.shape[0])
            self.plot_histograms_new(df_real=malign_col[ids],df_sm=np.array(adversarial_examples)[ids, feature],name=f"f:{feature}",N_BINS=NBINS)
            

        '''
        plt.figure(figsize=(10, 10))
        plt.hist(np.array(l2_list), bins=15, label=f"l2_list")
        plt.title(f"Histogram of l2 norm")
        plt.savefig(f"plots_hists_debug/{self.exp_name}/{self.trial_id}/{self.it}/hist_l2_norm_it-{self.it}.png")
        plt.close()
        
        '''
        x1et= adversarial_examples
        x2et= malign_inputs
        
        num_iguales=self.get_num_iguales (adversarial_examples[:200],malign_inputs[:200]) 
        
        dist_outliers,num_outliers,num_cub_ov,diffs= self.plot_cubitos (x1et,x2et,"cmp_distr_cubitos")
        
        assert malign_inputs.shape[0] == num_cub_ov
        d_med= dist_outliers/num_outliers if num_outliers >0 else 0
        n=num_cub_ov-num_outliers
        alpha=n/num_cub_ov
        alpha_u=1.+alpha
        alpha_d=1.-alpha
        ratio=diffs/num_cub_ov
        tf.print ("Distancia outliers rm-gm:(tot,num_outliers, media)",dist_outliers,num_outliers,d_med, 
                  "num_cub_ov,diffs, alpha, ",num_cub_ov,diffs,alpha, 
                  f"1+a > d/k > 1-a. {alpha_u:.3f} > {ratio:.3f} > {alpha_d:.3f}",
                  f"num_iguales: {num_iguales}")
                
    def get_num_iguales (self,sample_orig, sample_dest, tensor=True):
        num_iguales=0
        if tensor:
            sample_orig=sample_orig.numpy()
        casi_cero=0.001
        #tf.print (sample_orig.shape,sample_dest.shape)
        for i in range(sample_orig.shape[0]):
            for j in range (sample_dest.shape[0]):
                diff=sample_orig[i]-sample_dest[j]
                dist=np.sqrt(np.sum(diff*diff))
                if dist < casi_cero:
                    if dist >0. :
                        tf.print (f"son_casi_iguales. dist {dist:0.5f}:",sample_orig[i],"--",sample_dest[j])
                    num_iguales+=1
                    break
        return num_iguales
    
    def plot_cubitos (self,x1et,x2et,filename_cmp_distr_cubitos):
                  
        # CMP distribuciones M y G segun cubitos
        
        tf.print ("> plot_cubitos",filename_cmp_distr_cubitos)
        minimo=min(x1et.shape[0],x2et.shape[0])
        a1=x1et[:minimo]
        a2=x2et[:minimo]

        rango_x=mu.muestras.rango_x
        min_x=mu.muestras.min_x
        tf.print ("rango_x, min_x",rango_x,min_x)
        
        #dic= get_dict_distancia_alb (x1et,rango_x,min_x,random=False,scale=True)
        #print (len(dic.keys()))
        
        
        tf.print ("FACTOR_ESCALA",FACTOR_ESCALA)
        dist.set_ESCALA (FE=FACTOR_ESCALA,FR=0.1)
        diffs,tot_bins,coincide,vals,dic= dist.distancia_alb (a2,a1,rango_x,min_x,random=False,scale=True)
        tf.print (f"diffs:{diffs},tot_bins:{tot_bins},coincide:{coincide}, ratio={round(coincide/tot_bins,2)}")
        
        #print (dic.items())
        
        aa=dict(sorted(dic.items(), key=lambda item: item[1][0],reverse=False))
        
        l1=[]
        l2=[]
        lk=[]
        dist_diff=0
        tot_num_outliers=0
        num_cub_ov=0
        diffs=0
        for k,v in aa.items():
            if self.debug2:
                tf.print(f"{k}: {v}")
            l1.append(v[0])
            l2.append(v[1])
            if v[0]== 0 :
                dist_min=np.inf
                for k2,v2 in aa.items():
                    if v2[0] >0:
                        #tf.print ("XXXXXX:",v[2].shape,v2[2].shape)
                        d=np.linalg.norm(v[2]-v2[2])
                        if d < dist_min:
                            dist_min=d
                            
                if self.debug2:
                    tf.print ("dist de outliers:", dist_min, v[1])
                num_outliers=v[1]
                dist_diff+= dist_min*num_outliers
                tot_num_outliers+= num_outliers
            else:
                num_cub_ov+=v[0]
                diffs+=np.abs(v[0]-v[1])
        tf.print ("** Distancia de outliers **", dist_diff)
        tf.print (f"Cubitos-Valores dict ord ({filename_cmp_distr_cubitos}). epoch: {self.epoch} diffs: {diffs} cubitos_orig:{num_cub_ov}")
        #if filename_cmp_distr_cubitos != "referencia":
        #    tf.print (aa)
                                         
                
        #plt.title('Comparison of synthetic and real data distributions. The 4-dimensional vector is flattened into and sorted by frequency in ascending order on the x-axis.')
        #plt.xlabel(f'Epoch:'{self.epoch})
        plt.figure(figsize=(10, 6))
        plt.ylabel('#samples')
            
        plt.plot(l1)
        plt.plot(l2)
        #plt.show()
        plt.savefig(f"plots_hists_debug/{self.exp_name}/{self.trial_id}/{self.it}/{filename_cmp_distr_cubitos}_it-{self.it}.png")
        plt.close()
        return dist_diff, tot_num_outliers, num_cub_ov,diffs
        

    def plot_2d_data_points(self, adversarial_examples, malign_inputs, benign_inputs):
        Path(f"plots_2d_debug/{self.exp_name}/{self.trial_id}/{self.it}").mkdir(parents=True, exist_ok=True)

        # Plot the data points in a two dimensional scatter plot
        colors = sns.color_palette("tab10")

        fig, ax = plt.subplots(3, 1, figsize=(10, 10))

        ax[0].scatter(
            adversarial_examples[:, 0], adversarial_examples[:, 1], label="adversarial_examples", color=colors[0]
        )
        ax[1].scatter(malign_inputs[:, 0], malign_inputs[:, 1], label="malign_inputs", color=colors[1])
        ax[2].scatter(benign_inputs[:, 0], benign_inputs[:, 1], label="benign_inputs", color=colors[2])

        fig.legend()
        fig.suptitle("2D scatter plot of data points (features 0 and 1)")
        fig.tight_layout()
        fig.savefig(
            f"plots_2d_debug/{self.exp_name}/{self.trial_id}/{self.it}/2d_data_points_it-{self.it}_feats-01.png"
        )
        plt.close()

        fig, ax = plt.subplots(3, 1, figsize=(10, 10))

        ax[0].scatter(
            adversarial_examples[:, 2], adversarial_examples[:, 3], label="adversarial_examples", color=colors[0]
        )
        ax[1].scatter(malign_inputs[:, 2], malign_inputs[:, 3], label="malign_inputs", color=colors[1])
        ax[2].scatter(benign_inputs[:, 2], benign_inputs[:, 3], label="benign_inputs", color=colors[2])

        fig.legend()
        fig.suptitle("2D scatter plot of data points (features 2 and 3)")
        fig.tight_layout()
        fig.savefig(
            f"plots_2d_debug/{self.exp_name}/{self.trial_id}/{self.it}/2d_data_points_it-{self.it}_feats-23.png"
        )
        plt.close()
        
        
    # ------------------------------
    
    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        #alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        #norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def gen_test_cm(self, msg, random_latent_vectors, malign_inputs, msg2="", training=True, batch_size=512,p_stats=False):
        tf.print("gen_test_cm, con training=", training, msg)
        
        # Discriminador MALIGN (REAL_LABEL) vs GENERADOS (FAKE_LABEL). 
        # en Generador GENERADOS -> MALIGNOS (REAL_LABEL)
        if self.solo_noise:
            adversarial_examples = self.generator(random_latent_vectors, training=training)
        else:  
            adversarial_examples = self.generator((random_latent_vectors, malign_inputs), training=training)

        adv_logits = self.discriminator(adversarial_examples, training=training)

        # No se utilizan pen WGAN ero se devuelven en return
        adversarial_labels_cat = [REAL_LABEL] * len(adversarial_examples)
        
        if not self.WGAN:
            adv_logits_cat = np.where(adv_logits > 0.5, 1, 0)
            #adversarial_labels_cat = [REAL_LABEL] * len(adversarial_examples)

            m = confusion_matrix(adversarial_labels_cat, adv_logits_cat, labels=[0, 1])
            tf.print("Confusion matrix del Discriminator training=",training,":\n FAKE(0) vs REAL(1). GENERADOS con REAL_LABEL \n", m)

            # convert shape from (batch_size, 1) to (batch_size,)
            adv_logits = tf.reshape(adv_logits, [-1])

            tf.print("adv logits shape", adv_logits.shape)
            tf.print("adversarial_labels_cat shape", len(adversarial_labels_cat))

            g_loss = self.g_loss_fn(adversarial_labels_cat, adv_logits)
        else: 
            # WAsserstein el critico genera valores float, no hay referencia de etiquetas
            g_loss = self.g_loss_fn(adv_logits)
            
            
            
            
        tf.print ("g_loss discriminador:",g_loss)

        # calculate the adversarial loss
        if self.alfa > 0:
            # Distilled BlackBox: BENING vs MALIGN
            # Generator GENERATED (from MALIGN)  -> BENIGN
            
            # Get the latent and malign inputs
            # OJO 
            #random_latent_vectors = tf.random.uniform(shape=(batch_size, self.latent_dim), minval=-1, maxval=1)
            #malign_inputs = mu.muestras.sample_examples(batch_size, class_label=MALIGN_LABEL)
            #adversarial_examples = self.generator((random_latent_vectors, malign_inputs), training=training)

            # Como sampleamos dos veces los ejemplos de entrada que usamos para la generadora, la contribución al gradiente luego va a ser la mitad y la mitad. La media de ambas derivadas.
            # Hay que recordar que al calcular el gradiente se obtiene la derivada por ejemplo. Como tenemos muchos ejemplos. Se hace la media de ellos.

            # Unscaled data
            '''
            var = self.scaler.var_[self.feats]
            std = np.sqrt(var)
            mean = self.scaler.mean_[self.feats]

            # convert to tensor
            var = tf.convert_to_tensor(var, dtype=tf.float32)
            mean = tf.convert_to_tensor(mean, dtype=tf.float32)
            tf.print (var.shape,mean.shape,adversarial_examples.shape)
            
            non_scaled_adversarial_examples = tf.add(
                tf.cast(tf.multiply(adversarial_examples, std), tf.float32), tf.cast(mean, tf.float32)
            )
            
            '''
            
            non_scaled_adversarial_examples= mu.muestras.inverse_normalise (adversarial_examples,tensor=True)

            bb_logits = mc.predict_training(
                non_scaled_adversarial_examples, self.distilled_bb_model_path
            )  # distilled BB model
            
            bb_logits_cat=np.argmax(bb_logits, axis=1)
            tf.print ("bb_logits.shape",bb_logits.shape,"bb_logits_cat",bb_logits_cat.shape)
            
            # calculate the cross entropy loss
            cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

            adversarial_labels_cat = [BENIGN_LABEL] * len(adversarial_examples)
            adversarial_labels_oh = tf.one_hot(adversarial_labels_cat, NUM_CLASSES)

            adv_loss = cross_entropy(adversarial_labels_oh, bb_logits)
            tf.print ("adv_loss:",adv_loss)
            
            m = confusion_matrix(adversarial_labels_cat, bb_logits_cat, labels=[0, 1])
            tf.print("Confusion matrix del Distilled (Black Box) \n BENIGN(0) vs MALIGN(1) Generados con etiqueta BENIGN:\n", m)
            # the distilled BB model is not part of the class, it has its own loss function but is the same as d_loss_fn, the cross-entropy
        else:
            adv_loss = 0

        if self.debug2 or self.beta > 0:
            # Distancias de MALIGN y GENERATED
            
            # Get the latent and malign inputs
            # OJO lo he quitado para que todos usen la misma entrada
            #random_latent_vectors = tf.random.uniform(shape=(batch_size, self.latent_dim), minval=-1, maxval=1)
            #malign_inputs = mu.muestras.sample_examples(batch_size, class_label=MALIGN_LABEL)
            #adversarial_examples = self.generator((random_latent_vectors, malign_inputs), training=training)

            # compute the l2 norm between the original and the adversarial examples
            perturbation = adversarial_examples - malign_inputs
            #tf.print ("adversarial_examples\n",adversarial_examples.shape,adversarial_examples)
            #tf.print ("malign_inputs\n",malign_inputs.shape,malign_inputs)
            #tf.print ("pertur:\n",perturbation)
            tf.print ("\ntipo de distancia aplicada en beta:",self.tipo_distancia_train)
            if self.tipo_distancia_train == "cuadrados":
                final=tf.reduce_sum(tf.square(perturbation), axis=1)
            else:
                final=tf.sqrt(tf.reduce_sum(tf.square(perturbation), axis=1))
            pert_loss = tf.reduce_mean(final)
            pert_loss1 = tf.reduce_sum(final)
            #pert_loss2= tf.reduce_mean(tf.norm (perturbation,axis=1))
            tf.print("pert_loss (antes de mean) final.shape", final.shape)  
            tf.print ("euclidea desglosada,reduce_mean, reduce_sum:", pert_loss,pert_loss1)
            if p_stats:
                l_val= final.numpy()
                tf.print (f"pseudo-get_1 lista_pp:beta_eucl min:{np.min(l_val)}, max:{np.max(l_val)}, media:{np.mean(l_val)}, \n lista_pp:beta_eucl p90:{np.percentile(l_val,90)}, p75:{np.percentile(l_val,75)}, p25:{np.percentile(l_val,25)}, p10:{np.percentile(l_val,10)}")
            tf.print(f"pert_loss suma:{pert_loss1} media:{pert_loss}")
    
        else:
            pert_loss = 0

        

        if False and self.debug2:
            if self.it % 1 == 0:
                l2_list = []

                for i in range(100):
                    diff = adversarial_examples[i] - malign_inputs[i]
                    l2 = tf.sqrt(tf.reduce_sum(tf.square(diff)))
                    tf.print("l2 shape", l2.shape)
                    l2_list.append(l2)

                    tf.print(
                        "malign: ", malign_inputs[i], "adversarial: ", adversarial_examples[i], "l2", l2,
                    )

                mean_l2 = tf.reduce_mean(l2_list)

                tf.print("mean_l2", mean_l2)

                benign_inputs = mu.muestras.sample_examples(batch_size, class_label=BENIGN_LABEL)
                malign_inputs2 = mu.muestras.sample_examples(batch_size, class_label=MALIGN_LABEL)
                self.compute_histograms(adversarial_examples, malign_inputs2, benign_inputs, l2_list)

                loss_dist = 0.0
                num_samples_dist = batch_size

                samples_gen = adversarial_examples[:num_samples_dist]
                samples_real = malign_inputs[:num_samples_dist]
                l1, l2, l3, l3_mean = dist.get_1(samples_real, samples_gen, inf=0.0, sup=0.95)
                distancia_alb = l3_mean
                """
                distances = map(mc.get_closest_dist, gen_adversarials_dist,
                                malign_inputs[:num_samples_dist])
                tf.print ("distances:",distances,type(distances))
                #loss_dist = np.sum(list(distances)) / num_samples_dist
                loss_dist = tf.math.add_n(list(distances)) / num_samples_dist
                """
                tf.print("distancia_alb:", distancia_alb, type(loss_dist))

                self.plot_2d_data_points(
                    adversarial_examples, malign_inputs, benign_inputs,
                )

        if self.debug:
            tf.print("g_loss" + msg2, g_loss, type(g_loss), g_loss.shape)

        #print("GPU memory usage (after)")
        #mc.get_gpu_memory_usage()

        return g_loss, adv_loss, pert_loss, adversarial_examples, adversarial_labels_cat

    def gen_test_pred_cm(self, adversarial_examples, adversarial_labels_cat):
        '''
        var = self.scaler.var_[self.feats]
        std = np.sqrt(var)
        mean = self.scaler.mean_[self.feats]

        # convert to tensor
        var = tf.convert_to_tensor(var, dtype=tf.float32)
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)

        non_scaled_adversarial_examples = tf.add(
            tf.cast(tf.multiply(adversarial_examples, std), tf.float32), tf.cast(mean, tf.float32)
        )
        '''
        non_scaled_adversarial_examples= mu.muestras.inverse_normalise (adversarial_examples,tensor=True)

        bb_adv_logits = mc.predict(non_scaled_adversarial_examples, self.black_box_model, self.bb_model_path)
        adversarial_labels_cat = [BENIGN_LABEL] * len(adversarial_examples)
        m = confusion_matrix(adversarial_labels_cat, bb_adv_logits, labels=[0, 1])
        tf.print("Confusion matrix de Adversarial Generados contra Black Box .predict(): (BENIGN(0) vs MALIGN(1))\n GENERATED label BENING", m)

    
    def disc_test_cm_wgan (self, msg, fake_images, real_images, msg2="", training=True):
        # Generate fake images from the latent vector
        #fake_images = self.generator(random_latent_vectors, training=True)
        # Get the logits for the fake images
        fake_logits = self.discriminator(fake_images, training=True)
        # Get the logits for the real imagesssss
        real_logits = self.discriminator(real_images, training=True)

        # Calculate the discriminator loss using the fake and real image logits
        d_cost = self.d_loss_fn(real_logits=real_logits, fake_logits=fake_logits)
        # Calculate the gradient penalty
        batch_size=real_images.shape[0]
        gp = self.gradient_penalty(batch_size, real_images, fake_images)
        # Add the gradient penalty to the original discriminator loss
        grad_penalty= gp * self.gp_weight
        d_loss = d_cost + grad_penalty
        
        #if self.debug:
        tf.print("d_loss WGAN" + msg2, d_loss, type(d_loss), d_loss.shape)

        #print("GPU memory usage (after the execution)")
        #mc.get_gpu_memory_usage()

        return d_loss, grad_penalty
    
    
    def disc_test_cm(self, msg, all_batch_examples, all_batch_labels_cat, msg2="", training=True):
        tf.print("disc_test_cm", msg)

        tf.print("all_batch_examples.shape", all_batch_examples.shape)

        # Get the logits for the adversarial inputs
        discriminator_logits = self.discriminator(all_batch_examples, training=training)

        if self.debug:
            tf.print("discriminator_logits:", discriminator_logits)

        #print("GPU memory usage (after getting disc logits)")
        #mc.get_gpu_memory_usage()
        
        

        disc_logits_cat = np.where(discriminator_logits > 0.5, 1, 0)

        if self.debug:
            tf.print("disc_logits_cat.shape", disc_logits_cat.shape)

        if self.debug:
            tf.print("disc_logits_cat", disc_logits_cat)

        # convert shape from (batch_size, 1) to (batch_size,)
        discriminator_logits = tf.reshape(discriminator_logits, [-1])

        tf.print("discriminator_logits shape", discriminator_logits.shape)
        tf.print("all_batch_labels_cat shape", all_batch_labels_cat.shape)

        m = confusion_matrix(all_batch_labels_cat, disc_logits_cat, labels=[0, 1])
        tf.print("Confusion matrix DISCRIMINADOR training=",training,":\n FAKE(0) vs REAL (1) ALL_batch_examples\n", m)

        d_loss = self.d_loss_fn(all_batch_labels_cat, discriminator_logits)

        #if self.debug:
        tf.print("d_loss" + msg2, d_loss, type(d_loss), d_loss.shape)

        #print("GPU memory usage (after the execution)")
        #mc.get_gpu_memory_usage()

        return d_loss

    def get_samples_malign_BB(self, num_muestras):
        batch_size = num_muestras
        lista = []
        tot = 0
        while tot < num_muestras:
            malign_inputs = mu.muestras.sample_examples(batch_size // 3, class_label=MALIGN_LABEL)
            benign_inputs = mu.muestras.sample_examples(batch_size // 3, class_label=BENIGN_LABEL)
            # random_vectors = np.random.normal(0, 15,size=(batch_size // 10, self.feature_dims) )
            random_vectors = np.random.uniform(
                self.min_rnd, self.max_rnd, size=(batch_size // self.noise_num, self.feature_dims)
            )
            all_batch_examples = np.concatenate([benign_inputs, random_vectors, malign_inputs], axis=0)
            '''
            var = self.scaler.var_[self.feats]
            std = np.sqrt(var)
            mean = self.scaler.mean_[self.feats]

            # convert to tensor
            var = tf.convert_to_tensor(var, dtype=tf.float32)
            mean = tf.convert_to_tensor(mean, dtype=tf.float32)

            non_scaled_all_batch_examples = tf.add(
                tf.cast(tf.multiply(all_batch_examples, std), tf.float32), tf.cast(mean, tf.float32)
            )
            '''
            non_scaled_all_batch_examples= mu.muestras.inverse_normalise (all_batch_examples,tensor=True)

            # Label malign, benign and noise examples with the black box model
            all_batch_labels_cat = mc.predict(non_scaled_all_batch_examples, self.black_box_model, self.bb_model_path)
            malign_BB = non_scaled_all_batch_examples[all_batch_labels_cat == MALIGN_LABEL]
            lista.append(malign_BB)
            tot += malign_BB.shape[0]

        lista = np.concatenate(lista, axis=0)
        return lista[:num_muestras]

    # def get_gpu_memory_usage(self):
    #     """Creates a thread that prints the GPU memory usage every second."""

    #     def print_gpu_memory_usage():
    #         while True:
    #             print("GPU memory usage:", tf.memory.experimental.get_memory_info(device="gpu:0")["current"])
    #             time.sleep(1)

    #     t = threading.Thread(target=print_gpu_memory_usage)
    #     t.start()

    def train_batch(self, batch_size, **kwargs):
        """
        Main training function for a batch size which is sampled from the dataset.
        For each batch, we are going to perform the following steps:
        # 1. Sample a minibatch of Cripto Attacks M
        # 2. Generate adversarials with the generator M'
        # 3. Train the generator and get the generator loss
        # 4. Sample a minibatch of Benign examples B
        # 5. Label B and M' with the back box model
        # 6. Train the discriminator and get the discriminator loss
        # 7. Return the generator and discriminator losses as a loss dictionary
        """

        # --------------------------
        # Train the discriminator first
        # --------------------------

        for i in range(self.d_steps):  # Extra steps
            tf.print("TRAIN DISC step:", i, "de ", self.d_steps)

            # Prepare examples SOLO MALIGN (FAKE vs REAL) . NO benign and noise
            malign_inputs = mu.muestras.sample_examples(batch_size // 2, class_label=MALIGN_LABEL)
            benign_inputs = mu.muestras.sample_examples(batch_size // 2, class_label=BENIGN_LABEL)
            

            # random_vectors = np.random.normal(0, 15, size=(batch_size // 10, self.feature_dims))
            random_vectors = np.random.uniform(
                self.min_rnd, self.max_rnd, size=(batch_size // self.noise_num, self.feature_dims)
            )

            assert len(random_vectors) == 0

            tf.print("malign_inputs", malign_inputs.shape)
            tf.print("benign_inputs", benign_inputs.shape)
            tf.print("random_vectors", random_vectors.shape)

            all_batch_examples = np.concatenate([benign_inputs, random_vectors, malign_inputs], axis=0)

            tf.print("all_batch_examples", all_batch_examples.shape)
            
            tf.print(
                "BENIGN", benign_inputs.shape[0], " RANDOM:", random_vectors.shape[0], "MALIGN", malign_inputs.shape[0]
            )
            
            ## ----
            
            # Unscale data
            
            non_scaled_all_batch_examples= mu.muestras.inverse_normalise (all_batch_examples,tensor=True)
            
            '''
            var = self.scaler.var_[self.feats]
            std = np.sqrt(var)
            mean = self.scaler.mean_[self.feats]
            # convert to tensor
            var = tf.convert_to_tensor(var, dtype=tf.float32)
            mean = tf.convert_to_tensor(mean, dtype=tf.float32)

            non_scaled_all_batch_examples = tf.add(
                tf.cast(tf.multiply(all_batch_examples, std), tf.float32), tf.cast(mean, tf.float32)
            )
            
            tf.print ("var:",var.shape,"mean",mean.shape,"examples",all_batch_examples.shape)
            '''              

            # Label malign, benign and noise examples with the black box model
            all_batch_labels_cat = mc.predict(non_scaled_all_batch_examples, self.black_box_model, self.bb_model_path,)

            # check de BB
            
            labels_ok= np.concatenate(
                [np.array([BENIGN_LABEL]*benign_inputs.shape[0]),np.array ([MALIGN_LABEL]*malign_inputs.shape[0])], axis=0)
            m = confusion_matrix(labels_ok, all_batch_labels_cat, labels=[0, 1])
            tf.print ("Confusion de valores reales b+M sobre BB\n",m)
            
            #
            ### PREPARACION MUESTRAS SOLO Malign y Generados
            ## ---
            
            benign_inputs = np.empty(shape=(0, self.feature_dims))
            all_batch_examples = np.concatenate([benign_inputs, random_vectors, malign_inputs], axis=0)

            tf.print("all_batch_examples", all_batch_examples.shape)
            
            tf.print(
                "BENIGN", benign_inputs.shape[0], " RANDOM:", random_vectors.shape[0], "MALIGN", malign_inputs.shape[0]
            )
            

            assert len(benign_inputs) == 0

            # # Label malign, benign and noise examples with the black box model
            # all_batch_labels_cat = mc.predict(non_scaled_all_batch_examples, self.black_box_model, self.bb_model_path,)

            # True labels (the attacker does not know them so this code is disabled)
            # all_batch_labels_cat = (
            #     [MALIGN_LABEL] * len(malign_inputs)
            #     + [RUIDO_LABEL] * len(random_vectors)
            #     + [BENIGN_LABEL] * len(benign_inputs)
            # )

            #tf.print("BENIGN", benign_inputs.shape[0], "MALIGN", malign_inputs.shape[0], "NOISE", random_vectors.shape[0])

            # Generate adversarials from the latent vector and criptoattack (malign)
            malign_inputs = mu.muestras.sample_examples(batch_size // 2, class_label=MALIGN_LABEL)  # [:, f_UPC]
            # random_latent_vectors = tf.random.normal(shape=(malign_inputs.shape[0], self.latent_dim))
            random_latent_vectors = tf.random.uniform(
                shape=(malign_inputs.shape[0], self.latent_dim), minval=self.min_rnd, maxval=self.max_rnd
            )
            if self.solo_noise:
                adversarials_examples = self.generator(random_latent_vectors, training=True)
            else:
                adversarials_examples = self.generator((random_latent_vectors, malign_inputs), training=True)
            # adversarials_labels_cat = [MALIGN_LABEL] * len(adversarials_examples)

            tf.print("MALIGN_ADV_generados", adversarials_examples.shape[0])

            # # Labels are obtained from the BlackBox model (we do not know the labels)
            # adversarials_labels_cat = mc.predict(
            #     non_scaled_adversarials_examples, self.black_box_model, self.bb_model_path,
            # ).astype(int)

           
            all_batch_labels_cat = [REAL_LABEL] * len(malign_inputs) # len(all_batch_examples) No hay benignos ni aleatorios
            adversarials_labels_cat = [FAKE_LABEL] * len(adversarials_examples)

            with tf.GradientTape() as tape:
                
                if self.WGAN:
                    d_loss, grad_penalty = self.disc_test_cm_wgan (
                        "DISC WGAN: antes de entrenamiento discriminadora",
                        adversarials_examples,
                        malign_inputs,
                        msg2="1",
                        training=True,
                    )
                    tf.print (f"d_loss:{d_loss} grad_penalty:{grad_penalty}")
                    
                else:
                    # Add ADVERSARIALS examples to malign,benign,noise
                    all_batch_examples = np.concatenate([all_batch_examples, adversarials_examples], axis=0)
                    all_batch_labels_cat = np.concatenate([all_batch_labels_cat, adversarials_labels_cat], axis=0).astype(int)
                    if self.debug:
                        tf.print("histogram:", np.bincount(all_batch_labels_cat))

                    if self.debug:
                        tf.print(all_batch_labels_cat)

                    # tf.print (" all_batch_labels_cat:", all_batch_labels_cat)

                    # Calculate model loss
                    d_loss = self.disc_test_cm(
                        "DISC: antes de entrenamiento discriminadora",
                        all_batch_examples,
                        all_batch_labels_cat,
                        msg2="1",
                        training=True,
                    )

                # Calculate l2 reg loss
                loss_reg_l2_d = mc.add_model_regularizer_loss(self.discriminator)

                d_loss_tot = self.RATIO_LOSS_D * d_loss + self.RATIO_REG_D * loss_reg_l2_d
                tf.print(
                    "d_loss_tot= RATIO_LOSS_D * d_loss + RATIO_REG_D * loss_reg_l2_d\n",
                    d_loss_tot,
                    "=",
                    self.RATIO_LOSS_D,
                    d_loss,
                    self.RATIO_REG_D,
                    loss_reg_l2_d,
                )

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss_tot, self.discriminator.trainable_variables)

            if self.debug:
                tf.print("d_gradient", d_gradient)

            assert sum_gradient(d_gradient) != 0, "Gradient disc = 0"

            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

            # chequear
            if not self.WGAN:
                d_loss_despues = self.disc_test_cm(
                        "DISC: despues de entrenamiento discriminadora",
                        all_batch_examples,
                        all_batch_labels_cat,
                        msg2="2",
                        training=True,
                    )
                tf.print ("d_loss despues de train discriminadora:",d_loss_despues)

                if self.debug:
                    d_loss_2 = self.disc_test_cm(
                        "DISC: despues de entrenamiento discriminadora (training=False)",
                        all_batch_examples,
                        all_batch_labels_cat,
                        msg2="2",
                        training=False,
                    )

                if self.debug:
                    d_loss_2 = self.disc_test_cm(
                        "DISC: despues de entrenamiento discriminadora (training=True)",
                        all_batch_examples,
                        all_batch_labels_cat,
                        msg2="3",
                        training=True,
                    )
                
        # --------------------------
        # Train the generator
        # --------------------------
        p_stats=True
        with tf.GradientTape(persistent=False) as tape:
            tf.print("TRAIN GEN")

            # Get the latent and malign inputs
            # random_latent_vectors = tf.random.normal(shape=(batch_size,self.latent_dim))
            random_latent_vectors = tf.random.uniform(shape=(batch_size, self.latent_dim), 
                                                      minval=self.min_rnd, maxval=self.max_rnd)
            malign_inputs = mu.muestras.sample_examples(batch_size, class_label=MALIGN_LABEL)
            # malign_inputs = self.get_samples_malign_BB (batch_size)

            # Calculate model loss
            g_loss, g_loss_adv, g_loss_pert, gen_adversarials, gen_adversarial_labels_cat = self.gen_test_cm(
                "check gen antes de train",
                random_latent_vectors,
                malign_inputs,
                "1",
                training=True,
                batch_size=batch_size,
                p_stats=p_stats
            )

            # Calculate the regularization loss
            """
            loss_r1 = 0.
            if self.REG_LOSS:
                lr = self.generator.losses
                loss_r1 = tf.math.reduce_sum(lr)
            """
            loss_reg_l2_g = mc.add_model_regularizer_loss(self.generator)

            # Calculate the  distance loss
            loss_dist = 0.0
            num_samples_dist = batch_size
            
            
            #assert self.RATIO_DIST_G == 0
            samples_fk = gen_adversarials[:num_samples_dist]
            samples_real = malign_inputs[:num_samples_dist]
            
            #debug
            #malign_inputs2 = mu.muestras.sample_examples(batch_size, class_label=MALIGN_LABEL)
            #samples_real2 = malign_inputs2[:num_samples_dist]
            #dist.get_1(samples_real, samples_real2, inf=0.0, sup=1, 
            #                                     stochastic=True, umbral_dist_alb=self.umbral_dist_alb,
            #                                     reverse=False,debug=True,p_stats=p_stats,msg="r2r",tensor=False, 
            #                                     tipo_distancia=self.tipo_distancia) 
            

            
            if self.debug2 or (self.RATIO_DIST_G > 0):
                
                np.set_printoptions(precision=3)
                
                '''
                
                
                tf.print ("\n############### detalle #######################\n")
                mm=malign_inputs[:10]
                fk=samples_fk[:10]
                
                dist.get_1_pseudo_eu (mm,fk, self.tipo_distancia_train,debug=True)
                
                l1_s, l2_s, l3_s, l3_mean_s = dist.get_1(mm, fk, inf=self.inf_train, sup=self.sup_train, 
                                                 stochastic=True, umbral_dist_alb=np.inf,
                                                 reverse=False, debug=True, p_stats=p_stats, 
                                                 tipo_distancia=self.tipo_distancia_train) 
                tf.print ("\n############### fin detalle #######################\n")
                
                # Lo hacemos para comparar en depuracion co la otra distancia
                tf.print ("----\nAplico a g_loss_tot dist_alb con umbral:",self.umbral_dist_alb)
                #l1, l2, l3, l3_mean = dist.get_1(samples_real, samples_fk, inf=0.0, sup=1.0, 
                #                                 stochastic=True, umbral_dist_alb=self.umbral_dist_alb,
                #                                 reverse=False,debug=True,p_stats=p_stats) 
                # stochastic=False #Greedy
                tf.print ("tipo de distancia aplicada en DIST_ALB:",self.tipo_distancia_train)
                tf.print ("SIN umbral_dist")
                l1_s, l2_s, l3_s, l3_mean_s = dist.get_1(malign_inputs, samples_fk, inf=self.inf_train, sup=self.sup_train, 
                                                 stochastic=True, umbral_dist_alb=np.inf,
                                                 reverse=False, debug=False, p_stats=p_stats, 
                                                 tipo_distancia=self.tipo_distancia_train) 
                
                l1, l2, l3, l3_mean = dist.get_1(malign_inputs, samples_fk, inf=self.inf_train, sup=self.sup_train, 
                                                 stochastic=False, umbral_dist_alb=np.inf,
                                                 reverse=False, debug=False, p_stats=p_stats, 
                                                 tipo_distancia=self.tipo_distancia_train) 
                '''
                
                tf.print ("CON umbral_dist",self.umbral_dist_alb)
                '''
                
                l1_s, l2_s, l3_s, l3_mean_s = dist.get_1(malign_inputs, samples_fk, inf=self.inf_train, sup=self.sup_train, 
                                                 stochastic=True, umbral_dist_alb=self.umbral_dist_alb,
                                                 reverse=False, debug=False, p_stats=p_stats, 
                                                 tipo_distancia=self.tipo_distancia_train) 
                
                l1, l2, l3, l3_mean = dist.get_1(malign_inputs, samples_fk, inf=self.inf_train, sup=self.sup_train, 
                                                 stochastic=False, umbral_dist_alb=self.umbral_dist_alb,
                                                 reverse=False, debug=False, p_stats=p_stats, 
                                                 tipo_distancia=self.tipo_distancia_train) 
                if self.stochastic_train :
                    l1, l2, l3, l3_mean = (l1_s, l2_s, l3_s, l3_mean_s) 
                '''
                l1, l2, l3, l3_mean = dist.get_1(malign_inputs, samples_fk, inf=self.inf_train, sup=self.sup_train, 
                                                 stochastic=self.stochastic_train, umbral_dist_alb=self.umbral_dist_alb,
                                                 reverse=False, debug=False, p_stats=p_stats, 
                                                 tipo_distancia=self.tipo_distancia_train) 
                    
                tf.print ("dist_alb reverse=False: loss, loss_norm, loss_norm_t, loss_norm_t / len_loss_norm_t ",
                          l1,l2,l3,l3_mean)
                
                #l1b, l2b, l3b, l3_meanb = dist.get_1(samples_real, samples_fk, inf=0.0, sup=0.9, 
                #                                     stochastic=True, umbral_dist_alb=self.umbral_dist_alb, 
                #                                     reverse=True, debug=True,p_stats=p_stats) 
                #tf.print ("dist_alb reverse=True:loss, loss_norm, loss_norm_t, loss_norm_t / len_loss_norm_t ",
                #          l1b,l2b,l3b,l3_meanb)
                #loss_dist = (l3_mean + l3_meanb)/2. #l3_mean
                
                loss_dist = l3_mean
                tf.print ("loss_dist:",loss_dist)
                tf.print ("---")
                
                #samples_fk = gen_adversarials[:num_samples_dist]
                #samples_real = malign_inputs[:num_samples_dist]
                #l1, l2, l3, l3_mean = dist.get_1(samples_real, samples_fk, inf=0.0, sup=1,stochastic=True)
                #loss_dist = l3_mean
                """
                distances = map(mc.get_closest_dist, gen_adversarials_dist,
                                malign_inputs[:num_samples_dist])
                tf.print ("distances:",distances,type(distances))
                #loss_dist = np.sum(list(distances)) / num_samples_dist
                loss_dist = tf.math.add_n(list(distances)) / num_samples_dist
                """
                
                
            if self.RATIO_DIST_G > 0:
                assert self.dist_alb 
                tf.print("loss_dist (dist_alb):", loss_dist, type(loss_dist))
                local_beta=0.
            else:
                local_beta=self.beta
                tf.print("loss_dist (Euclidea):", g_loss_pert, type(g_loss_pert))

            # Total Cost
            # g_loss = f_loss + self.RATIO_REG * loss_r1 + self.RATIO_DIST * loss_dist
            g_loss_tot = (
                (self.RATIO_LOSS_G * g_loss)
                + (self.alfa * g_loss_adv)
                + (local_beta * g_loss_pert)
                + (self.RATIO_DIST_G * loss_dist)
                + (self.RATIO_REG_G * loss_reg_l2_g)
            )
            tf.print(
                "g_loss_tot = \nself.RATIO_LOSS_G * g_loss + self.alfa * g_loss_adv + local_beta * g_loss_pert + self.RATIO_DIST_G * loss_dist + self.RATIO_REG_G * loss_reg_l2_g\n",
                g_loss_tot,
                "=\n",
                self.RATIO_LOSS_G,
                "*",
                g_loss,
                "+",
                self.alfa,
                "*",
                g_loss_adv,
                "+",
                local_beta,
                "*",
                g_loss_pert,
                "+",
                self.RATIO_DIST_G,
                "*",
                loss_dist,
                "+",
                self.RATIO_REG_G,
                "*",
                loss_reg_l2_g,
            )

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss_tot, self.generator.trainable_variables)
        self.print_grad(gen_gradient,"gradients")

        '''
        #Gradientes para comparar dalb+th=0 y eucl, deben dar lo mismo
        gen_gradient_dalb = tape.gradient(loss_dist, self.generator.trainable_variables)
        self.print_grad(gen_gradient_dalb,"gen_gradient_dalb")
        gen_gradient_eucl = tape.gradient(g_loss_pert, self.generator.trainable_variables)
        self.print_grad(gen_gradient_eucl,"gen_gradient_eucl")
        '''
        
        if self.RATIO_LOSS_G == 0 and self.RATIO_DIST_G == 0 and self.RATIO_REG_G == 0 and self.alfa ==0 and self.beta==0 :
            tf.print ("\n\n\nProbando SOLO Smirnov, GRADIENTE debe ser 0.0, real:", sum_gradient(gen_gradient),"\n\n\n")
        else:
            assert sum_gradient(gen_gradient) != 0, "Gradient gen = 0"

        # Update the weights of the generator using the generator optimizer
        #tf.print ("gen_grad", type(gen_gradient),type(gen_gradient)[0],gen_gradient)
        #pause()
        '''
        tf.print ("---\nCLIPPED ANTES de apply_gradients al optimizer----")
        #gen_gradient2 = [tf.clip_by_value(grad, -1000., 1000.) for grad in gen_gradient]
        tf.print ("gradient:\n",type(gen_gradient[0]),gen_gradient[0])
        #tf.print ("\ngradient2:\n",gen_gradient2[0],gen_gradient2[0])
        for var, g in zip(self.generator.trainable_variables, gen_gradient2):
            # in this loop g is the gradient of each layer
            tf.print ("------")
            tf.print(f'{var.name}, shape: {g.shape}',"gradients min,max,mean,std",np.min(g),np.max(g),np.mean(g),np.std(g))
            #tf.print("gradients..")
            #tf.print("gradients min,max,mean,std",np.min(g),np.max(g),np.mean(g),np.std(g))
        tf.print ("----")
        '''
        hay_nan=False
        for g in gen_gradient:
            if np.sum(np.isnan(g)) >0 :
                hay_nan=True
                tf.print ("hay nans. no llamo al optimizador")
                break
                
        if not hay_nan:
            self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        #self.gen_test_pred_cm(gen_adversarials, gen_adversarial_labels_cat)

        # check training generator
        if (self.RATIO_DIST_G > 0 or local_beta >0):
            tf.print ("Gradientes actualizados")
            tf.print ("uso las mismas muestras que las del entrenamiento para ver si mejora")
            l1, l2 = dist.get_1_np(samples_real, samples_fk, inf=0.0, sup=1,stochastic=True,debug=False, 
                                   tipo_distancia=self.tipo_distancia) # 0.95
            tf.print ("dist_alb en get_1_np",l1,l2)
            
            tf.print ("uso otras muestras diferentes")
            rm1 = mu.muestras.sample_examples(batch_size, class_label=MALIGN_LABEL)
            gm1 = self.sample_generator(num_samples=batch_size, training=True)
            l1, l2 = dist.get_1_np(rm1, gm1, inf=0.0, sup=1,stochastic=True,debug=False, 
                                   tipo_distancia=self.tipo_distancia) # 0.95
            tf.print ("despues de train dist_alb en get_1_np",l1,l2)
            #d1=mc.compute_metrics(gm1, rm1,"EU")

            '''
            random_latent_vectors = tf.random.uniform(shape=(batch_size, self.latent_dim), minval=-1, maxval=1)
            malign_inputs = rm1

            adversarial_examples = self.generator([malign_inputs, random_latent_vectors], training=False)
            gm2=gen_adversarials
            d2=mc.compute_metrics(gm2, rm1,"EU")
            print ("debug_distancias: ",d1,d2)
            '''
        
        
        if False and self.debug:
            self.gen_test_cm(
                "check Generator con misma input (MG dataset) despues de train",
                random_latent_vectors,
                malign_inputs,
                "2",
                training=True,
            )
            #
            # random_latent_vectors = tf.random.normal(shape=(batch_size,self.latent_dim))
            random_latent_vectors = tf.random.uniform(shape=(batch_size, self.latent_dim), 
                                                      minval=self.min_rnd, maxval=self.max_rnd)
            malign_inputs = mu.muestras.sample_examples(batch_size, class_label=MALIGN_LABEL)

            self.gen_test_cm(
                "check Generator con *distinta* input (MG dataset) despues de train",
                random_latent_vectors,
                malign_inputs,
                "3",
                training=True,
            )

        return {"d_loss": d_loss, "g_loss": g_loss}

    def print_grad(self, gen_gradient, msg):
        tf.print("\n"+msg, type(gen_gradient))
        for var, g in zip(self.generator.trainable_variables, gen_gradient):
            # in this loop g is the gradient of each layer
            tf.print ("------")
            tf.print(f'{var.name}, shape: {g.shape}',"gradients min,max,mean,std",np.min(g),np.max(g),np.mean(g),np.std(g))
            #tf.print("gradients..")
            #tf.print("gradients min,max,mean,std",np.min(g),np.max(g),np.mean(g),np.std(g))
        tf.print ("----\n")
        
    def train_batch_solo_disc(self, batch_size, **kwargs):
        """
        Main training function for a batch size which is sampled from the dataset.
        For each batch, we are going to perform the following steps:
        # 1. Sample a minibatch of Cripto Attacks M
        # 2. Generate adversarials with the generator M'
        # 3. Train the generator and get the generator loss
        # 4. Sample a minibatch of Benign examples B
        # 5. Label B and M' with the back box model
        # 6. Train the discriminator and get the discriminator loss
        # 7. Return the generator and discriminator losses as a loss dictionary
        """

        assert self.d_steps == 0

        # Train the discriminator first
        for i in range(self.d_steps):  # Extra steps
            print("train_batch_solo_disc step:", i, "de ", self.d_steps)
            # Get the latent vector and input examples
            # random_latent_vectors = tf.random.normal(shape=(int(np.floor(batch_size / 2)), self.latent_dim))

            # Prepare examples malign, benign and noise
            malign_inputs = mu.muestras.sample_examples(batch_size // 3, class_label=MALIGN_LABEL)
            # benign_inputs = mu.muestras.sample_examples(batch_size // 3, class_label=BENIGN_LABEL)
            benign_inputs = np.empty(shape=(0, self.feature_dims))

            assert len(benign_inputs) == 0

            # Generate adversarials from the latent vector and criptoattack
            # adversarials = self.generator((random_latent_vectors, malign_inputs), training=False)
            # random_vectors = np.random.normal(0, 15,size=(batch_size // 10, self.feature_dims) )
            random_vectors = np.random.uniform(
                self.min_rnd, self.max_rnd, size=(batch_size // self.noise_num, self.feature_dims)
            )

            assert len(random_vectors) == 0

            # Label adversarials and benign examples with the black box model
            all_batch_examples = np.concatenate([benign_inputs, random_vectors, malign_inputs])
            # np.random.shuffle(all_batch_examples)

            tf.print(
                "BENIGN", benign_inputs.shape[0], "MALIGN", malign_inputs.shape[0], "NOISE", random_vectors.shape[0]
            )

            # Labels are obtained from the BlackBox model (we do not know the labels)
            # all_batch_labels_cat = mc.predict(
            #     non_scaled_all_batch_examples, self.black_box_model, self.bb_model_path,
            # ).astype(int)

            all_batch_labels_cat = [REAL_LABEL] * len(malign_inputs)

            with tf.GradientTape() as tape:
                # print ("tipos:",type(all_batch_labels_cat),type(all_batch_labels_cat[2]),all_batch_labels_cat)
                if self.debug:
                    tf.print("histogram:", np.bincount(all_batch_labels_cat))
                # tf.print (" all_batch_labels_cat:", all_batch_labels_cat)

                if self.debug:
                    tf.print(
                        "BENIGN",
                        benign_inputs.shape[0],
                        " RANDOM:",
                        random_vectors.shape[0],
                        "MALIGN",
                        malign_inputs.shape[0],
                    )

                # Calculate model loss
                d_loss = self.disc_test_cm(
                    "DISC: antes de entrenamiento discriminadora",
                    all_batch_examples,
                    all_batch_labels_cat,
                    msg2="1",
                    training=True,
                )

                # Calculate l2 reg loss
                loss_reg_l2_d = mc.add_model_regularizer_loss(self.discriminator)

                d_loss_tot = self.RATIO_LOSS_D * d_loss + self.RATIO_REG_D * loss_reg_l2_d
                print(
                    "d_loss_tot= RATIO_LOSS_D * d_loss + RATIO_REG_D * loss_reg_l2_d\n",
                    d_loss_tot,
                    "=",
                    self.RATIO_LOSS_D,
                    d_loss,
                    self.RATIO_REG_D,
                    loss_reg_l2_d,
                )

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss_tot, self.discriminator.trainable_variables)

            if self.debug:
                tf.print("d_gradient", d_gradient)

            assert sum_gradient(d_gradient) != 0, "Gradient disc = 0"

            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

            # chequear
            if self.debug:
                d_loss_2 = self.disc_test_cm(
                    "despues de entrenamiento discriminadora",
                    all_batch_examples,
                    all_batch_labels_cat,
                    msg2="2",
                    training=False,
                )

        return {"d_loss": d_loss, "g_loss": 0.0}
    
    # -------------------------------
    # MALGAN
    # -------------------------------
    
    
    def disc_test_cm_malgan (self, msg, all_batch_examples, all_batch_labels_cat, all_batch_labels_oh, msg2="", training=True):
        tf.print ("------")
        tf.print("disc_test_cm_malgan", msg)
        #tf.print("GPU memory usage (before executing anything)")
        #mc.get_gpu_memory_usage()

        tf.print("all_batch_examples.shape", all_batch_examples.shape)

        # Unscale data
        '''
        var = self.scaler.var_[self.feats]
        std = np.sqrt(var)
        mean = self.scaler.mean_[self.feats]

        # convert to tensor
        var = tf.convert_to_tensor(var, dtype=tf.float32)
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)

        non_scaled_all_batch_examples = tf.add(
            tf.cast(tf.multiply(all_batch_examples, std), tf.float32), tf.cast(mean, tf.float32)
        )
        '''
        non_scaled_all_batch_examples= mu.muestras.inverse_normalise (all_batch_examples,tensor=True)

        print("non_scaled_all_batch_examples", non_scaled_all_batch_examples.shape)

        # Get the logits for the adversarial inputs
        discriminator_logits = self.discriminator(non_scaled_all_batch_examples, training=training)
        if self.debug:
            tf.print("discriminator_logits:", discriminator_logits)

        #tf.print("GPU memory usage (after getting disc logits)")
        #mc.get_gpu_memory_usage()

        disc_logits_cat = np.argmax(discriminator_logits, axis=1)

        if self.debug:
            tf.print("disc_logits_cat.shape", disc_logits_cat.shape)
            tf.print("disc_logits_cat", disc_logits_cat)

        tf.print("discriminator_logits shape", discriminator_logits.shape)
        tf.print("all_batch_labels_oh shape", all_batch_labels_oh.shape)

        m = confusion_matrix(all_batch_labels_cat, disc_logits_cat, labels=[0, 1])
        tf.print("Confusion matrix disc",msg2,":\n", m)
        d_loss = self.d_loss_fn(all_batch_labels_oh, discriminator_logits)

        #if self.debugALB:
        tf.print("d_loss: " + msg2, d_loss, type(d_loss), d_loss.shape)
        tf.print (" ")
        
        #tf.print("GPU memory usage (after the execution)")
        #mc.get_gpu_memory_usage()

        return d_loss
    
    
    def gen_test_cm_malgan (self, msg, random_latent_vectors, malign_inputs, msg2="", training=True, batch_size=512):
        training=True
        tf.print("gen_test_cm_malgan Chequeo contra DISCRIMINADOR", msg, "training:", training )
        #tf.print("GPU memory usage (before)")
        #mc.get_gpu_memory_usage()
        
        adversarial_examples = self.generator(random_latent_vectors, training=training)
        
        # Unscaled data
        '''
        var = self.scaler.var_[self.feats]
        std = np.sqrt(var)
        mean = self.scaler.mean_[self.feats]

        # convert to tensor
        var = tf.convert_to_tensor(var, dtype=tf.float32)
        mean = tf.convert_to_tensor(mean, dtype=tf.float32)

        non_scaled_adversarial_examples = tf.add(
            tf.cast(tf.multiply(adversarial_examples, std), tf.float32), tf.cast(mean, tf.float32)
        )
        '''
        non_scaled_adversarial_examples= mu.muestras.inverse_normalise (adversarial_examples,tensor=True)

        adv_logits = self.discriminator(non_scaled_adversarial_examples, training=False)
        adv_logits_cat = np.argmax(adv_logits, axis=1)

        #adversarial_labels_cat = [BENIGN_LABEL] * len(non_scaled_adversarial_examples)
        adversarial_labels_cat = [BENIGN_LABEL] * len(adversarial_examples)
        adversarial_labels_oh = tf.one_hot(adversarial_labels_cat, NUM_CLASSES)

        m = confusion_matrix(adversarial_labels_cat, adv_logits_cat, labels=[0, 1])
        tf.print("Confusion matrix discriminator en train GEN:\n", m)
        
        g_loss = self.g_loss_fn(adversarial_labels_oh, adv_logits)
        if self.debugALB:
            tf.print("gen_test_cm g_loss" + msg2, g_loss, type(g_loss), g_loss.shape)
        
        #Chequeo contra BB
        if True or self.debug:
            tf.print ("Chequeo contra Black Box")
            adv_logits_bb_cat = mc.predict(
                    non_scaled_adversarial_examples, self.black_box_model, self.bb_model_path,in_tape_gradient=True
                ).astype(int)

            m = confusion_matrix(adversarial_labels_cat, adv_logits_bb_cat, labels=[0, 1])
            tf.print ("Confusion matrix BB en train GEN (cuanto mas benignos mejor)\n",m)
        
        

        #benign_inputs = mu.muestras.sample_examples(batch_size, class_label=BENIGN_LABEL)
        #l2_list = []
        #self.compute_histograms(adversarial_examples, malign_inputs, benign_inputs, l2_list)

        

        if False and self.debug2:
            if self.it % 1 == 0:
                l2_list = []

                '''
                for i in range(100):
                    diff = adversarial_examples[i] - malign_inputs[i]
                    l2 = tf.sqrt(tf.reduce_sum(tf.square(diff)))
                    tf.print("l2 shape", l2.shape)
                    l2_list.append(l2)

                    tf.print(
                        "malign: ", malign_inputs[i], "adversarial: ", adversarial_examples[i], "l2", l2,
                    )

                mean_l2 = tf.reduce_mean(l2_list)

                tf.print("mean_l2", mean_l2)
                '''
                

                benign_inputs = mu.muestras.sample_examples(batch_size, class_label=BENIGN_LABEL)
                self.compute_histograms(adversarial_examples, malign_inputs, benign_inputs, l2_list)

                loss_dist = 0.0
                num_samples_dist = batch_size

                samples_gen = adversarial_examples[:num_samples_dist]
                samples_real = malign_inputs[:num_samples_dist]
                l1, l2, l3, l3_mean = dist.get_1(samples_real, samples_gen, inf=0.0, sup=0.95)
                distancia_alb = l3_mean
                """
                distances = map(mc.get_closest_dist, gen_adversarials_dist,
                                malign_inputs[:num_samples_dist])
                tf.print ("distances:",distances,type(distances))
                #loss_dist = np.sum(list(distances)) / num_samples_dist
                loss_dist = tf.math.add_n(list(distances)) / num_samples_dist
                """
                tf.print("distancia_alb:", distancia_alb, type(loss_dist))

                self.plot_2d_data_points(
                    adversarial_examples, malign_inputs, benign_inputs,
                )

        

        #tf.print("GPU memory usage (after)")
        #mc.get_gpu_memory_usage()

        return g_loss, adversarial_examples, adversarial_labels_cat
    
    def train_batch_MALGAN (self, batch_size, **kwargs):
        """
        Main training function for a batch size which is sampled from the dataset.
        For each batch, we are going to perform the following steps:
        # 1. Sample a minibatch of Cripto Attacks M
        # 2. Generate adversarials with the generator M'
        # 3. Train the generator and get the generator loss
        # 4. Sample a minibatch of Benign examples B
        # 5. Label B and M' with the back box model
        # 6. Train the discriminator and get the discriminator loss
        # 7. Return the generator and discriminator losses as a loss dictionary
        """
        tf.print ("train_batch MALGAN batch_size:",batch_size,self.k)
        
        # --------------------------
        # Train the discriminator first
        # --------------------------

        for i in range(self.d_steps):  # Extra steps
            tf.print("TRAIN DISC step:", i, "de ", self.d_steps)

            # Prepare examples malign, benign and noise
            malign_inputs = mu.muestras.sample_examples(batch_size // 2, class_label=MALIGN_LABEL)
            benign_inputs = mu.muestras.sample_examples(batch_size // 2, class_label=BENIGN_LABEL)
            # random_vectors = np.random.normal(0, 15,size=(batch_size // 10, self.feature_dims) )
            
            # Ahora se cogen 0 ejemplos -> self.noise_num muy alto
            random_vectors = np.random.uniform(
                self.min_rnd, self.max_rnd, size=(batch_size // self.noise_num, self.feature_dims)
            )
            
            assert len(random_vectors) == 0

            tf.print("malign_inputs", malign_inputs.shape)
            tf.print("benign_inputs", benign_inputs.shape)
            tf.print("random_vectors", random_vectors.shape)
            
            all_batch_examples = np.concatenate([benign_inputs, random_vectors, malign_inputs], axis=0)
            
            tf.print("all_batch_examples", all_batch_examples.shape)
            
            tf.print(
                "BENIGN", benign_inputs.shape[0], " RANDOM:", random_vectors.shape[0], "MALIGN", malign_inputs.shape[0]
            )
            
            # Unscale data
            
            '''
            var = self.scaler.var_[self.feats]
            std = np.sqrt(var)
            mean = self.scaler.mean_[self.feats]
            
            tf.print ("var:",var.shape,"mean",mean.shape,"examples",all_batch_examples.shape)

            # convert to tensor
            var = tf.convert_to_tensor(var, dtype=tf.float32)
            mean = tf.convert_to_tensor(mean, dtype=tf.float32)

            non_scaled_all_batch_examples = tf.add(
                tf.cast(tf.multiply(all_batch_examples, std), tf.float32), tf.cast(mean, tf.float32)
            )
            '''
            
            non_scaled_all_batch_examples= mu.muestras.inverse_normalise (all_batch_examples,tensor=True)

            # Label malign, benign and noise examples with the black box model
            all_batch_labels_cat = mc.predict(non_scaled_all_batch_examples, self.black_box_model, self.bb_model_path,)

            tf.print(
                "BENIGN", benign_inputs.shape[0], " RANDOM:", random_vectors.shape[0], "MALIGN", malign_inputs.shape[0]
            )
            
            all_batch_examples_BM= all_batch_examples
            all_batch_labels_cat_BM =all_batch_labels_cat 
            all_batch_labels_oh_BM= np.eye(NUM_CLASSES)[all_batch_labels_cat_BM]
            
            # check de BB
            
            labels_ok= np.concatenate(
                [np.array([BENIGN_LABEL]*benign_inputs.shape[0]),np.array ([MALIGN_LABEL]*malign_inputs.shape[0])], axis=0)
            m = confusion_matrix(labels_ok, all_batch_labels_cat, labels=[0, 1])
            tf.print ("Confusion de valores reales b+M sobre BB\n",m)
            
            
            
            # Generate adversarials from the latent vector and criptoattack (malign)
            #for tr in [False,True]:
            tr=True
            malign_inputs = mu.muestras.sample_examples(batch_size // 3, class_label=MALIGN_LABEL)  # [:, f_UPC]
            # random_latent_vectors = tf.random.normal(shape=(malign_inputs.shape[0], self.latent_dim))
            random_latent_vectors = tf.random.uniform(
                shape=(malign_inputs.shape[0], self.latent_dim), minval=-1, maxval=1
            )
            #adversarials_examples = self.generator([random_latent_vectors,malign_inputs], training=True)
            adversarials_examples = self.generator(random_latent_vectors, training=True)
            if len(self.bns) >0 :
                tf.print ("generador med y var movil training:",tr)
                for u in range(len(self.bns)):
                    tf.print ("med, var, unit:",u,self.bns[u].moving_mean, self.bns[u].moving_variance)

            # adversarials_labels_cat = [MALIGN_LABEL]*len(adversarials_examples)

            # Unscale data
            '''
            var = self.scaler.var_[self.feats]
            std = np.sqrt(var)
            mean = self.scaler.mean_[self.feats]

            # convert to tensor
            var = tf.convert_to_tensor(var, dtype=tf.float32)
            mean = tf.convert_to_tensor(mean, dtype=tf.float32)

            non_scaled_adversarials_examples = tf.add(
                tf.cast(tf.multiply(adversarials_examples, std), tf.float32), tf.cast(mean, tf.float32)
            )
            '''
            
            non_scaled_adversarials_examples= mu.muestras.inverse_normalise (adversarials_examples,tensor=True)

            ## Ojo,
            ### son malignos, asi que los podriamos etiquetar directamente como malignos
            # Labels are obtained from the BlackBox model (we do not know the labels)
            #adversarials_labels_cat = mc.predict(
            #    non_scaled_adversarials_examples, self.black_box_model, self.bb_model_path,
            #).astype(int)
            self.APRENDE_DISC_GEN=True
            aprende_DISC_gen=self.APRENDE_DISC_GEN
            if aprende_DISC_gen:
                # Les ponemos etiqueta para que Discriminadora aprenda. Si no, no puede aprender luego la generadora
                adversarials_labels_cat= [MALIGN_LABEL]*adversarials_examples.shape[0]
            else:
                adversarials_labels_cat = mc.predict(
                    non_scaled_adversarials_examples, self.black_box_model, self.bb_model_path,
                )
            
            tf.print("MALIGN_ADV", adversarials_examples.shape[0], "GEn training=",tr)
            labels_ok=np.array ([MALIGN_LABEL]*adversarials_examples.shape[0])
            m = confusion_matrix(labels_ok, adversarials_labels_cat, labels=[0, 1])
            tf.print ("Confusion de valores generados GM sobre BB con etiqueta MALIGN. Evasion rate= %Benings\n",m)

            # Fuerzo a MALIGN, para que el DISCRIMINADOR aprenda a acercarlos de los MALIGNO. 
            # ASi el GENERADOR aprendera a hacerlos parecidos a los BENIGNOS para subir el evasion_rate, 
            #Y luego faltaria la funcion distancia para acercarles a lo sMALIGNOS
            #adversarial_labels_cat = [MALIGN_LABEL]*adversarials_examples.shape[0]

            with tf.GradientTape() as tape:
                # Add ADVERSARIALS examples to malign,benign,noise
                all_batch_examples = np.concatenate([all_batch_examples, adversarials_examples], axis=0)
                all_batch_labels_cat = np.concatenate([all_batch_labels_cat, adversarials_labels_cat], axis=0).astype(int)
                all_batch_labels_oh = np.eye(NUM_CLASSES)[all_batch_labels_cat] #tf.one_hot(all_batch_labels_cat, NUM_CLASSES)
                if self.debug:
                    tf.print("histogram:", np.bincount(all_batch_labels_cat))

                if self.debug:
                    tf.print(all_batch_labels_cat)

                # tf.print (" all_batch_labels_cat:", all_batch_labels_cat)
                
                
                # tf.print (" all_batch_labels oh:", all_batch_labels_oh)

                '''
                # Test Adversarials 
                adversarial_label_cat=np.array ([MALIGN_LABEL]*adversarials_examples.shape[0])
                adversarial_label_oh = np.eye(NUM_CLASSES)[adversarial_label_cat]
                d_loss_x = self.disc_test_cm(
                    "DISC: (tr=True) antes de entrenamiento discriminadora, solo adversarials (ver proporcion comparado con BB)",
                    adversarials_examples,
                    adversarial_label_cat,
                    adversarial_label_oh,
                    msg2="training=True",
                    training=True,
                )
                d_loss_x = self.disc_test_cm(
                    "DISC: (tr=False) antes de entrenamiento discriminadora, solo adversarials (ver proporcion comparado con BB)",
                    adversarials_examples,
                    adversarial_label_cat,
                    adversarial_label_oh,
                    msg2="training=False",
                    training=False,
                )
                
                d_loss_x = self.disc_test_cm(
                    "DISC: B+M real antes de entrenamiento discriminadora",
                    all_batch_examples_BM,
                    all_batch_labels_cat_BM,
                    all_batch_labels_oh_BM,
                    msg2="Solo B+M real",
                    training=True,
                )
                '''

                # Calculate model loss
                d_loss = self.disc_test_cm_malgan(
                    "DISC: antes de entrenamiento discriminadora",
                    all_batch_examples,
                    all_batch_labels_cat,
                    all_batch_labels_oh,
                    msg2="1",
                    training=True,
                )
               

                # Calculate l2 reg loss
                loss_reg_l2_d = mc.add_model_regularizer_loss(self.discriminator)

                d_loss_tot = self.RATIO_LOSS_D * d_loss + self.RATIO_REG_D * loss_reg_l2_d
                tf.print(
                    "d_loss_tot= RATIO_LOSS_D * d_loss + RATIO_REG_D * loss_reg_l2_d\n",
                    d_loss_tot,
                    "=",
                    self.RATIO_LOSS_D,
                    d_loss,
                    self.RATIO_REG_D,
                    loss_reg_l2_d,
                )

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss_tot, self.discriminator.trainable_variables)

            if self.debug:
                tf.print("d_gradient", d_gradient)

            assert sum_gradient(d_gradient) != 0, "Gradient disc = 0"

            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

            # chequear
            if self.debugALB:
                d_loss_2 = self.disc_test_cm_malgan(
                    "DISC: despues de entrenamiento discriminadora",
                    all_batch_examples,
                    all_batch_labels_cat,
                    all_batch_labels_oh,
                    msg2="2",
                    training=False,
                )

        # --------------------------
        # Train the generator
        # --------------------------
        p_stats=True
        with tf.GradientTape() as tape:
            
            tf.print("TRAIN GEN")
            tf.print ("batch_size:", batch_size)
            # OJO_DEBUG
            # ---------
            #batch_size=15
            # ---------
            # Get the latent and malign inputs
            # random_latent_vectors = tf.random.normal(shape=(batch_size,self.latent_dim))
            random_latent_vectors = tf.random.uniform(shape=(batch_size, self.latent_dim), minval=-1, maxval=1)
            malign_inputs = mu.muestras.sample_examples(batch_size, class_label=MALIGN_LABEL)
            # malign_inputs = self.get_samples_malign_BB (batch_size)

            # Calculate model loss
            g_loss, gen_adversarials, gen_adversarial_labels_cat = self.gen_test_cm_malgan(
                "check gen antes de train", random_latent_vectors, malign_inputs, "gen_test_cm_malgan g_loss", training=True
            )

            # Calculate the regularization loss
            """
            loss_r1 = 0.
            if self.REG_LOSS:
                lr = self.generator.losses
                loss_r1 = tf.math.reduce_sum(lr)
            """
            loss_reg_l2_g = mc.add_model_regularizer_loss(self.generator)

            # Calculate the  distance loss
            loss_dist = 0.0
            num_samples_dist = batch_size

            ####### DIST_ALB
            if self.RATIO_DIST_G > 0:
                
                #num_samples_dist=150
                samples_fk = gen_adversarials[:num_samples_dist]
                samples_real = malign_inputs[:num_samples_dist]
                
                l1, l2, l3, l3_mean = dist.get_1(malign_inputs, samples_fk, inf=self.inf_train, sup=self.sup_train, 
                                                 stochastic=self.stochastic_train, umbral_dist_alb=self.umbral_dist_alb,
                                                 reverse=False, debug=False, p_stats=p_stats, 
                                                 tipo_distancia=self.tipo_distancia_train) 
                    
                tf.print ("dist_alb reverse=False: loss, loss_norm, loss_norm_t, loss_norm_t / len_loss_norm_t ",
                          l1,l2,l3,l3_mean)
                
               
                loss_dist = l3_mean
                tf.print ("loss_dist:",loss_dist)
                tf.print ("---")
            else:
                loss_dist=0.
            
            ##### EUCLIDEA p2p dist(x[i]-g[i])
            if self.beta > 0:
                tf.print ("---\ncalculo euclidea de beta:")
                samples_fk=gen_adversarials
                samples_real=malign_inputs
                perturbation = samples_fk - samples_real
                cuad=tf.square(perturbation)
                tf.print ("cuad:",type(cuad),cuad.shape,cuad)
                final=tf.sqrt(tf.reduce_sum(cuad, axis=1))
                tf.print ("sqr:",type(final),final.shape,final)
                pert_loss = tf.reduce_mean(final)
                pert_loss1 = tf.reduce_sum (final)
                pert_loss2= tf.reduce_mean(tf.norm (perturbation,axis=1))
                tf.print ("tf.norm shape",pert_loss2.shape,type(pert_loss2))
                tf.print ("euclidea desglosada,reduce_mean, reduce_sum, reduce_mean de norm:",
                          pert_loss,pert_loss1,pert_loss2)
            else:
                pert_loss=0.0
            
            # Total Cost
            # g_loss = f_loss + self.RATIO_REG * loss_r1 + self.RATIO_DIST * loss_dist
            assert (self.RATIO_DIST_G == 0) or (self.beta ==0), "No puede ser dist_alb y beta a la vez"
            g_loss_tot = self.RATIO_LOSS_G * g_loss + self.RATIO_DIST_G * loss_dist + self.RATIO_REG_G * loss_reg_l2_g + self.beta * pert_loss
            tf.print(
                "g_loss_tot = self.RATIO_LOSS_G * g_loss + self.RATIO_DIST_G * loss_dist+ self.RATIO_REG_G * loss_reg_l2_g + self.beta * pert_loss\n",
                "g_loss_tot",
                g_loss_tot,
                "=",
                self.RATIO_LOSS_G,
                g_loss,
                self.RATIO_DIST_G,
                loss_dist,
                self.RATIO_REG_G,
                loss_reg_l2_g,
                self.beta,
                pert_loss
            )

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss_tot, self.generator.trainable_variables)
        
        '''
        tf.print("\ngen_gradient", type(gen_gradient))
        for var, g in zip(self.generator.trainable_variables, gen_gradient):
            # in this loop g is the gradient of each layer
            tf.print ("------")
            tf.print(f'{var.name}, shape: {g.shape}',"gradients min,max,mean,std",np.min(g),np.max(g),np.mean(g),np.std(g))
            #tf.print("gradients..")
            #tf.print("gradients min,max,mean,std",np.min(g),np.max(g),np.mean(g),np.std(g))
        tf.print ("----\n")
        if self.debug:
            tf.print("gen_gradient", gen_gradient)
        '''

        if self.RATIO_LOSS_G == 0 and self.RATIO_DIST_G == 0 and self.RATIO_REG_G == 0:
            tf.print ("\n\n\nProbando SOLO Smirnov, GRADIENTE == 0.0\n\n\n")
        else:
            assert sum_gradient(gen_gradient) != 0, "Gradient gen = 0"

        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        # self.gen_test_pred_cm(gen_adversarials, gen_adversarial_labels_cat)
        
        #Debug distancias
        
        # -----------
        if False and self.RATIO_DIST_G > 0:
            tf.print ("Gradientes actualizados")
            tf.print ("uso las mismas muestras que las del entrenamiento para ver si mejora")
            l1, l2 = dist.get_1_np(samples_real, samples_fk, inf=0.0, sup=1,stochastic=True,debug=True) # 0.95
            tf.print ("dist_alb en get_1_np",l1,l2)
            
            tf.print ("uso otras muestras diferentes")
            rm1 = mu.muestras.sample_examples(batch_size, class_label=MALIGN_LABEL)
            gm1 = self.sample_generator(num_samples=batch_size, training=True)
            l1, l2 = dist.get_1_np(rm1, gm1, inf=0.0, sup=1,stochastic=True,debug=True) # 0.95
            tf.print ("despues de train dist_alb en get_1_np",l1,l2)
            #d1=mc.compute_metrics(gm1, rm1,"EU")

            '''
            random_latent_vectors = tf.random.uniform(shape=(batch_size, self.latent_dim), minval=-1, maxval=1)
            malign_inputs = rm1

            adversarial_examples = self.generator([malign_inputs, random_latent_vectors], training=False)
            gm2=gen_adversarials
            d2=mc.compute_metrics(gm2, rm1,"EU")
            print ("debug_distancias: ",d1,d2)
            '''
            
        #---------------------
        

        # check training generator
        if False and self.debugALB:
            tf.print ("\n----\nChequeo despues de train GEN con mismos datos que training")
            self.gen_test_cm_malign(
                "check con misma input (MG dataset) despues de train",
                random_latent_vectors,
                malign_inputs,
                "2",
                training=False,
            )
            
            
            #
            # random_latent_vectors = tf.random.normal(shape=(batch_size,self.latent_dim))
            tf.print ("\n----\nChequeo despues de train GEN con nuevos datos diferentes a los de training")
            random_latent_vectors = tf.random.uniform(shape=(batch_size, self.latent_dim), minval=-1, maxval=1)
            malign_inputs = mu.muestras.sample_examples(batch_size, class_label=MALIGN_LABEL)

            self.gen_test_cm_malign(
                "check con *distinta* input (MG dataset) despues de train",
                random_latent_vectors,
                malign_inputs,
                "3",
                training=True,
            )
            tf.print ("----")

        return {"d_loss": d_loss, "g_loss": g_loss}
    
    
    # -------------------------------
    
    def debug_train (self):
        xbatch_size=1000
        tf.print ("----\n uso muestras set #0a con self.sample generator 512")
        rm1 = mu.muestras.sample_examples(batch_size, class_label=MALIGN_LABEL)
        gm1 = self.sample_generator(num_samples=batch_size, training=True)
        l1, l2 = dist.get_1_np(rm1, gm1, inf=0.0, sup=1,stochastic=True,debug=False, 
                               tipo_distancia=self.tipo_distancia) # 0.95
        tf.print ("despues de train dist_alb en get_1_np",l1,l2)

        tf.print ("----\n uso muestras set #0b con self.sample generator 1000")
        rm1 = mu.muestras.sample_examples(xbatch_size, class_label=MALIGN_LABEL)
        gm1 = self.sample_generator(num_samples=xbatch_size, training=True)
        l1, l2 = dist.get_1_np(rm1, gm1, inf=0.0, sup=1,stochastic=True,debug=False, 
                               tipo_distancia=self.tipo_distancia) # 0.95
        tf.print ("despues de train dist_alb en get_1_np",l1,l2)


        # Generate adversarials from the latent vector and criptoattack (malign)
        malign_inputs = mu.muestras.sample_examples(xbatch_size, class_label=MALIGN_LABEL,debug=False)  # [:, f_UPC]
        malign_inputs2 = mu.muestras.sample_examples(xbatch_size, class_label=MALIGN_LABEL,debug=False) 


        # chequeo con set muestras #1
        tf.print ("-----\n uso muestras, set #1 1000")
        # random_latent_vectors = tf.random.normal(shape=(malign_inputs.shape[0], self.latent_dim))
        random_latent_vectors = tf.random.uniform(
            shape=(malign_inputs.shape[0], self.latent_dim), minval=self.min_rnd, maxval=self.max_rnd
        )
        training=True
        if self.solo_noise:
            adversarial_examples = self.generator(random_latent_vectors, training=training)
        else:
            adversarial_examples = self.generator([random_latent_vectors,malign_inputs], training=training)
        #adversarial_examples = self.generator.predict([malign_inputs, random_latent_vectors])

        l1, l2 = dist.get_1_np(malign_inputs,adversarial_examples, inf=0.0, sup=1,stochastic=True, 
                               tipo_distancia=self.tipo_distancia) # 0.95
        tf.print ("fuera de train dist_alb en get_1_np",l1,l2)

        # chequeo set #2
        tf.print ("-----\n uso otras muestras diferentes set #2 1000")
        random_latent_vectors = tf.random.uniform(
            shape=(malign_inputs.shape[0], self.latent_dim), minval=self.min_rnd, maxval=self.max_rnd
        )
        training=True
        if self.solo_noise:
            adversarial_examples2 = self.generator(random_latent_vectors, training=training)
        else:
            adversarial_examples2 = self.generator([random_latent_vectors,malign_inputs], training=training)
        #adversarial_examples2 = self.generator.predict([malign_inputs, random_latent_vectors])

        l1, l2 = dist.get_1_np(malign_inputs2,adversarial_examples2, inf=0.0, sup=1,stochastic=True, 
                               tipo_distancia=self.tipo_distancia) # 0.95
        tf.print ("fuera de train dist_alb en get_1_np",l1,l2)

        # chequeo 3 con self.sample_generator
        tf.print ("------\n uso otras muestras diferentes pero con sample_generator set #3 1000")
        rm1 = mu.muestras.sample_examples(xbatch_size, class_label=MALIGN_LABEL)
        gm1 = self.sample_generator(num_samples=xbatch_size, training=True)
        l1, l2 = dist.get_1_np(rm1, gm1, inf=0.0, sup=1,stochastic=True,debug=False, 
                               tipo_distancia=self.tipo_distancia) # 0.95
        tf.print ("fuera de train dist_alb en get_1_np",l1,l2)
            

    def train(self, epochs=10, batch_size=32, train_gen=True, **kwargs):
        """
        Main function training for training during epochs.
        """
        self.it = 0
        # self.lista_dist=[]
        # self.lista_dist_filtra_bb=[]
        self.lista_bh = []
        self.lista_mh = []
        self.lista_nh = []
        ##
        self.lista_bb = []
        self.lista_mm = []
        self.lista_gg = []
        #
        self.lista_mb = []
        self.lista_gb = []
        self.lista_gm = []
        #
        self.lista_mbf = []
        self.lista_gbf = []
        self.lista_gmf = []
        
        #
        # Debugging
        #
        '''
        for i in range(3):
            tf.print ("Para comparar la distribucion de referencia de 2 muestras malignas rm")
            #tf.print ("Comparo con distancias:",self.tipo_distancia_train)
            rm1= mu.muestras.sample_examples(512, class_label=MALIGN_LABEL)
            rm2= mu.muestras.sample_examples(512, class_label=MALIGN_LABEL)
            dist.get_1_pseudo_eu (rm1,rm2, self.tipo_distancia_train)

            l1_s, l2_s = dist.get_1_np(rm1,rm2, inf=self.inf_train, sup=self.sup_train, 
                                                 stochastic=True, umbral_dist_alb=np.inf,
                                                 reverse=False, debug=False, p_stats=True, 
                                                 tipo_distancia=self.tipo_distancia_train) 

            l1_s, l2_s = dist.get_1_np(rm1,rm2, inf=self.inf_train, sup=self.sup_train, 
                                                 stochastic=False, umbral_dist_alb=np.inf,
                                                 reverse=False, debug=False, p_stats=True, 
                                                 tipo_distancia=self.tipo_distancia_train) 
        '''
        # ######## TRAIN_BATCH
        
        assert (self.model_gan == "advgan") or (self.model_gan == "malgan") , "Modelo desconocido:"+self.model_gan
        
        tf.print ("MODELO: ",self.model_gan) # MALGAN o AdvGAN
        
        tr_batch= self.train_batch if self.model_gan == "advgan" else self.train_batch_MALGAN
        
        # ######## END TRAIN_BATCH
        
        for i in range(epochs):
            tf.print("######\nEPOCH", i, "\n########")
            #self.it = i
            self.epoch= i
            '''
            if train_gen:
                print(">>> Entrenando GAN")
                self.history.append(self.train_batch(batch_size, **kwargs))
            else:
                print(">>> Entrenando SOLO DISC")
                self.history.append(self.train_batch_solo_disc(batch_size, **kwargs))
            '''
            print(">>> Entrenando GAN")
            self.history.append(tr_batch(batch_size, **kwargs))

            tf.print("·" * 10)
            tf.print(">>> history", i, self.history[-1])
            
            # Compute histograms and cubitos
            
            xbatch_size=1000
            if False:
                self.debug_train ()
                
            # ---------
            tf.print ("Generar histograms and cubitos")
            
            benign_inputs = mu.muestras.sample_examples(xbatch_size, class_label=BENIGN_LABEL)
            # Generate adversarials from the latent vector and criptoattack (malign)
            malign_inputs = mu.muestras.sample_examples(xbatch_size, class_label=MALIGN_LABEL,debug=False)  # [:, f_UPC]
            malign_inputs2 = mu.muestras.sample_examples(xbatch_size, class_label=MALIGN_LABEL,debug=False) 

            # random_latent_vectors = tf.random.normal(shape=(malign_inputs.shape[0], self.latent_dim))
            random_latent_vectors = tf.random.uniform(
                shape=(malign_inputs.shape[0], self.latent_dim), minval=self.min_rnd, maxval=self.max_rnd
            )
            training=True
            if self.solo_noise:
                adversarial_examples = self.generator(random_latent_vectors, training=training)
            else:
                adversarial_examples = self.generator([random_latent_vectors,malign_inputs], training=training)

            l2_list = []

            #num_iguales=self.get_num_iguales (malign_inputs[:200],malign_inputs2[:200],tensor=False) 
            #tf.print ("iguales-rm1-rm2:",num_iguales)

            #Ojo
            #self.compute_histograms(adversarial_examples, malign_inputs, benign_inputs, l2_list)
            self.compute_histograms(adversarial_examples, malign_inputs2, benign_inputs, l2_list)

            dist_outliers,num_outliers,num_cub_ov,diffs=self.plot_cubitos (malign_inputs,malign_inputs2, "referencia")

            assert malign_inputs.shape[0] == num_cub_ov
            d_med= dist_outliers/num_outliers if num_outliers >0 else 0
            n=num_cub_ov-num_outliers
            alpha=n/num_cub_ov
            alpha_u=1.+alpha
            alpha_d=1.-alpha
            ratio=diffs/num_cub_ov
            tf.print ("Distancia outliers rm-rm:(tot,num_outliers, media)",dist_outliers,num_outliers,d_med, 
                      "num_cub_ov,diffs, alpha, ",num_cub_ov,diffs,alpha, 
                      f"1+a > d/k > 1-a. {alpha_u:.3f} > {ratio:.3f} > {alpha_d:.3f}" )
            
            # ------
            
            self.modulo_matrix_print=1
            
            if (self.epoch +1)%self.modulo_matrix_print == 0:
                
                # Metrics
                # self.iterate_metrics()
                tf.print("\n------------------\nDISTANCE_matrix\n----------------------")
                dist_eu, dist_ws = self.distance_matrix(act_metrics=True, num_muestras=self.k, 
                                                        distance_matrix="distance_matrix")
                # self.lista_dist.append()
                tf.print(
                    "\n------------------\nDISTANCE_bn filtrada solo los malignos_adv clasificados como benignos por BB\n----------------------"
                )
                
                '''
                try:
                    dist_eu_f, dist_ws_f = self.distances_bn(act_metrics=True, num_muestras=self.k, 
                                                         distance_matrix="distance_matrix_filter")
                except ZeroDivisionError:
                    dist_eu_f, dist_ws_f = (0. , 0. )
                    print("No hay muestras que pasen")
                '''
                dist_eu_f, dist_ws_f = self.distances_bn(act_metrics=True, num_muestras=self.k, 
                                                         distance_matrix="distance_matrix_filter")
        
                
                # self.lista_dist_filtra_bb.append()
                #
                tf.print ("calculate distance malign that pass BB wrt benigns")
                self.distance_benign_to_malign_filtered (num_muestras=self.k,ratio_ev_malign=3)

                # save distances
                tf.print ("dist_eu:",dist_eu, "epoch:",i)
                np.save(str(self.distances_dir) + "/dist_eu_epoch-" + str(i), dist_eu)
                np.save(str(self.distances_dir) + "/dist_ws_epoch-" + str(i), dist_ws)

                np.save(str(self.distances_dir) + "/dist_eu_f_epoch-" + str(i), dist_eu_f)
                np.save(str(self.distances_dir) + "/dist_ws_f_epoch-" + str(i), dist_ws_f)

                #
                # euc, wass, (3,3)
                '''
                x = dist_eu
                bb = x[0, 0]
                mm = x[1, 1]
                gg = x[2, 2]
                mb = (x[0, 1] + x[1, 0]) / 2
                gb = (x[0, 2] + x[2, 0]) / 2
                gm = (x[1, 2] + x[2, 1]) / 2
                #
                x = dist_eu_f
                mbf = (x[0, 1] + x[1, 0]) / 2
                gbf = (x[0, 2] + x[2, 0]) / 2
                gmf = (x[1, 2] + x[2, 1]) / 2
                '''
                
                # Metrics hits BB
                bh, mh, nh = self.bb_hits()
                
                # Guardar las metricas
                with open(self.metrics_dir / f"metrics_dict_it_{self.it:>05}.plk", "wb") as f:
                    pickle.dump(self.metrics_dict, f)
                tf.print ("self.metrics_dict[distance_matrix] despues de salvar en fich:\n",
                          str(self.metrics_dir / f"metrics_dict_it_{self.it:>05}.plk"),"\n",
                          self.metrics_dict["distance_matrix"]["EU"][f"{self.it:>05}"])
                
                # Save metrics
                tf.print ("Guardar BBhits en",str(self.metrics_dir) + "/bh_epoch-" + str(self.it))
                np.save(str(self.metrics_dir) + "/bh_epoch-" + str(self.it), bh)
                np.save(str(self.metrics_dir) + "/mh_epoch-" + str(self.it), mh)
                np.save(str(self.metrics_dir) + "/nh_epoch-" + str(self.it), nh)

                self.lista_bh.append(bh)
                self.lista_mh.append(mh)
                self.lista_nh.append(nh)
                
                self.it += 1

            # Clustering
            #self.evaluate_clustering()

            # Pinta
            # x = self.lista_bh
            # y = self.lista_mh
            # z = self.lista_nh

            # plt.plot(x, label="benign h")
            # plt.plot(y, label="malign h")
            # plt.plot(z, label="noise h")
            # plt.legend(loc="upper left")
            # # plt.show()
            # plt.savefig(str(self.bb_hits_plots_dir) + "bb_hits-" + str(i) + ".png")

            # if i % 50 == 0:
            #     self.save()
            
            if (self.epoch % 100) ==0:
                path_gen=f"./{self.model_gan}_{self.dataset}_output/{self.exp_name}/{self.trial_id}/GAN_models/gen_{self.epoch}.h5"
                tf.print ("guardando generadora en:",path_gen)
                self.generator.save (path_gen)
        
        #tf.print("#########\nRESUMEN FINAL\n##########")
        #self.distance_matrix()
        # self.iterate_metrics()

    def no_train(self, epochs=10, batch_size=32):
        """
        Main function training for training during epochs.
        """
        self.it = 0
        self.lista_dist = []
        self.lista_dist_filtra_bb = []
        self.lista_bh = []
        self.lista_mh = []
        self.lista_nh = []

        for i in range(epochs):
            tf.print("######\nEPOCH", i, "\n########")
            self.it = i

            # Metrics
            bh, mh, nh = self.bb_hits()

            # Save metrics
            np.save(str(self.metrics_dir) + "bh_no_train_epoch-" + str(i), bh)
            np.save(str(self.metrics_dir) + "mh_no_train_epoch-" + str(i), mh)
            np.save(str(self.metrics_dir) + "nh_no_train_epoch-" + str(i), nh)

            self.lista_bh.append(bh)
            self.lista_mh.append(mh)
            self.lista_nh.append(nh)

            # Pinta
            # x = self.lista_bh
            # y = self.lista_mh
            # z = self.lista_nh

            # plt.plot(x, label="benign h")
            # plt.plot(y, label="malign h")
            # plt.plot(z, label="noise h")
            # plt.legend(loc="upper left")
            # # plt.show()
            # plt.savefig(str(self.bb_hits_plots_dir) + "bb_hits_no_train_epoch-" + str(i))

    
    def save(self):
        """
        Save the models separately.
        """
        self.discriminator.save(self.models_dir / f"disc_{self.it}.h5")
        self.generator.save(self.models_dir / f"gen_{self.it}.h5")

    # ----------------------
    # OBSOLETO
    # ----------------------

   
