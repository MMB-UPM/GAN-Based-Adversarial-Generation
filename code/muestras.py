"""
Module Muestras
"""

import numpy as np
import tensorflow as tf
import pickle

BENIGN = 0
MALIGN = 1



# CLASS DEFINITION
class Muestras:
    def __init__(self, dir="./", lista_features=[],RATIO_STD_CORTE=None,tipo_scaler="",scaler_path="",dataset_features=[]):
        
        tf.print ("dir:",dir)
        tf.print(f"Using dataset located in {dir}")

        self.tipo_scaler=tipo_scaler
        tf.print ("TIPO SCALER:",tipo_scaler)
        if tipo_scaler not in ["Standard", "MaxMin"]:
            tf.print ("error scaler inexistente:",tipo_scaler)
            pause()
        
        self.scaler_path=scaler_path
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        self.feats=dataset_features
        
        self.contador = 0
        self.dataset = None

        x = np.load(dir + "data2_scaled.npy")

        if len(lista_features) > 0:
            tf.print("Selecting features from X:", lista_features)
            x = x[:, lista_features]

        y = np.load(dir + "labels2.npy")
        
        #Quitar elementos con valores muy altos (outliers)
        ncols=x.shape[1]
        print ("num filas:",x.shape[0])
        
        indices0 = np.where(y == BENIGN)[0]
        indices1 = np.where(y == MALIGN)[0]
        
        self.tot_rows=x.shape[0]
        self.tot_labelB=np.sum(y == BENIGN)
        self.tot_labelM=np.sum(y == MALIGN)
        
        
        #UMBRAL_OUTLIERS=np.std(x,axis=0)*5.0
        #RATIO_STD_CORTE=5.0
        #UMBRAL_OUTLIERS=np.std(x,axis=0)*RATIO_STD_CORTE
        
        tf.print ("tot min,max,mean,std",np.min(x,axis=0),np.max(x,axis=0),np.mean(x,axis=0),np.std(x,axis=0))
        xx=x[indices0]
        tf.print ("indices 0 min,max,mean,std",np.min(xx,axis=0),np.max(xx,axis=0),np.mean(xx,axis=0),np.std(xx,axis=0))
        xx=x[indices1]
        tf.print ("indices 1 min,max,mean,std",np.min(xx,axis=0),np.max(xx,axis=0),np.mean(xx,axis=0),np.std(xx,axis=0))
        
        # CTU : RATIO_STD_CORTE=[0.5,0.5,0.5,0.5,5.0,2.2,8.0]
        if RATIO_STD_CORTE != None:
            UMBRAL_OUTLIERS=np.std(x,axis=0)*RATIO_STD_CORTE
            tf.print ("RATIO_STD_CORTE:",RATIO_STD_CORTE,"std:",np.std(x,axis=0),"UMBRAL_OUTLIERS:",UMBRAL_OUTLIERS)
            
            #UMBRAL_OUTLIERS=3.5
            #UMBRAL_OUTLIERS=np.inf # Dberia ser un array de los umbrales de cada feature
            if any(UMBRAL_OUTLIERS < np.inf):
                for i in range(ncols):
                    ids=x[:,i]<UMBRAL_OUTLIERS[i]
                    tf.print ("long:",len(ids))
                    x=x[ids]
                    y=y[ids]
        else: 
            tf.print ("NO HAY RATIO DE CORTE. Se usan todas las muestras")

        indices0 = np.where(y == BENIGN)[0]
        indices1 = np.where(y == MALIGN)[0]
        
        tf.print ("despues de recortar")
        tf.print ("tot min,max,mean,std",np.min(x,axis=0),np.max(x,axis=0),np.mean(x,axis=0),np.std(x,axis=0))
        xx=x[indices0]
        tf.print ("indices 0 min,max,mean,std",np.min(xx,axis=0),np.max(xx,axis=0),np.mean(xx,axis=0),np.std(xx,axis=0))
        xx=x[indices1]
        tf.print ("indices 1 min,max,mean,std",np.min(xx,axis=0),np.max(xx,axis=0),np.mean(xx,axis=0),np.std(xx,axis=0))
        
        
        # Reducir el dataset
        #NUM_SAMPLES=10000
        #indices0=np.random.choice(indices0,NUM_SAMPLES)
        #indices1=np.random.choice(indices1,NUM_SAMPLES)

        self.dataset = {"x": x, "y": y, "indices0": indices0, "indices1": indices1}

        tf.print("Dataset muestras: x.shape, y.shape, len(indices0), len(indices1)")
        tf.print(
            self.dataset["x"].shape,
            self.dataset["y"].shape,
            len(self.dataset["indices0"]),
            len(self.dataset["indices1"]),
        )
        
        #Para hacer cubitos
        self.x1et= x[y==MALIGN]
        self.rango_x= np.max(self.x1et,axis=0)-np.min(self.x1et,axis=0)
        self.min_x=np.min(self.x1et,axis=0)
        tf.print (f"muestras rango_x:{self.rango_x} min_x:{self.min_x}")
        self.primeravez=True

    def sample_examples(self, batch_size, class_label,debug=False):
        """
        Sample examples of class cl=(0,1) : (normal_traffic, cripto_traffic)
        from dataset in ds_dir, with batch_size given and subset test (test=True)
        or train (test=False)
        """
        str_ind = "indices0" if class_label == BENIGN else "indices1"
        indices = self.dataset[str_ind]
        #tf.print ("sample_examples muestras. cojo de indice:",str_ind)
        x = self.dataset["x"]

        if batch_size > 0:
            replace = len(indices) < batch_size
            idx = np.random.choice(indices, batch_size, replace=replace)
        else:
            idx = indices

        muestra=x[idx]
        add_rnd=False
        xx=muestra
        #print ("muestra",class_label,
        #       "min,max,mean,std",np.min(xx,axis=0),np.max(xx,axis=0),np.mean(xx,axis=0),np.std(xx,axis=0))
        if debug:
            tf.print ("muestra",class_label,
               "min,max,mean,std",np.min(xx,axis=0),np.max(xx,axis=0),np.mean(xx,axis=0),np.std(xx,axis=0))
        
        if self.primeravez:  
            self.primeravez=False
            if debug:
                tf.print ("TODAS sample_example:\n")
                for e in x[indices]:
                    tf.print (e)
                tf.print ("--")
                
        if debug:         
            tf.print ("sample_example:\n")
            for e in muestra:
                tf.print (e)
                
        if add_rnd and class_label == BENIGN:
            tf.print ("adding noise to benign sample....")
            val_rnd=np.random.rand(*muestra.shape)*3.5
            muestra+=val_rnd
            tf.print ("medias:",np.mean(muestra,axis=0)," std:",np.std(muestra,axis=0))
            
                      
        
        return muestra
    
    def inverse_normalise (self,scaled_examples,tensor=False):
    
        if tensor: #Con tensores
            if self.tipo_scaler == "Standard":
                var = self.scaler.var_[self.feats]
                std = np.sqrt(var)
                mean = self.scaler.mean_[self.feats]
                # convert to tensor
                var = tf.convert_to_tensor(var, dtype=tf.float32)
                mean = tf.convert_to_tensor(mean, dtype=tf.float32)

                non_scaled_examples = tf.add(
                    tf.cast(tf.multiply(scaled_examples, std), tf.float32), tf.cast(mean, tf.float32)
                )

                #tf.print ("var:",var.shape,"mean",mean.shape,"examples",all_batch_examples.shape)

            else: # "MaxMin"
                # Manually reverse the scaling
                scale = self.scaler.scale_[self.feats]
                min_value = self.scaler.min_[self.feats]          

                #original_data = (rm1_eu - min_value)/ scale
                non_scaled_examples = tf.cast(
                    tf.divide(tf.subtract(scaled_examples, min_value), scale),
                    tf.float32
                )

        else:  #Sin tensores
            if self.tipo_scaler == "Standard":
                var = self.scaler.var_[self.feats]
                std = np.sqrt(var)
                mean = self.scaler.mean_[self.feats]
                non_scaled_examples= scaled_examples*std + mean

            else: # "MaxMin"
                # Manually reverse the scaling
                scale = self.scaler.scale_[self.feats]
                min_value = self.scaler.min_[self.feats]           

                #original_data = (rm1_eu - min_value)/ scale
                non_scaled_examples= (scaled_examples - min_value)/ scale
        
        return non_scaled_examples 


#muestras = Muestras(dir="./dataset/crypto/")
#muestras.contador += 1

#print("Contador:", muestras.contador)
