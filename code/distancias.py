"""
Module distancias
last version: 21/03/2022

Uso:
    l1,l2,l3=get_1_np (samples_real,samples_fk,inf=0.0,sup=0.95)
    return np.array([l1,l2,l3])/sample_size

"""

import numpy as np
import tensorflow as tf
import muestras as mu
import random
import time
from scipy.spatial.distance import cityblock

import numpy as np
import matplotlib.pyplot as plt
import ot  # LA LIBRERIA IMPORTANTE
#import distancias as dt

from ot_tf import dmat as dmat_tf, sink as sink_tf

#import tf_wasserstein as twa
#from importlib import reload

#import ot_tf as ottf

#reload(twa)
#reload (ottf)


def moving_average(x):
    return np.convolve(x, np.ones(2), 'valid') / 2

# LA FUNCION PRINCIPAL
def wasser_Angel(X, Y, n_bins = 1000):    
    bin_edgesX = np.linspace(X.min(), X.max(), n_bins)
    histX, bin_edgesX = np.histogram(X, bins=bin_edgesX)
    binsX = moving_average(bin_edgesX)
    histX = histX/len(X)

    bin_edgesY= np.linspace(Y.min(), Y.max(), n_bins)
    histY, bin_edgesY = np.histogram(Y, bins=bin_edgesY)
    binsY = moving_average(bin_edgesY)
    histY = histY/len(Y)
    
    C = ot.dist(binsX.reshape((len(binsX),1)), binsY.reshape((len(binsY),1)), 'euclidean')
    return ot.emd2(histX.astype('float64'), histY.astype('float64'), C)

def wasser_aprox_orig (muestra_a, muestra_b, sh_reg= 5.5, sh_numItermax= 1000*2, sh_stopThr= 1e-9):    

    na=muestra_a.shape[0]
    nb=muestra_b.shape[0]
    '''
    En API
    numItermax_ws_sh = 1000
    
    stopThr_ws_sh = 1e-9
    
    reg_ws_sh = 5.5 #no hay valor por defecto.
    '''
        
    M = ot.dist(muestra_a.copy(), muestra_b.copy(), metric='euclidean')
    a = np.ones((na,)) / na
    b = np.ones((nb,)) / nb  # uniform distribution on samples
    
    pot_sinkhorn_loss = ot.sinkhorn2(a, b, M, reg=sh_reg, numItermax=sh_numItermax, 
                                     stopThr=sh_stopThr)
    
    
    return pot_sinkhorn_loss

def wasser_aprox (muestra_a, muestra_b, sh_reg= 5.5, sh_numItermax= 1000*2, sh_stopThr= 1e-9, sh_method="sinkhorn"):    

    na=muestra_a.shape[0]
    nb=muestra_b.shape[0]
    '''
    En API
    numItermax_ws_sh = 1000
    
    stopThr_ws_sh = 1e-9
    
    reg_ws_sh = 5.5 #no hay valor por defecto.
    '''
        
    M = ot.dist(muestra_a.copy(), muestra_b.copy(), metric='euclidean')
    cost_matrix=M
    a = np.ones((na,)) / na
    b = np.ones((nb,)) / nb  # uniform distribution on samples
    
    #pot_sinkhorn_loss_old = ot.sinkhorn2(a, b, M, reg=sh_reg, numItermax=sh_numItermax, 
    #                                 stopThr=sh_stopThr)
    # method = 'sinkhorn_epsilon_scaling'
    #print (f"wasser_aprox: method:{sh_method}")
    optimal_transport_plan = ot.sinkhorn(a, b, M, reg=sh_reg, numItermax=sh_numItermax, 
                                     stopThr=sh_stopThr, method=sh_method)
    
    pot_sinkhorn_loss= np.sum(optimal_transport_plan * cost_matrix)
    
    #diff=np.abs(pot_sinkhorn_loss_old - pot_sinkhorn_loss) 
    
    #assert diff < 0.000001, f"Error wasser_aprox ws_sh, diff sinkhorn*M y sinkhorn2:{diff}"
    
    return pot_sinkhorn_loss



def wasser_aprox_tf_mal (muestra_a, muestra_b, sh_reg= 5.5, sh_numItermax= 1000*2, sh_stopThr= 1e-9): 
    return twa.sinkhorn_loss(muestra_a,muestra_b,sh_reg,muestra_a.shape[0],sh_numItermax)

def wasser_aprox_tf (muestra_a, muestra_b, sh_reg= 5.5, sh_numItermax= 1000*2, sh_stopThr= 1e-9):
    na=muestra_a.shape[0]
    nb=muestra_b.shape[0]
    muestra_a=tf.cast(muestra_a,dtype=tf.float64)
    muestra_b=tf.cast(muestra_b,dtype=tf.float64)
    M_tf=dmat_tf(muestra_a, muestra_b)
    tf_sinkhorn_loss = sink_tf(M_tf, (na, nb), sh_reg, numItermax=sh_numItermax, stopThr=sh_stopThr)
    
    return tf_sinkhorn_loss

def wasser_aprox_2 (muestra_a, muestra_b, sh_reg= 5.5, sh_numItermax= 1000*2, sh_stopThr= 1e-9):    

    na=muestra_a.shape[0]
    nb=muestra_b.shape[0]
    
    '''
    En API
    numItermax_ws_sh = 1000
    
    stopThr_ws_sh = 1e-9
    
    reg_ws_sh = 5.5 #no hay valor por defecto.
    '''
        
    M = ot.dist(muestra_a.copy(), muestra_b.copy(), metric='euclidean')
    cost_matrix=M
    a = np.ones((na,)) / na
    b = np.ones((nb,)) / nb  # uniform distribution on samples
    
    matriz_asignacion = ot.sinkhorn(a, b, M, reg=sh_reg, numItermax=sh_numItermax, stopThr=sh_stopThr)
    
    print ("np.sum(matriz_asignacion):",np.sum(matriz_asignacion))
    error=False
    for i in range(matriz_asignacion.shape[0]):
        tot_prob=np.sum(matriz_asignacion[i])
        if tot_prob < 1/na*0.999:
            print (f"error wasser_aprox_v2. tot_prob={tot_prob} en fila {i}")
            error=True
    if error:
        pause()
    cols_asig=np.argmax(matriz_asignacion,axis=1)
    cols_uniq=len(np.unique(cols_asig))
    if cols_uniq != na:
        print (f"wasser_aprox_v2 error asignacion. una col asignada a varias filas")
        print ("cols_asig",cols_asig)
        cols_asig.sort()
        print ("cols_asig",cols_asig)
        print ("cols_uniq len",cols_uniq)
        pause()
    optimal_transport_plan= np.eye(n, dtype=int)[:,cols_asig]
    wasserstein_distance = np.sum(optimal_transport_plan * cost_matrix)
    
    return  wasserstein_distance    
        
    
    # ver si podemos obtener l matriz de probabilidades para asignar file-columna en vez de probabilidades
    

def wasser (muestra_a, muestra_b, ws_numItermax=100000*6): 
    
    na=muestra_a.shape[0]
    nb=muestra_b.shape[0]
    #, numItermax=100000*10, ,numThreads="max" por defecto en API
    
    cost_matrix = ot.dist(muestra_a.copy(), muestra_b.copy(), metric='euclidean')
    #print ("dt.cost_matrix.shape:",cost_matrix.shape)
           
    # Calcular la matriz de asignación y la distancia de Wasserstein
    #print ("ws numItermax=",ws_numItermax)
    optimal_transport_plan = ot.emd(np.ones(na), np.ones(nb), cost_matrix, numItermax=ws_numItermax)
    #print ("dt wasser. cuenta los 1 al final del proceso:",np.sum(optimal_transport_plan))
    n_asig=np.sum(optimal_transport_plan)
    assert n_asig == na , f"distancias.wasser np.sum(optimal_transport_plan) != num_samples, {n_assig} , {na}"
    wasserstein_distance = np.sum(optimal_transport_plan * cost_matrix)

    return  wasserstein_distance

def get_dist_WK(sample_a, sample_b,debug=False,num_samples=None,dist_alb=False,
                tipo_distancia="normal",  stochastic=False, inf=0.0, sup=1.0):
    
    debug=False
    
    #tf.print ("voy a get_1_np")
    #t1=time.time()
    if num_samples == None:
        num_samples= min (sample_a.shape[0],sample_b.shape[0])
        
    # sample_a=sample_a.numpy()
    # sample_b=sample_b.numpy()
    # np.random.shuffle(sample_a)
    # np.random.shuffle(sample_b)
    sample_a = sample_a[:num_samples]
    sample_b = sample_b[:num_samples]
    
    if "wasserstein".startswith (tipo_distancia):
        pause()
        #No se quien llama a esto
        return wasser(sample_a,sample_b)*num_samples
    else:
        if dist_alb:
            d_ab, d_ab_norm = get_1_np(sample_a, sample_b,  reverse=False, debug=debug,
                                       tipo_distancia=tipo_distancia, stochastic=stochastic, inf=inf, sup=sup)
            return d_ab_norm
            #d_ab2, d_ab_norm2 = get_1_np(sample_a, sample_b,inf=inf,sup=sup,reverse=True,debug=debug,tipo_distancia=tipo_distancia)
            #return (d_ab_norm+d_ab_norm2)/2.0
        else:
            assert 1 == 0 ,"get_dist_WK solo puede ser dista_alb"
            pause()
            #d_ab_norm= np.sum(np.linalg.norm(sample_a-sample_b,axis=1))
            #return d_ab_norm
                              
    # d_ba,d_ba_norm= get_1_np(sample_b,sample_a)
    # d_med=(d_ab+d_ba)/2.
    # print ("distancias, ab, ab_norm",d_ab,d_ab_norm)
    #t2=time.time()
    #tf.print (f"tiempo get_1_np:{t2-t1:0.3f}")
    #return (d_ab_norm+d_ab_norm2)/2.0



def get_1_pseudo_eu (adversarial_examples, malign_inputs, tipo_distancia_train="normal", p_stats=True,debug=False):
    perturbation = adversarial_examples - malign_inputs
    #tf.print ("adversarial_examples\n",adversarial_examples.shape,adversarial_examples)
    #tf.print ("malign_inputs\n",malign_inputs.shape,malign_inputs)
    #tf.print ("pertur:\n",perturbation)
    tf.print ("tipo de distancia aplicada en beta:",tipo_distancia_train)
    
                  
    if tipo_distancia_train == "cuadrados":
        final=tf.reduce_sum(tf.square(perturbation), axis=1)
    else:
        final=tf.sqrt(tf.reduce_sum(tf.square(perturbation), axis=1))
        
    if debug: 
        tf.print ("perturbation:\n",np.array(perturbation))
        tf.print ("distancias:\n",np.array(final))
        
    pert_loss = tf.reduce_mean(final)
    pert_loss1 = tf.reduce_sum(final)
    #pert_loss2= tf.reduce_mean(tf.norm (perturbation,axis=1))
    tf.print("pert_loss (antes de mean) final.shape", final.shape)  
    tf.print ("euclidea desglosada,reduce_mean, reduce_sum:", pert_loss,pert_loss1)
    if p_stats:
        l_val= final.numpy()
        tf.print (f"\npseudo-get_1 lista_pp:beta_eucl min:{np.min(l_val)}, max:{np.max(l_val)}, media:{np.mean(l_val)}, \n lista_pp:beta_eucl p90:{np.percentile(l_val,90)}, p75:{np.percentile(l_val,75)}, p25:{np.percentile(l_val,25)}, p10:{np.percentile(l_val,10)}")
        tf.print(f"pert_loss suma:{pert_loss1} media:{pert_loss}")
    return pert_loss, pert_loss1
    

def get_1_np(X_train, images, M=None, umbral_dist_alb=np.inf, inf=0.0, sup=0.95, debug=False,reverse=False, p_stats=False, stochastic=True, tipo_distancia="normal"):
    return get_1(X_train, images, M=M, umbral_dist_alb=umbral_dist_alb, inf=inf, sup=sup, debug=debug,p_stats=p_stats, tensor=False, reverse=reverse, stochastic=stochastic, tipo_distancia=tipo_distancia)

def dmat(x, y):
    """
    :param x: (na, 2)
    :param y: (nb, 2)
    :return:
    """
    x=tf.cast(x,dtype=tf.float32)
    y=tf.cast(y,dtype=tf.float32)
    mmp1 = tf.tile(tf.expand_dims(x, axis=1), [1, y.shape[0], 1])  # (na, nb, 2)
    mmp2 = tf.tile(tf.expand_dims(y, axis=0), [x.shape[0], 1, 1])  # (na, nb, 2)
    #tf.print ("shapes: na, nb, 2",mmp1.shape,mmp2.shape)
    mm = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(mmp1, mmp2)), axis=2))  # (na, nb)

    return mm

def dmat_np(x, y):
    """
    :param x: (na, 2)
    :param y: (nb, 2)
    :return:
    """
    mmp1 = np.tile(np.expand_dims(x, axis=1), [1, y.shape[0], 1])  # (na, nb, 2)
    mmp2 = np.tile(np.expand_dims(y, axis=0), [x.shape[0], 1, 1])  # (na, nb, 2)
    #tf.print ("shapes: na, nb, 2",mmp1.shape,mmp2.shape)
    mm = np.sqrt(np.sum(np.square(np.subtract(mmp1, mmp2)), axis=2))  # (na, nb)

    return mm


def get_1(X_train, images, M=None, inf=0.0, sup=0.95, debug=False, tensor=True,stochastic=True, reverse=False, umbral_dist_alb=np.inf,p_stats=False,msg="",tipo_distancia="normal"):
    
    #debug=False
    
    #tf.print ("\nget_1 sup=",sup, "stochastic:",stochastic, "umbral",umbral_dist_alb, "reverse:",reverse, "tipo_distancia:",tipo_distancia)
    #tf.print ("\nget_1 sup=",sup, "umbral",umbral_dist_alb, "reverse:",reverse, "tipo_distancia:",tipo_distancia)
    assert tipo_distancia != "cuadrados" , "ya no utilizamos distancia cuadrados para los plots"
    
    # assert (X_train.shape[0]==images.shape[0])
    loss, loss_norm, loss_norm_t = 0, 0, 0  # tf.constant(0.,dtype=tf.float32),0.,tf.constant(0.,dtype=tf.float32)
    # Filtras en la funcion de coste valores muy pequeños. aqui los contabilizamos
    MIN_DIST_EUCL = 0
    UMBRAL_DIST=umbral_dist_alb #20. # 0.45
    
    # try:
    N = X_train.shape[0]
    K = X_train.shape[-1]
    #_images = images  # .numpy() 
    
    #Convertir a numpy array para hacer mas rapido calculos de distancia
    if tf.is_tensor(images): 
        _images = images.numpy() 
    else:
        _images= images
    
        
    if M == None:
        M = int(N * 1.1)
        # print ("N M default:", N, M)
    M = min(M, _images.shape[0])
    # tf.print ("_images0:",_images.shape,type(_images))
    _images = _images[:M]
    
    if tensor:
        images = images[:M]
    
    # tf.print ("_images1:",_images.shape,type(_images))

    X_train = np.reshape(X_train, (N, K))
    '''
    if tensor:
        _images = tf.reshape(_images, (M, K))
    else:
        _images = np.reshape(np.array(_images),(N, K))
    '''
    
    #_images = np.reshape(np.array(_images),(N, K))
    _images = np.reshape(_images,(N, K))
    
    #if reverse:
    #    tmp=X_train
    #    X_train=_images
    #    _images=tmp
    assert M == N
    #tf.print ("get_1 (M,N,K):", M,N,K)
    
    if debug:
        im=np.round (_images,3)
        xx=np.round (X_train,3)
        tf.print ("X_train mal:\n",xx)
        tf.print ("images gen:\n",im)
        
    #if debug:
    #    tf.print("get_1 X_train:", X_train.shape)
    #    tf.print ("DATOS_1:\n",X_train)
    #    tf.print ("DATOS_2:\n", _images)
        
    #distancias = np.full((N, M), np.inf)
    #distancias2 = np.full((N, M), np.inf)
    #distancias_eu = np.full((N, M), np.inf)
    
    #t1=time.time()
    if tipo_distancia == "manhattan":
        tf.print (">>>distancia manhattan")
        assert not tensor , "no puede haber distancia manhattan con tensores"
        for i in range(N):
            # i: X_train
            for j in range(M):
                #j: images
                distancias_eu[i,j]= cityblock(X_train[i],_images[j])
                
        
    else:
        if tensor:
            #tf.print (">>>distancia normal",tipo_distancia)
            x_tf = tf.keras.layers.InputLayer(dtype=tf.float32, input_shape=X_train.shape[1:])
            x_tf = x_tf(tf.cast(X_train,tf.float32))

            y_tf = tf.keras.layers.InputLayer(dtype=tf.float32, input_shape=images.shape[1:])
            y_tf = y_tf(tf.cast(images,tf.float32))

            #t1=time.time()
            d_fast= dmat (x_tf,y_tf)
            #d_fast= dmat_np (X_train,_images)
            #t2=time.time()
            #tf.print ("d_fast time:",t2-t1)
            #t1=time.time()
            #C = ot.dist(X_train, _images, 'euclidean')
            C= d_fast.numpy()
            #t2=time.time()
            #tf.print ("C time:",t2-t1)

            #tf.print ("shapes:",d_fast.shape,C.shape)
            #tf.print ("diff mats",np.max(np.absolute(d_fast-C)))
            #t2=time.time()
            #tf.print ("d_fast time:",t2-t1)
            #tf.print ("d_fast:",type(d_fast),d_fast.shape,"\n",d_fast)
            distancias_eu=C
        else:
            d_fast=dmat_np(X_train,images)
            distancias_eu=d_fast
        
        
        
    #t2=time.time()
    #tf.print ("d_eu time:",t2-t1)
    #tf.print ("d_eu:",type(distancias_eu),distancias_eu.shape,"\n",distancias_eu)
    #difer= np.sum(distancias_eu - d_fast)
    #tf.print ("diferencias:",difer)
    #
    # print (distancias)
    
    
    """
    if not tensor:
        print ("--- distancias -----")
        for i in range(distancias.shape[0]):
            print ("\n i:",i," >>> ")
            for j in range(distancias.shape[1]):
                print (round(distancias[i,j],2),end=" , ") 
        print ("--- distancias -----")
    """
    
    if debug:
        dd=np.round (distancias_eu,4)
        tf.print ("distancias:\n",dd)
        #dd=np.round (distancias2,4)
        #tf.print ("distancias2:\n",dd)
        
    
    rango = N  # int(N*(1-restos))
    # loss=tf.constant(0.,dtype=tf.float32) 
    lista_loss = []
    lista_loss_x=[]
    loss=0.0
    # lista_loss1=[]
    
    if tipo_distancia == "cuadrados":
        #tf.print ("get_1 cuadrados")
        distancias= distancias2
        assert False , "No hay distancias cuadrados"
    else:
        #tf.print ("get_1 normal")
        distancias= distancias_eu
    
    #tf.print ("tipo d_fast:",type(d_fast[0,0].numpy()))
    '''
    t1=time.time()
    indices_min=np.argmin(distancias, axis=1)
    base=0
    t2=time.time()
    tf.print ("time np.argmin TODA matriz",t2-t1)
    '''
    #lista_loss=  [(0, 0, 0) for _ in range(N)]
    #t10=time.time()
    if stochastic:
        for k in range(rango):

            min_j=np.argmin(distancias[k,:])
            min_i=k
            min_d=d_fast[min_i,min_j]
             
            lista_loss.append((min_d, min_i, min_j))
            lista_loss_x.append (min_d)
            distancias[:,min_j]=np.inf

            loss+= min_d
    else: 
        # GReady. matriz NxM
        #indexado en filas i

        col_min_values = np.argmin (distancias,axis=1)
        #print ("col_min_values",col_min_values)
        min_values= distancias [np.arange(N),col_min_values]
        #indexado en cols j
        rows_per_col_min= np.empty (M, dtype=object)
        #rows_per_col_min.fill([])
        for i in range(rows_per_col_min.shape[0]):
            rows_per_col_min[i]= []
        #[rows_per_col_min[v].append(i) for i,v in enumerate(col_min_values)]
        
        for i, v in enumerate(col_min_values):
            #print ("voy a meter fila i",i," en pos array columna v",v)
            rows_per_col_min[v].append(i)
       
        #procesado_i = np.full((N,), False)
        #procesado_j = np.full((M,), False)
        for k in range(N):
            #print ("procesando ronda k",k)
            row_min=np.argmin (min_values)
            #print ("fila min:",row_min)
            col_min=col_min_values[row_min]
            min_d = distancias[row_min, col_min]
            min_values[row_min]= np.inf
            col_min_values[row_min] = -1
            distancias[:,col_min]=np.inf
            #print ("columna min",col_min)
            if len(rows_per_col_min[col_min]) >1:
                # hay otras filas afectadas, recolocalas
                #print ("otras filas afectadas",rows_per_col_min[col_min])
                for i in rows_per_col_min[col_min]:
                    #print ("recolocando fila",i)
                    if i != row_min:
                        col=np.argmin (distancias[i,:])
                        col_min_values[i]=col
                        min_values[i]= distancias[i,col]
                        rows_per_col_min[col].append(i)
                
            rows_per_col_min[col_min]=[]
            min_i=row_min
            min_j=col_min
            
            '''
            posicion_minimo = np.unravel_index(np.argmin(distancias), distancias.shape)
            min_i, min_j = posicion_minimo
            min_d = distancias[min_i, min_j]
            '''

            '''
            min_d = np.inf
            for i in range(N):
                if not procesado_i[i]:
                    j=np.argmin(distancias[i,:])
                    if distancias[i,j] < min_d:
                        min_d = distancias[i, j]
                        min_i, min_j = i, j
            '''
            
            lista_loss.append((min_d, min_i, min_j))
            lista_loss_x.append (min_d)
            
            '''
            distancias[:,min_j]=np.inf
            l_min_i.append(min_i)
            if (k % N_RECALC) == 0:
                distancias = np.delete(distancias, l_min_i, axis=0)
                l_min_i=[]
                #get_minimos
            '''
            
                
            #procesado_i[min_i]=True

            loss+= min_d
        
    '''
    else: 
        # GReady. matriz NxM
        #procesado_i = np.full((N,), False)
        #procesado_j = np.full((M,), False)
        for k in range(N):
            
            posicion_minimo = np.unravel_index(np.argmin(distancias), distancias.shape)
            min_i, min_j = posicion_minimo
            min_d = distancias[min_i, min_j]
            ''
            min_d = np.inf
            for i in range(N):
                if not procesado_i[i]:
                    j=np.argmin(distancias[i,:])
                    if distancias[i,j] < min_d:
                        min_d = distancias[i, j]
                        min_i, min_j = i, j
            ''
            
            lista_loss.append((min_d, min_i, min_j))
            lista_loss_x.append (min_d)
            distancias[:,min_j]=np.inf
            l_min_i.append(min_i)
            
            if (k % N_RECALC) == 0:
                distancias = np.delete(distancias, l_min_i, axis=0)
                l_min_i=[]
                #get_minimos
                
            #procesado_i[min_i]=True

            loss+= min_d
    
    '''
    
            
        
        
    #t20=time.time()
    #tf.print ("time bucle algoritmo:",t20-t10)
    
   
    if tensor:
        loss= loss.numpy()
        
    #tf.print(f"lista_loss orig len:{len(lista_loss)}")
    if debug:
        #tf.print("len lista_loss", len(lista_loss))
        tf.print(">>>>lista_loss>>>>>")
        tf.print([round(x[0], 4) for x in lista_loss])
        #tf.print(">>>>>>>")
    
    
    # get statistics
    if p_stats:
        l_val= list(map (lambda l: l[0],lista_loss))
        tf.print (f"get_1 {msg} lista_pp:d_alb min:{np.min(l_val)}, max:{np.max(l_val)}, media:{np.mean(l_val)}, \n lista_pp:d_alb p90:{np.percentile(l_val,90)}, p75:{np.percentile(l_val,75)}, p25:{np.percentile(l_val,25)}, p10:{np.percentile(l_val,10)}")
    
    lista_loss_orig=lista_loss
    lista_loss=lista_loss_x
        
    #tf.print ("tipo lista_loss:",type(lista_loss))
    if not(inf ==0 and sup == 1):
        pinf = int(len(lista_loss) * inf) if inf >0 else 0
        psup = int(len(lista_loss) * sup) if sup <1 else len(lista_loss)
        tf.print("pinf, psup, len lista_loss", pinf, psup, len(lista_loss))
        #lista_loss.sort(key=lambda x: x[0])
        lista_loss.sort()
        #lista_loss_orig=lista_loss
        lista_loss = lista_loss[pinf:psup]
    #else:
    #    lista_loss=lista_loss_x
    #    lista_loss_orig=lista_loss
    #tf.print ("tipo lista_loss:",type(lista_loss))
  
    
    #loss_norm = np.sum(lista_loss)
    loss_norm_t=tf.reduce_sum(lista_loss)
    loss_norm=loss_norm_t.numpy()
    len_loss_norm_t=len(lista_loss)
    #tf.print ("lens:",len(lista_loss_orig),len_loss_norm_t)
    
    #tf.print(">>> get_1 return loss, loss_norm, loss_norm_t, loss_norm_t / len_loss_norm_t", 
    #              loss, loss_norm, loss_norm_t, loss_norm_t / len_loss_norm_t)
    
    if tensor:
        if debug:
            tf.print(">>> get_1 return loss, loss_norm, loss_norm_t, loss_norm_t / len_loss_norm_t", 
                  loss, loss_norm, loss_norm_t, loss_norm_t / len_loss_norm_t)
        return loss, loss_norm, loss_norm_t, loss_norm_t / len_loss_norm_t
    else:
        #tf.print ("lista_loss vals:",lista_loss)
        if debug:
            tf.print(">>> get_1 return loss, loss_norm", loss, loss_norm)
        return loss, loss_norm


#####

     
    
    
### Distancias con cubitos

FACTOR_ESCALA=1.
FACTOR_RND=0.1

def set_ESCALA (FE,FR=0.1):
    global FACTOR_ESCALA
    global FACTOR_RND
    FACTOR_ESCALA=FE
    FACTOR_RANDOM=FR
    tf.print(f"distancias.py FACTOR_ESCALA:{FACTOR_ESCALA} FACTOR_RANDOM:{FACTOR_RANDOM}")
    

def gen_val (x,FACTOR_BINS):
    #print (type(x),x.shape)
    vint=int(round(x*FACTOR_BINS,0))
    return vint

def get_key (p,fb,rango_x,min_x,scale=False):
    #print ("get_key:",fb,scale)
    #p=(p-med_x)std_x
    
    p=np.reshape(p,(-1,))
    if scale:
        #print (p)
        #print ("min:",min_x)
        #print ("rango_x:",rango_x)
        p_orig=np.copy(p)
        p=(p-min_x)/rango_x
        #print (p)
    
    
    #print ("p.shape",p,p.shape)
    key=""
    for e in p:
        ee=gen_val(e,fb)
        key+=str(ee)+":"
        #if abs(ee) > 30:
        #    tf.print (">>>>> punto",p, "key",key,"ee:",ee)
        #    tf.print ("p_orig",p_orig,"min",min_x, "rango",rango_x)
    '''
    p0=gen_val(p[0],fb)
    p1=gen_val(p[1],fb)
    p2=gen_val(p[2],fb)
    p3=gen_val(p[3],fb)
    key=str(p0)+":"+str(p1)+":"+str(p2)+":"+str(p3)
    '''
    
    return key


#a1 = (np.random.rand(puntos,4)-0.5)*2
#a2 = (np.random.rand(puntos,4)-0.5)*2.1
#a2=a1+0.01


#FACTOR_ESCALA=6



def get_lkeys_distancia_alb (a1,rango_x,min_x,random=False,scale=True):
    scale=True
    #fb= 17.2 if scale else .009
    #fb= 8 if scale else .009
    fb= FACTOR_ESCALA if scale else .009
    #print ("FB:",fb)
    lkeys=[]
    #print ("a1")
    for i in range(a1.shape[0]):
        p=a1[i]
        #print (p.shape)
        if random:
            factor=[((np.random.rand()-0.5)*FACTOR_RND)+1 for i in range(p.shape[0])]
            #l_fact.append(factor)
            p=factor*p
        #print ("i1:",i)
        key=get_key (p,fb,rango_x,min_x,scale=scale)
        lkeys.append(key)
        
    return lkeys


def get_dict_distancia_alb (a1,rango_x,min_x,random=False,scale=True):
    scale=True
    #fb= 17.2 if scale else .009
    #fb= 8 if scale else .009
    fb= FACTOR_ESCALA if scale else .009
    #print ("FB:",fb)
    dict={}
    #print ("a1")
    for i in range(a1.shape[0]):
        p=a1[i]
        #print (p.shape)
        if random:
            factor=[((np.random.rand()-0.5)*FACTOR_RND)+1 for i in range(p.shape[0])]
            #l_fact.append(factor)
            p=factor*p
        #print ("i1:",i)
        key=get_key (p,fb,rango_x,min_x,scale=scale)
        if key in dict.keys():
            dict[key]+=1
        else:
            dict[key]=1
    return dict


def distancia_alb (a1,a2,rango_x,min_x,random=False,scale=True):
    scale=True
    #fb= 17.2 if scale else .009
    #fb= 8 if scale else .009
    fb= FACTOR_ESCALA if scale else .009
    #tf.print (">>>> distancia_alb")
    tf.print ("distancia_alb (cubitos) FB:",fb)
    dict={}
    if (a1.shape[0]!= a2.shape[0]):
        print ("num elems diferente")
        pause()
        return {}
    tf.print ("a1")
    for i in range(a1.shape[0]):
        p=a1[i]
        if random:
            factor=[((np.random.rand()-0.5)*FACTOR_RND)+1 for i in range(p.shape[0])]
            #l_fact.append(factor)
            p=factor*p
        #print ("i1:",i)
        #tf.print ("punto:",p)
        key=get_key (p,fb,rango_x,min_x,scale=scale)
        if key in dict.keys():
            dict[key][0]+=1
            dict[key][2]+=p
        else:
            # #1, #2, sum_coord
            dict[key]=[1,0,np.copy(p)]
            
    tf.print ("a2")
    for i in range(a2.shape[0]):
        #print (p)
        p=a2[i]
        #print (p)
        if random:
            factor=[((np.random.rand()-0.5)*FACTOR_RND)+1 for i in range(p.shape[0])]
            p=factor*p
            #l_fact2.append(factor)
        #print ("p:",p,type(p),type(p[0]))
        key=get_key (p,fb,rango_x,min_x,scale=scale)
        if key in dict.keys():
            dict[key][1]+=1
            dict[key][2]+=p
        else:
            dict[key]=[0,1,np.copy(p)]
    coincide=0
    diffs=0.
    tf.print ("bins:",len(dict.keys()))
    vals=[]
    for k in dict.keys():
        dict[k][2]= dict[k][2]/(dict[k][0]+dict[k][1]) #Calculo centroide 
        vals.append(dict[k][:-1]) # quito la suma de centroid
        diffs+=abs(dict[k][0]-dict[k][1])
        if dict[k][0] >0 and dict[k][1]>0:
            #print (k,dict[k])
            coincide+=1
    tot_bins=len(dict.keys())
    

    #print (dict.keys())
    #tf.print ("DICT:\n",dict)
    
    return diffs,tot_bins,coincide,vals,dict

def _distancia_cubitos (dict,debug=False,centroid_cubito=False):

    tf.print ("distancia_cubitos")
    d_diff=[]
    for k in dict.keys():
        diff=dict[k][0]-dict[k][1]
        if centroid_cubito:
            centroid=np.array([float(i) for i in k.split(":") if i != ""])
        else:
            centroid=dict[k][2]
        #centroid=np.array([float(i) for i in k.split(":") if i != ""])
        #centroid=dict[k][2]
        d_diff.append([centroid,diff,dict[k][0],dict[k][1]])


    #print (d_diff)
    d_diff.sort(key=lambda x: x[1],reverse=True)
    if debug:
        print ("distancia_cubitos inicio. sorted d_diff\n")
        print (d_diff)
           
    #print ("-----")
    distancia=0
    vueltas=0
    asig_min=0
    while d_diff[0][1] >0:
        vueltas+=1
        if debug:
            print ("------ NUEVA RONDA ----")
        minimo=np.inf
        ind_min=-1
        for i in range(1,len(d_diff)):
            dist_eu= np.linalg.norm(d_diff[0][0]- d_diff[i][0])
            if d_diff[i][1] < 0 and dist_eu < minimo:
                asig_min+=1
                minimo=dist_eu
                ind_min=i
                if debug:
                    print ("encontrado minimo dist centroid: distancia_centroides, indice punto",minimo,ind_min)
                    print ("centroides:",d_diff[0][0], d_diff[i][0], "dist",dist_eu)
                    print ("punto base:",d_diff[0])
                    print ("punto compara:",i,d_diff[i])
        # Asignar
        
        nelem= min (-d_diff[ind_min][1],d_diff[0][1])
        #tf.print ("muevo num elems:",nelem)
        d_diff[ind_min][3]-=nelem
        d_diff[ind_min][1]=d_diff[ind_min][2]-d_diff[ind_min][3]
        d_diff[0][3]+=nelem
        d_diff[0][1]=d_diff[0][2]-d_diff[0][3]
        distancia+= nelem*minimo
        if debug:
            print ("asigna: distancia,n_elems,minimo",distancia,nelem,minimo)
        #Ordenar
        #print ("sin orde:\n",d_diff)
        d_diff.sort(key=lambda x: x[1],reverse=True)
        if debug: 
            print ("nueva distancia_cubitos inicio. sorted d_diff\n",d_diff)
    tf.print ("num vueltas:",vueltas,asig_min)
    return distancia

def busca_ind (d_diff):
    ind=-1
    for i,e in enumerate(d_diff):
        if e[1]>0:
            return i
            
    assert (ind != -1)
    return -1

def _distancia_cubitos_s (dict,debug=False,centroid_cubito=False):

    debug=False
    d_diff=[]
    for k in dict.keys():
        diff=dict[k][0]-dict[k][1]
        if centroid_cubito:
            centroid=np.array([float(i) for i in k.split(":") if i != ""])
        else:
            centroid=dict[k][2]
        d_diff.append([centroid,diff,dict[k][0],dict[k][1]])


    #print (d_diff)
    random.shuffle(d_diff)
    #print (d_diff)
    tot_diff= np.sum([abs(e[1]) for e in d_diff])
    if debug:
        print ("tot_diff",tot_diff)
    distancia=0
    tot_num_ifs=0
    num_ifs=0
    num_rondas=0
    while tot_diff>0:
        #t1=time.time()
        if debug:
            print ("------ NUEVA RONDA ----")
        minimo=np.inf
        ind_min=-1
        ind_base=busca_ind(d_diff)
        if debug:
            print ("ini punto base",d_diff[ind_base])
        tot_num_ifs+=num_ifs
        num_ifs=0
        for i in range(0,len(d_diff)):
            #tt1=time.time()
            if i != ind_base:
                #tf.print ("shapes:",d_diff[ind_base][0].shape,d_diff[i][0].shape)
                #dist_eu= np.linalg.norm(d_diff[ind_base][0]- d_diff[i][0])
                diff=d_diff[ind_base][0]- d_diff[i][0]
                dist_eu=np.sqrt(np.sum(diff*diff))
                #if dist_eu != dist_eu1:
                #    tf.print ("ERROR",dist_eu,dist_eu1)
                    
                if d_diff[i][1] < 0 and dist_eu < minimo:
                    num_ifs+=1
                    minimo=dist_eu
                    ind_min=i
                    if debug:
                        print ("encontrado minimo dist centroid: distancia_centroides, indice punto",minimo,ind_min)
                        print ("centroides:",d_diff[ind_base][0], d_diff[i][0], "dist",dist_eu)
                        print ("punto base:",d_diff[ind_base])
                        print ("punto compara:",i,d_diff[i])
            #tt2=time.time()
            #tf.print (f" >> interior {tt2-tt1:.3f}")
        #assert ind_min != -1
        # Asignar
        nelem= min (-d_diff[ind_min][1],d_diff[ind_base][1])
        d_diff[ind_min][3]-=nelem
        d_diff[ind_min][1]=d_diff[ind_min][2]-d_diff[ind_min][3]
        d_diff[ind_base][3]+=nelem
        d_diff[ind_base][1]=d_diff[ind_base][2]-d_diff[ind_base][3]
        distancia+= nelem*minimo
        
        if debug:
            print ("asignacion distancia,nelem,minimo",distancia,nelem,minimo)
        
        random.shuffle(d_diff)
        tot_diff= np.sum([abs(e[1]) for e in d_diff])
        #t2=time.time()
        #tf.print (f"Ronda time: {t2-t1:.3f} ifs:{num_ifs}")
        num_rondas+=1
    tf.print (f"tot ifs:{tot_num_ifs} tot rondas:{num_rondas}")
    return distancia

# #####################
# ----- OBSO 
# #####################

def _get_1_np(X_train, images, M=None, inf=0.0, sup=0.92):

    # assert (X_train.shape[0]==images.shape[0])
    loss, loss_norm = 0, 0  # tf.constant(0.,dtype=tf.float32),0.,tf.constant(0.,dtype=tf.float32)
    # Filtras en la funcion de coste valores muy pequeños. aqui los contabilizamos
    MIN_DIST_EUCL = 0

    # try:
    N = X_train.shape[0]
    K = X_train.shape[-1]
    _images = images  # .numpy()
    if M == None:
        M = int(N * 1.1)
        # print ("N M default:", N, M)
    M = min(M, _images.shape[0])
    # print ("_images0:",_images.shape,type(_images))
    _images = _images[:M]
    # print ("_images1:",_images.shape,type(_images))

    X_train = np.reshape(X_train, (N, K))
    _images = np.reshape(_images, (M, K))
    # print ("_images2:",_images.shape,type(_images))
    # print ("X_train:",X_train.shape)

    distancias = np.full((N, M), np.inf)
    for i in range(N):
        diff = X_train[i] - _images
        # print ("diff.shape,X_train[i].shape,_images.shape",diff.shape,X_train[i].shape,_images.shape)
        dist = np.linalg.norm(diff, axis=1)
        # dist= tf.norm (diff,axis=1)
        # tf.print ("dist.shape",dist.shape,type(dist))
        distancias[i] = dist
        # print ("X_train[i]",X_train[i])
        # print ("images",images)
        # print ("i:",i,"dist:",dist)
    #

    procesado_i = np.full((N,), False)
    procesado_j = np.full((M,), False)
    rango = N  # int(N*(1-restos))
    # loss=tf.constant(0.,dtype=tf.float32)
    lista_loss = []
    # lista_loss1=[]

    for k in range(rango):
        # print ("calculando elem:",k)
        min_d = np.inf
        for i in range(N):
            if not procesado_i[i]:
                for j in range(M):
                    if not procesado_j[j] and distancias[i][j] < min_d:
                        min_d = distancias[i, j]
                        min_i, min_j = i, j

        # assert min_d.numpy() != np.inf
        # print (X_train[min_i],images[min_j],min_d)
        if min_d < np.inf:
            if min_d > MIN_DIST_EUCL:
                lista_loss.append((min_d, min_i, min_j))
                # lista_loss1.append (min_d)
                # min_d_t=tf.math.reduce_euclidean_norm(tf.math.subtract(X_train[min_i],images[min_j]))
                # min_d=np.linalg.norm(X_train[min_i]-images[min_j])
                loss += min_d
            procesado_i[min_i] = True
            procesado_j[min_j] = True
        else:
            print("distancias:", distancias)
            print("X_train:", X_train)
            print("images", images)
            print("_images", _images)
            pause()

    # print ("len lista_loss", len(lista_loss))
    # loss/=len(lista_loss)
    lista_loss.sort(key=lambda x: x[0])

    pinf = int(len(lista_loss) * inf)
    psup = int(len(lista_loss) * sup)
    lista_loss = lista_loss[pinf:psup]
    # print ("pinf, psup, len lista_loss", pinf,psup, len(lista_loss))

    lista_loss = list(map(lambda x: x[0], lista_loss))  # [x[0] for x in lista_loss]
    print(lista_loss)
    print(
        "distancia_alb: (media, mediana, p90):",
        np.mean(lista_loss),
        np.median(lista_loss),
        np.percentile(lista_loss, 90),
    )
    loss_norm = np.sum(lista_loss)

    # print  (">>>",loss,loss_norm)

    return loss, loss_norm

def _get_dist_WK(sample_a, sample_b,sup=1,debug=False,num_samples=None):
    
    if num_samples == None:
        num_samples= min (sample_a.shape[0],sample_b.shape[0])
    diffs,tot_bins,coincide,vals,dict1= distancia_alb(
        sample_a[:num_samples],sample_b[:num_samples],mu.muestras.rango_x,mu.muestras.min_x,random=False,scale=True)
    '''
    for centroid_cub in [True,False]:
        print ("voy a distancia_cubitos")
        t1=time.time()
        distancia= distancia_cubitos (dict1,debug=debug,centroid_cubito=centroid_cub)
        t2=time.time()
        print (f"tiempo distancia_cubitos:{t2-t1:0.3f}")
        print ("voy a distancia_cubitos_stoch")
        t1=time.time()
        l_distancia= [ distancia_cubitos_s (dict1,debug=debug,centroid_cubito=centroid_cub) for i in range(5)]
        t2=time.time()
        print (f"tiempo distancia_cubitos stochs:{(t2-t1)/5:0.3f}")
        #if debug:
        print ("get_dist_WK distancia, l_distancias:",distancia,":", l_distancia)
    
    '''
    tf.print ("voy a distancia_cubitos")
    t1=time.time()
    distancia= distancia_cubitos_s (dict1,debug=debug,centroid_cubito=False)
    t2=time.time()
    tf.print (f"tiempo distancia_cubitos:{t2-t1:0.3f}")
    
    return distancia

def get_1_obso (X_train, images, M=None, inf=0.0, sup=0.95, debug=False, tensor=True,stochastic=True, reverse=False, umbral_dist_alb=np.inf,p_stats=False,msg="",tipo_distancia="normal"):
    
    #debug=False
    
    tf.print ("\nget_1 sup=",sup, "stochastic:",stochastic, "umbral",umbral_dist_alb, "reverse:",reverse, "tipo_distancia:",tipo_distancia)
    assert tipo_distancia != "cuadrados" , "ya no utilizamos distancia cuadrados para los plots"
    
    # assert (X_train.shape[0]==images.shape[0])
    loss, loss_norm, loss_norm_t = 0, 0, 0  # tf.constant(0.,dtype=tf.float32),0.,tf.constant(0.,dtype=tf.float32)
    # Filtras en la funcion de coste valores muy pequeños. aqui los contabilizamos
    MIN_DIST_EUCL = 0
    UMBRAL_DIST=umbral_dist_alb #20. # 0.45
    
    # try:
    N = X_train.shape[0]
    K = X_train.shape[-1]
    #_images = images  # .numpy() 
    
    #Convertir a numpy array para hacer mas rapido calculos de distancia
    if tf.is_tensor(images): 
        _images = images.numpy() 
    else:
        _images= images
    
        
    if M == None:
        M = int(N * 1.1)
        # print ("N M default:", N, M)
    M = min(M, _images.shape[0])
    # tf.print ("_images0:",_images.shape,type(_images))
    _images = _images[:M]
    
    if tensor:
        images = images[:M]
    
    # tf.print ("_images1:",_images.shape,type(_images))

    X_train = np.reshape(X_train, (N, K))
    '''
    if tensor:
        _images = tf.reshape(_images, (M, K))
    else:
        _images = np.reshape(np.array(_images),(N, K))
    '''
    
    #_images = np.reshape(np.array(_images),(N, K))
    _images = np.reshape(_images,(N, K))
    
    #if reverse:
    #    tmp=X_train
    #    X_train=_images
    #    _images=tmp
    assert M == N
    tf.print ("get_1 (M,N,K):", M,N,K)
    
    if debug:
        im=np.round (_images,3)
        xx=np.round (X_train,3)
        tf.print ("X_train mal:\n",xx)
        tf.print ("images gen:\n",im)
        
    #if debug:
    #    tf.print("get_1 X_train:", X_train.shape)
    #    tf.print ("DATOS_1:\n",X_train)
    #    tf.print ("DATOS_2:\n", _images)
        
    #distancias = np.full((N, M), np.inf)
    #distancias2 = np.full((N, M), np.inf)
    #distancias_eu = np.full((N, M), np.inf)
    
    t1=time.time()
    if tipo_distancia == "manhattan":
        tf.print (">>>distancia manhattan")
        assert not tensor , "no puede haber distancia manhattan con tensores"
        for i in range(N):
            # i: X_train
            for j in range(M):
                #j: images
                distancias_eu[i,j]= cityblock(X_train[i],_images[j])
                
        
    else:
        tf.print (">>>distancia normal",tipo_distancia)
        x_tf = tf.keras.layers.InputLayer(dtype=tf.float32, input_shape=X_train.shape[1:])
        x_tf = x_tf(tf.cast(X_train,tf.float32))
        
        y_tf = tf.keras.layers.InputLayer(dtype=tf.float32, input_shape=images.shape[1:])
        y_tf = y_tf(tf.cast(images,tf.float32))
        
        t1=time.time()
        d_fast= dmat (x_tf,y_tf)
        #d_fast= dmat_np (X_train,_images)
        t2=time.time()
        tf.print ("d_fast time:",t2-t1)
        t1=time.time()
        #C = ot.dist(X_train, _images, 'euclidean')
        C= d_fast.numpy()
        t2=time.time()
        tf.print ("C time:",t2-t1)
        
        tf.print ("shapes:",d_fast.shape,C.shape)
        tf.print ("diff mats",np.max(np.absolute(d_fast-C)))
        t2=time.time()
        #tf.print ("d_fast time:",t2-t1)
        #tf.print ("d_fast:",type(d_fast),d_fast.shape,"\n",d_fast)
        distancias_eu=C
        
        
        """
        t1=time.time()
        for i in range(N):
            #diff = np.multiply ((X_train[i] - _images),np.array([1.,1.,1.,1.,1.,0,0]))
            #tf.print(X_train[i].shape, _images.shape)
            #diff = 10*(X_train[i] - _images)

            #tf.print ("diff:",type(diff)) 
            #dist_orig = tf.norm(diff, axis=1)
            #tf.print ("dist_orig:",dist_orig.shape)

            #tf.print ("mult:",mult.shape)

            #tf.print ("dist:",dist.shape)
            '''
            t1=time.time()
            diff_t = tf.math.subtract (X_train[i],images)
            cuad_t=tf.math.square (diff_t)
            dist_t= tf.math.reduce_sum(cuad_t,axis=1)
            dist_t= tf.math.sqrt(dist_t)
            t2=time.time()
            tf.print (f"con tensores:{t2-t1:.5f}")
            tf.print ("dist tensor:",type (dist_t),dist_t.shape,dist_t)

            '''

            #t1=time.time()
            diff = X_train[i] - _images
            cuad= np.square (diff)
            dist= np.sum(cuad,axis=1)
            dist2=dist
            dist=np.sqrt (dist)
            #t2=time.time()
            #tf.print (f"con numpy:{t2-t1:.5f}")
            #tf.print ("dist numpy:",type (dist),dist.shape,dist)

            '''
            tf.print ("------")
            tf.print (dist_orig)
            tf.print ("------")
            tf.print (np.sqrt(dist))
            tf.print ("------")
            tf.print (dist)
            tf.print ("------")
            '''


            #dist2 = tf.math.sigmoid(tf.norm(diff2, axis=1))
            #tf.print ("dist:",i,dist[0],dist[-1])
            #tf.print ("dist2:",i,dist2[0],dist2[-1])
            #tf.print (dist.shape, dist2.shape)
            # tf.print ("dist.shape",dist.shape,type(dist))
            #distancias[i] = dist+dist2
            #distancias[i] = dist
            distancias_eu[i] = dist
            distancias2[i] = dist2
            #if debug:
            #    tf.print ("Distancias:",i,"\n",dist)
            # print ("X_train[i]",X_train[i])
            # print ("images",images)
            # print ("i:",i,"dist:",dist)
    
        """
    #t2=time.time()
    #tf.print ("d_eu time:",t2-t1)
    #tf.print ("d_eu:",type(distancias_eu),distancias_eu.shape,"\n",distancias_eu)
    #difer= np.sum(distancias_eu - d_fast)
    #tf.print ("diferencias:",difer)
    #
    # print (distancias)
    
    
    """
    if not tensor:
        print ("--- distancias -----")
        for i in range(distancias.shape[0]):
            print ("\n i:",i," >>> ")
            for j in range(distancias.shape[1]):
                print (round(distancias[i,j],2),end=" , ") 
        print ("--- distancias -----")
    """
    
    if debug:
        dd=np.round (distancias_eu,4)
        tf.print ("distancias:\n",dd)
        #dd=np.round (distancias2,4)
        #tf.print ("distancias2:\n",dd)
        
    #procesado_i = np.full((N,), False)
    #procesado_j = np.full((M,), False)
    rango = N  # int(N*(1-restos))
    # loss=tf.constant(0.,dtype=tf.float32) 
    lista_loss = []
    lista_loss_x=[]
    loss=0.0
    # lista_loss1=[]
    
    if tipo_distancia == "cuadrados":
        tf.print ("get_1 cuadrados")
        distancias= distancias2
        assert False , "No hay distancias cuadrados"
    else:
        tf.print ("get_1 normal")
        distancias= distancias_eu
        
    t1=time.time()
    for k in range(rango):
        # print ("calculando elem:",k)
        #tf.print ("Ronda:",k)
        min_j=np.argmin(distancias[k,:])
        min_i=k
        #min_d=distancias[i,min_j]
        min_d=tf.math.reduce_euclidean_norm(tf.math.subtract(X_train[min_i],images[min_j]))
        # min_d_t=tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.subtract(X_train[min_i],images[min_j])))
        lista_loss.append((min_d, min_i, min_j))
        lista_loss_x.append (min_d)
        distancias[:,min_j]=np.inf
        loss+=min_d.numpy()
        
    # No schotcastic 
    procesado_i
    for i in range(N):
        if not procesado_i[i]:
            for j in range(M):
                if not reverse:
                    if not procesado_j[j] and distancias[i][j] < min_d:
                        min_d = distancias[i, j]
                        min_i, min_j = i, j
                        if debug:
                            tf.print (f"Encontrado fila:{i}, col:{j} val:{min_d} (not reverse)")
                        #break # como distancia euclidea
                else:
                    if not procesado_j[j] and distancias[j][i] < min_d:
                        min_d = distancias[j, i]
                        min_i, min_j = i, j
                        #break #como distancia euclidea
            
        
    t2=time.time()
    tf.print ("time bucle algoritmo:",t2-t1)
    
    """   
        
        min_d = np.inf
        for i in range(N):
            
            
            
            if stochastic:
                assert min_d == np.inf
            if not procesado_i[i]:
                for j in range(M):
                    if not reverse:
                        if not procesado_j[j] and distancias[i][j] < min_d:
                            min_d = distancias[i, j]
                            min_i, min_j = i, j
                            if debug:
                                tf.print (f"Encontrado fila:{i}, col:{j} val:{min_d} (not reverse)")
                            #break # como distancia euclidea
                    else:
                        if not procesado_j[j] and distancias[j][i] < min_d:
                            min_d = distancias[j, i]
                            min_i, min_j = i, j
                            #break #como distancia euclidea
                if stochastic:
                    break # solo busco para una fila
                    
        assert min_d != np.inf  
        if False:
            tf.print ("Distancias fila ",i)
            tf.print (' '.join([f"{d:.3f}" for d in distancias[i]]))
        
        # DEBUG distancias
        #tf.print (f"distancia:{min_d:.4f}")
        #tf.print ("real:",min_i,X_train[min_i])
        #tf.print ("gen:",min_j,_images[min_j])
        
        # assert min_d.numpy() != np.inf
        # print (X_train[min_i],images[min_j],min_d)
        if min_d < np.inf:
            if min_d >= MIN_DIST_EUCL:
                lista_loss.append((min_d, min_i, min_j))
                if debug:
                    tf.print (f"get_1, dist min:{min_d:.4f}")
                    if not reverse:
                        i=min_i
                        j=min_j
                    else:
                        i=min_j
                        j=min_i
                    tf.print (f"X_train {min_i}:",X_train[i])
                    tf.print (f"images {min_j}:",_images[j])
                    '''
                    diff = X_train[i] - _images[j]
                    cuad= np.square (diff)
                    suma= np.sum(cuad)
                    dist=np.sqrt (suma)
                    tf.print (f"check dist:{dist:.4f}")
                    tf.print ("diff, cuad, suma", diff, cuad, suma)  
                    '''
                    
                #tf.print ("lista_loss elem:", min_d, min_i, min_j)
                #tf.print ("min_d > 0")
                # lista_loss1.append (min_d)
                # min_d_t=tf.math.reduce_euclidean_norm(tf.math.subtract(X_train[min_i],images[min_j]))
                # min_d=np.linalg.norm(X_train[min_i]-images[min_j])
                loss += min_d
                if debug:
                    tf.print ("acumulado loss:",loss)
            procesado_i[min_i] = True
            procesado_j[min_j] = True
            #tf.print ("lista_loss elem:", min_d, min_i, min_j)
        else:
            print("distancias:", distancias)
            print("X_train:", X_train)
            print("images", images)
            print("_images", _images)
            pause()

    
    #if tensor:
    #    tf.print ("lista_loss sort:\n", lista_loss)
    # loss/=len(lista_loss)
    """
    
    tf.print(f"lista_loss orig len:{len(lista_loss)}")
    if debug:
        #tf.print("len lista_loss", len(lista_loss))
        tf.print(">>>>lista_loss>>>>>")
        tf.print([round(x[0], 4) for x in lista_loss])
        #tf.print(">>>>>>>")
    
    
    # get statistics
    if p_stats:
        l_val= list(map (lambda l: l[0],lista_loss))
        tf.print (f"get_1 {msg} lista_pp:d_alb min:{np.min(l_val)}, max:{np.max(l_val)}, media:{np.mean(l_val)}, \n lista_pp:d_alb p90:{np.percentile(l_val,90)}, p75:{np.percentile(l_val,75)}, p25:{np.percentile(l_val,25)}, p10:{np.percentile(l_val,10)}")
        
    
    if not(inf ==0 and sup == 1):
        pinf = int(len(lista_loss) * inf) if inf >0 else 0
        psup = int(len(lista_loss) * sup) if sup <1 else len(lista_loss)
        tf.print("pinf, psup, len lista_loss", pinf, psup, len(lista_loss))
        lista_loss.sort(key=lambda x: x[0])
        lista_loss_orig=lista_loss
        lista_loss = lista_loss[pinf:psup]
    else:
        lista_loss=lista_loss_x
        lista_loss_orig=lista_loss
    """
    if tensor:
        num_umbral=0
        loss_norm_t = tf.constant(0.0, dtype=tf.float32)  # 0.
        len_loss_norm_t = len(lista_loss)
        if tipo_distancia == "cuadrados":
            f_min_d_t = lambda x: tf.math.reduce_sum(tf.math.square(x))
        else:
            f_min_d_t = lambda x: tf.math.sqrt(tf.math.reduce_sum(tf.math.square(x)))
                
        for index, l1 in enumerate(lista_loss):
            #min_d_t = tf.math.reduce_euclidean_norm(tf.math.subtract(X_train[l1[1]], images[l1[2]]))
            #tf.print ("i,j, dist",l1[1],l1[2],l1[0])
            if l1[0] < UMBRAL_DIST:
                if not reverse:
                    #diff=10*tf.math.subtract(X_train[l1[1]], images[l1[2]])
                    diff=tf.math.subtract(X_train[l1[1]], images[l1[2]])
                    #tf.print ("images[l1[2]]",type(images[l1[2]]))
                else:
                    #diff=10*tf.math.subtract(X_train[l1[2]], images[l1[1]])
                    diff=tf.math.subtract(X_train[l1[2]], images[l1[1]])
                    #tf.print ("images[l1[1]]",type(images[l1[2]]))
                #tf.print ("diff:",type(diff),diff.shape)     
            else:
                # Mucha distancia, coge el random
                num_umbral+=1
                if not reverse:
                    #diff=tf.math.subtract(X_train[l1[2]], images[l1[2]])
                    # i es 1, y es el Generado
                    #diff=tf.math.subtract(X_train[l1[1]], images[l1[1]])
                    diff= images[l1[1]] - X_train[l1[1]] #adversarial_examples - malign_inputs
                    #tf.print ("images[l1[1]].shape",images[l1[1]].shape,images[l1[1]])
                    #tf.print ("X_train[l1[1]].shape",X_train[l1[1]].shape,X_train[l1[1]])
                else:
                    #diff=tf.math.subtract(X_train[l1[1]], images[l1[1]])
                    #diff=tf.math.subtract(X_train[l1[2]], images[l1[2]])
                    diff= images[l1[2]] - X_train[l1[2]]
                
            #min_d_t = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(diff)))
            min_d_t= f_min_d_t(diff)
            
            #tf.print ("diffs:\n",diff)
            #tf.print ("min1:",min_d_t,type(tf.math.square(diff)))
            #min_d_t= tf.sqrt(tf.reduce_sum(tf.square(diff), axis=1))
            #tf.print ("min2:",min_d_t,type(tf.square(diff,axis=1)))
            
            #tf.print ("min_d_t tensor:",type(min_d_t))
            #min_d_t=l1[0] No es tensor
            #tf.print ("min_d_t",min_d_t,type(min_d_t))
            loss_norm_t = loss_norm_t + min_d_t
            
        tf.print ("tensor num_umbral:",num_umbral)

            #if index < 100:
            #    tf.print("malign: ", X_train[l1[1]], "gen: ", images[l1[2]], "dist (malign, gen): ", min_d_t)
    
    #    map(lambda x: tf.math.reduce_euclidean_norm(tf.math.subtract(X_train[x[1]],images[x[2]])) ,lista_loss))) / len(lista_loss)
    lista_loss = list(map(lambda x: x[0], lista_loss))  # [x[0] for x in lista_loss]
    tf.print (f"lista_loss: len:{len(lista_loss)}, suma:{np.sum(lista_loss)}, max:{np.max(lista_loss)}, min:{np.min(lista_loss)}, mean:{np.mean(lista_loss)}, std:{np.std(lista_loss)}")
    
    out = np.sum(list(map(lambda x: 1 if x> UMBRAL_DIST else 0, lista_loss)))
    tf.print (f"lista_loss fuera del umbral({UMBRAL_DIST}): {out}")
    """
    
    #loss_norm = np.sum(lista_loss)
    loss_norm_t=tf.reduce_sum(lista_loss)
    loss_norm=loss_norm_t.numpy()
    len_loss_norm_t=len(lista_loss)
    tf.print ("lens:",len(lista_loss_orig),len_loss_norm_t)
    
    tf.print(">>> get_1 return loss, loss_norm, loss_norm_t, loss_norm_t / len_loss_norm_t", 
                  loss, loss_norm, loss_norm_t, loss_norm_t / len_loss_norm_t)
    
    if tensor:
        if debug:
            tf.print(">>> get_1 return loss, loss_norm, loss_norm_t, loss_norm_t / len_loss_norm_t", 
                  loss, loss_norm, loss_norm_t, loss_norm_t / len_loss_norm_t)
        return loss, loss_norm, loss_norm_t, loss_norm_t / len_loss_norm_t
    else:
        #tf.print ("lista_loss vals:",lista_loss)
        if debug:
            tf.print(">>> get_1 return loss, loss_norm", loss, loss_norm)
        return loss, loss_norm

