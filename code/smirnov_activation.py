from scipy import interpolate
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp

class SmirnovActivation:
    def __init__(self, input_dim,puntos_spline=2000):
        # Set "cols" (number of features) value
        self.cols=input_dim[-1]

        print({self.cols})

        # Set "seq" value
        self.seq = 1 # Always to one, don't touch!

        # Set spline points, we set it to 500 so it doesn't take too long to compute the transform
        self.puntos_spline = puntos_spline #2000

        self.custom_fs = []
    
    def create(self, XX_train):
        for i in range(self.cols):
            original_sample = XX_train[:,i].reshape((-1,1))

            print ("Generando Spline: ", i, XX_train.shape, original_sample.shape)

            NN_output_fun, NN_output_fun_der = self.create_NN_output_function(original_sample, puntos_spline=self.puntos_spline) 
            self.custom_fs.append(NN_output_fun)

    def convert_to_uniform(self, x, dist, **params):
        if dist == 'normal':
            return stats.norm.cdf(x, **params)
        
        if dist == 'exponential':
            return stats.expon.cdf(x, **params)
        
    def convert_from_uniform(self, x, dist, **params):
        if dist == 'normal':
            return stats.norm.ppf(x, **params)
        
        if dist == 'exponential':
            return stats.expon.ppf(x, **params)
        
    def smirnov_transform(self, x, dist_orig, dist_dest, params_orig, params_dest):
        y = self.convert_to_uniform(x, dist_orig, **params_orig)
        
        return self.convert_from_uniform(y, dist_dest, **params_dest)

    def smirnov_transform_normal(self, x, dist, **params):
        params_dest = {'loc': 0, 'scale': 1}
        
        return self.smirnov_transform(self, x, dist_orig = dist, dist_dest = 'normal', params_orig = params, params_dest = params_dest)

    def smirnov_transform_normal_inv(x, dist, **params):
        params_orig = {'loc': 0, 'scale': 1}
        
        return self.smirnov_transform(self, x, dist_orig = 'normal', dist_dest = dist, params_orig = params_orig, params_dest = params)

    def ecdf(self, x):
        x = np.sort(x)
        n = len(x)
        
        def _ecdf(v):
            # side='right' because we want Pr(x <= v)
            return (np.searchsorted(x, v, side='right') + 1) / n
        
        return _ecdf

    def clip_ecdf(self, a):
        if np.isinf(a) or np.isnan(a):
            return 1
        else:
            return a
        
    def auto_smirnov_transform_normal(self, x):
        #y = ecdf(x)(x)
        clip_ecdf_vec = np.vectorize(clip_ecdf)
        y = ecdf(x)(x)
        
        return self.clip_ecdf_vec(convert_from_uniform(y, dist='normal', loc=0, scale=1))

    def auto_smirnov_transform_normal_inv(self, x, original):
        y = self.convert_to_uniform(x, dist='normal', loc=0, scale=1)
        
        def _inv_ecdf(a):
            return np.quantile(original, q=a)
        
        return _inv_ecdf(y)
        
    def create_NN_output_function(self, original,puntos_spline=2000):
        f = lambda x: self.auto_smirnov_transform_normal_inv(x, original)
        min_value = -100
        min_fun = f(min_value)
        
        while f(min_value + 1) < min_fun + 0.0001:
            min_value += 1
            
        max_value = 100
        max_fun = f(max_value)
        
        while f(max_value - 1) > max_fun - 0.0001:
            max_value -= 1
            
        print ("min_val:",min_value,"max_val:",max_value)
        
        x = np.linspace(start=min_value, stop=max_value, num=puntos_spline)
        values = [f(x0).astype(np.float32) for x0 in x]

        def _NN_output_function(x):
            return tfp.math.interp_regular_1d_grid(x=x, x_ref_min=min_value, x_ref_max=max_value, y_ref=values)

        def _NN_output_function_derivative(x):
            valora = x
            return 1
            return interpolate.spalde(x.eval(session=tf.compat.v1.Session()), tck)
        
        return _NN_output_function, _NN_output_function_derivative