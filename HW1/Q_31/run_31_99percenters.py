import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
import numpy as np
from sklearn.model_selection import train_test_split
import time

df = pd.read_csv('OMML2020_Assignment_1_Dataset.csv')


np.random.seed(1908315)
Y = df['Y'].to_numpy()
b = np.ones(df.shape[0])
df['b'] = b
X_MLP = df[['b', 'X_1','X_2']].to_numpy()
X_tot, X_test, y_tot, y_test = train_test_split(X_MLP, Y, test_size = 0.15)

# define train and test
X_tot_ = np.transpose(X_tot)
X_test_ = np.transpose(X_test)
y_test_ = y_test.reshape(1,len(y_test))
y_tot_ = y_tot.reshape(1,len(y_tot))

# size 
input_size = 2 + 1
output_size = 1




#####################################################################
##################### cambiare valori in base a 11 #######################
#####################################################################


N = 40
sigma = 0.5
rho = 0.00001
epochs = 5


# random weights
W1 = np.random.randn(input_size, N)
V = np.random.randn(N,output_size)
par = W1.shape
omega_MLP = np.concatenate((W1, V), axis = None)

start = time.time()

for ep in epochs:
    # optimize V
    omega_MLP_ = minimize(MLP, x0 = omega_MLP, args = (X_tot_, sigma, par, rho, y_tot_),jac=JAC_MLP_v, method = 'L-BFGS-B', tol = 1e-9, options = {'maxiter': 100, 'disp': False})
    # optimize W1
    omega_MLP_1 = minimize(MLP, x0 = omega_MLP_, args = (X_tot_, sigma, par, rho, y_tot_),jac=JAC_MLP_W, method = 'L-BFGS-B', tol = 1e-9, options = {'maxiter': 100, 'disp': False})
    
    
    #####################################################################
    ##################### implementare earlystopping con validation  #######################
    #####################################################################
    if np.linalg.norm(omega_MLP - omega_MLP_1)**2 < 1e-6:
        break

delta_t = time.time() - start





# evaluate error
e = MLP_test(omega_MLP.x, X_tot_, sigma, par, rho, y_tot_)
e_val = MLP_test(omega_MLP.x, X_test_, sigma, par, rho, y_test_)


what = ['N','sigma','rho', 'tollerance', 'max_numb_iter', 'optimz_solver', 'nfev', 'niter', 'time', 'train_error', 'test_error' ]
values_ = [N, sigma, rho, 1e-9, 100, 'L-BFGS-B', omega_MLP.nfev, omega_MLP.nit, delta_t,  e, e_val]



#####################################################################
##################### print in maniera decente #######################
#####################################################################



# plot figure
# Plot_MLP(omega_MLP.x, sigma, (3,N), rho, 100)