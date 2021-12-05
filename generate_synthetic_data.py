import scipy.io as sio
import scipy.linalg as spalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math

def sigmoid(x):
    return 1./(1. + np.exp(-x))

seed = 7
np.random.seed(seed)

# Mixing matrices of the two views
A1 = np.random.randn(5,5)
A2 = np.random.randn(5,5)

# Number of samples
L = 2000

# View-specific components
rand1 = 1.0*np.random.randn(L, 3)-0.5
rand2 = 1.5*np.random.randn(L, 3)+0.8

# Generate the shared components
rand_var = np.random.uniform(-1.0, 1.0, (L, 1) )
shared = np.hstack([rand_var, rand_var**2])

shared = shared-np.mean(shared,0)

# Orthorgonal basis of the shared component, used for caomputing subspace distance
Q, _ = np.linalg.qr(shared)

# Shared component concatenated with the view-specific components
CD1=np.hstack([shared,rand1])
CD2=np.hstack([shared,rand2])

# Do the mixing, scale some channels for better visualizations
mix1 = CD1.dot(A1.T)*np.array([1,2,1,2,1])
mix2 = CD2.dot(A2.T)

# Data matrices of the two views
v1 = np.zeros_like(mix1)
v2 = np.zeros_like(mix2)

# The first view, with 5 different nonlinear functions
v1[:,0] = 3*sigmoid(mix1[:,0])+0.1*mix1[:,0]
plt.scatter(mix1[:,0], v1[:,0])
plt.show()

v1[:,1] = 5*sigmoid(mix1[:,1])+0.2*mix1[:,1]
plt.scatter(mix1[:,1], v1[:,1])
plt.show()

v1[:,2] = 0.2*np.exp(mix1[:,2])
plt.scatter(mix1[:,2], v1[:,2])
plt.show()

v1[:,3] = -4*sigmoid(mix1[:,3])-0.3*mix1[:,3]
plt.scatter(mix1[:,3], v1[:,3])
plt.show()

v1[:,4] = -3*sigmoid(mix1[:,4])-0.2*mix1[:,4]
plt.scatter(mix1[:,4], v1[:,4])
plt.show()

# The second view, with another 5 different nonlinear functions
v2[:,0] = 5*np.tanh(mix2[:,0])+0.2*mix2[:,0]
plt.scatter(mix2[:,0], v2[:,0])
plt.show()

v2[:,1] = 2*np.tanh(mix2[:,1])+0.1*mix2[:,1]
plt.scatter(mix2[:,1], v2[:,1])
plt.show()

v2[:,2] = 0.1*(mix2[:,2])**3+mix2[:,2]
plt.scatter(mix2[:,2], v2[:,2])
plt.show()

v2[:,3] = -5*np.tanh(mix2[:,3])-0.4*mix2[:,3]
plt.scatter(mix2[:,3], v2[:,3])
plt.show()

v2[:,4] = -6*np.tanh(mix2[:,4])-0.3*mix2[:,4]
plt.scatter(mix2[:,4], v2[:,4])
plt.show()


# Plot the shared components
plt.scatter(shared[:,0], shared[:,1], s=300)
plt.xlabel('$s_{1,\ell}$',fontsize=40)
plt.ylabel('$s_{2,\ell}$',fontsize=40)
plt.show()

# Save the data as .mat format
sio.savemat('synthetic_data_2view.mat',{'view1':v1, 'view2':v2, 'shared':shared,
                'mix1':mix1, 'mix2':mix2, 'Q':Q})
