# Importações
import numpy as np

''' ATENÇÃO: Usando numpy, todas as operações serão feitas na CPU
    Futuramente, iremos ver como fazer as mesmas operações usando tensors do
    PyTorch na GPU. '''

# *** Usando numpy (CPU) ***

# Scalar (Tensor de 0 dimensões, tensor 0D)
t0d = np.array(20)
print(t0d.shape)

# Vetor (Tensor de 1 dimensão, tensor 1D)
t1d = np.array([18, 27, 45, 23, 37])
print(t1d.shape)

# Matriz (Tensor de 2 dimensões, tensor 2D)
t2d = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(t2d.shape)

# Cubo (Tensor de  3 dimensões, tensor 3D)
t3d = np.array([[[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]],
                [[2, 3, 4],
                 [5, 6, 7],
                 [8, 9, 10]]])
print(t3d.shape)
