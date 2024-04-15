# Importações
import torch

# Configuração do CUDA
cuda = torch.device('cuda')

''' ATENÇÃO: Usando tensors do PyTorch com CUDA na GPU. '''

# *** Usando torch (GPU) ***

# Scalar
t0d = torch.tensor(0).cuda()
print(t0d.shape)

# Vetor (Tensor de 1 dimensão, tensor 1D)
t1d = torch.tensor([18, 27, 45, 23, 37]).cuda()
print(t1d.shape)

# Matriz (Tensor de 2 dimensões, tensor 2D)
t2d = torch.tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]).cuda()
print(t2d.shape)

# Cubo (Tensor de  3 dimensões, tensor 3D)
t3d = torch.tensor([[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]],
                    [[2, 3, 4],
                    [5, 6, 7],
                    [8, 9, 10]]]).cuda()
print(t3d.shape)
