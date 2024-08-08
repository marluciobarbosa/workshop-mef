# O problema

A equação que está sendo resolvida é a equação de Poisson em duas dimensões, com condições de contorno de Dirichlet. 

**Equação de Poisson:**

```math
-\nabla^2 u(x,y) = f(x,y) \quad \text{em} \quad \Omega = (0,1) \times (0,1)
```

onde:

$u(x,y)$ é a função desconhecida que estamos resolvendo
$\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$ é o operador Laplaciano
$f(x,y)$ é a função de fonte (no nosso caso, $f(x,y) =  2 * \pi^2 * \operatorname{sen}(\pi * x) * \operatorname{sen}(\pi * y)$)

**Condições de contorno de Dirichlet:**

```math
u(x,y) = 0 \quad \text{em} \quad \partial\Omega
```

onde $\partial\Omega$ representa a fronteira do domínio, que neste caso são os quatro lados do quadrado unitário:

```math
\begin{align*}
u(0,y) &= 0 \quad \text{para} \quad 0 \leq y \leq 1 \\
u(1,y) &= 0 \quad \text{para} \quad 0 \leq y \leq 1 \\
u(x,0) &= 0 \quad \text{para} \quad 0 \leq x \leq 1 \\
u(x,1) &= 0 \quad \text{para} \quad 0 \leq x \leq 1
\end{align*}
```

Esta equação, com estas condições de contorno, descreve um problema de valor de fronteira em que estamos buscando uma função $u(x,y)$ que satisfaça a equação de Poisson no interior do domínio quadrado unitário, e que seja zero em todas as fronteiras deste quadrado.

**Solução Analítica:**

Um possível solução analítica  é  dada por:

```math
u(x,y) = sen(\pi*x) * sen(\pi*y))
```

# O ambiente

## Requisitos

O código foi testado utilizando o sistema operacional Ubuntu 24.04 LTS.

As seguintes dependências devem ser instaladas:

**NVIDIA HPC SDK**

```bash
sudo apt update
sudo apt install -y libtbb-dev libeigen3-dev curl wget git cmake
curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list
sudo apt-get update -y
sudo apt-get install -y nvhpc-24-7 cuda-toolkit-12-6
```

**NVIDIA CUDA**

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

*NOTA:* Instale o NVIDIA Driver apropriado.

**Atualizando o arquivo `.bashrc`**

```bash
echo '# NVIDIA SDK' >> ~/.bashrc
echo 'NVARCH=`uname -s`_`uname -m`; export NVARCH' >> ~/.bashrc
echo 'NVCOMPILERS=/opt/nvidia/hpc_sdk; export NVCOMPILERS' >> ~/.bashrc
echo 'MANPATH=$MANPATH:$NVCOMPILERS/$NVARCH/24.7/compilers/man; export MANPATH' >> ~/.bashrc
echo 'PATH=$NVCOMPILERS/$NVARCH/24.7/compilers/bin:$PATH; export PATH' >> ~/.bashrc
echo 'export PATH=$NVCOMPILERS/$NVARCH/24.7/comm_libs/mpi/bin:$PATH #MPI' >> ~/.bashrc
echo 'export MANPATH=$MANPATH:$NVCOMPILERS/$NVARCH/24.7/comm_libs/mpi/man #MPI' >> ~/.bashrc
echo '' >> ~/.bashrc
echo '# NVIDIA CUDA' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

## Compilando

Para compilar execute o comando a seguir, ajustando-o para o seu ambiente.

```bash
cmake -B CAMINHO_DIRETORIO_DE_BUILD \
        -DCMAKE_CXX_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/bin/nvc++ \
        -DCMAKE_C_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/bin/nvc \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_KOKKOS=ON\
        -DUSE_CUDA=ON \
        -DUSE_OPENMP=ON \
        -DUSE_OPENACC=ON \
        -S CAMINHO_DIRETORIO_RAIZ_DO_PROJETO
```
Substitua `CAMINHO_DIRETORIO_DE_BUILD` e `CAMINHO_DIRETORIO_RAIZ_DO_PROJETO` de acordo com os endereços absolutos dos seus diretórios.

Os binários estarão disponíveis em: `CAMINHO_DIRETORIO_DE_BUILD`/bin.

# Finalidade

Os códigos e instruções nesse repositório são parte de um material de apoio para  instrução formal tutelada. __Não utilize esse código em ambiente de produção sem os ajustes necessários__.

O **processo de refinamento** dos códigos e suas nuances são discutidos e trabalhados em **ambiente acadêmico**.