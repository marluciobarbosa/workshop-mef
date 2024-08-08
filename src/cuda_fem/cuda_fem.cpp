#include <Eigen/Dense>     // Inclui a biblioteca Eigen para operações com matrizes densas.
#include <Eigen/Sparse>    // Inclui a biblioteca Eigen para operações com matrizes esparsas.
#include <thrust/host_vector.h>  // Inclui a biblioteca Thrust para vetores em CPU.
#include <thrust/device_vector.h>// Inclui a biblioteca Thrust para vetores em GPU.
#include <cassert>         // Inclui a biblioteca assert para verificações durante o desenvolvimento.
#include <chrono>          // Inclui a biblioteca chrono para medição de tempo.
#include <cmath>           // Inclui a biblioteca cmath para funções matemáticas padrão.
#include <fstream>         // Inclui a biblioteca fstream para operações de leitura e escrita em arquivos.
#include <iomanip>         // Inclui a biblioteca iomanip para manipulação de formatação de saída.
#include <iostream>        // Inclui a biblioteca iostream para operações de entrada e saída padrão.

// Kernel CUDA para aplicar condições de contorno na matriz de rigidez e vetor de cargas
__global__ void aplicarCondicoesContornoKernel(double* d_K, double* d_f, int* d_contorno, int contornoSize, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calcula o índice global do thread.
    if (idx < contornoSize) { // Verifica se o índice está dentro do tamanho do vetor de contorno.
        int bIdx = d_contorno[idx]; // Obtém o índice do nó de contorno.
        for (int j = 0; j < N; ++j) {
            d_K[bIdx * N + j] = 0.0; // Zera a linha correspondente na matriz de rigidez.
        }
        d_K[bIdx * N + bIdx] = 1.0; // Define 1 na diagonal para garantir solução única.
        d_f[bIdx] = 0.0; // Define o vetor de forças para 0 nesse nó (condição de Dirichlet nula).
    }
}

// Kernel CUDA para construir a matriz de rigidez e o vetor de carga
__global__ void construirMatrizEVetorKernel(
    int* elementos, double* nos, double* d_K, double* d_f, 
    int numElementos, int N, double pesoDeGauss, double* pontoDeGauss) {

    int e = blockIdx.x * blockDim.x + threadIdx.x; // Calcula o índice global do thread.
    if (e < numElementos) { // Verifica se o índice está dentro do número de elementos.
        int elemIdx = e * 3; // Ajuste para acessar corretamente o elemento.
        int elemento[3]; // Array para armazenar os índices dos nós do elemento atual.
        elemento[0] = elementos[elemIdx + 0];
        elemento[1] = elementos[elemIdx + 1];
        elemento[2] = elementos[elemIdx + 2];

        double coords_e[3][2]; // Array para armazenar as coordenadas dos nós do elemento atual.
        for (int i = 0; i < 3; ++i) {
            coords_e[i][0] = nos[elemento[i] * 2 + 0];
            coords_e[i][1] = nos[elemento[i] * 2 + 1];
        }

        double dN_dxi[2][3] = {{-1, 1, 0}, {-1, 0, 1}}; // Derivadas das funções de forma nas coordenadas locais.
        double J[2][2] = {{0, 0}, {0, 0}}; // Matriz Jacobiana do elemento.

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 3; ++k) {
                    J[i][j] += dN_dxi[i][k] * coords_e[k][j]; // Cálculo da matriz Jacobiana.
                }
            }
        }

        double detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0]; // Determinante da matriz Jacobiana.
        if (fabs(detJ) < 1e-10) return; // Verifica se o elemento é degenerado (detJ próximo de zero).

        double invJ[2][2] = {
            { J[1][1] / detJ, -J[0][1] / detJ },
            {-J[1][0] / detJ,  J[0][0] / detJ }
        }; // Matriz inversa de J.

        double dN_dxy[2][3] = {{0, 0, 0}, {0, 0, 0}}; // Derivadas das funções de forma nas coordenadas globais.
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 2; ++k) {
                    dN_dxy[i][j] += invJ[i][k] * dN_dxi[k][j]; // Transformação das derivadas para coordenadas globais.
                }
            }
        }

        double N_e[3] = {1.0 - pontoDeGauss[0] - pontoDeGauss[1], pontoDeGauss[0], pontoDeGauss[1]}; // Funções de forma no ponto de Gauss.

        double K_e[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}; // Matriz de rigidez local do elemento.
        double F_e[3] = {0, 0, 0}; // Vetor de forças local do elemento.

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 2; ++k) {
                    K_e[i][j] += dN_dxy[k][i] * dN_dxy[k][j] * detJ * pesoDeGauss; // Cálculo da matriz de rigidez local usando integração numérica.
                }
            }
        }

        for (int i = 0; i < 3; ++i) {
            double x = 0.0, y = 0.0; // Inicializa as coordenadas para o cálculo do ponto de integração.
            for (int j = 0; j < 3; ++j) {
                x += N_e[j] * coords_e[j][0]; // Calcula a coordenada x do ponto de integração.
                y += N_e[j] * coords_e[j][1]; // Calcula a coordenada y do ponto de integração.
            }
            F_e[i] = N_e[i] * 2 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y) * detJ * pesoDeGauss; // Cálculo do vetor de forças local.
        }

        for (int i = 0; i < 3; ++i) {
            atomicAdd(&d_f[elemento[i]], F_e[i]); // Soma o vetor de forças local ao vetor de forças global usando operação atômica.
            for (int j = 0; j < 3; ++j) {
                atomicAdd(&d_K[elemento[i] * N + elemento[j]], K_e[i][j]); // Soma a matriz de rigidez local à matriz de rigidez global usando operação atômica.
            }
        }
    }
}

class FEMSolver2D {
public:
    FEMSolver2D(int nx, int ny); // Construtor que inicializa o solucionador com número de nós nx e ny.
    void construir_malha(); // Método para construir a malha de elementos finitos.
    void construir_matriz_e_vetor(); // Método para construir a matriz de rigidez e o vetor de carga.
    void aplicar_condicoes_contorno(); // Método para aplicar condições de contorno Dirichlet.
    void resolver(); // Método para resolver o sistema linear resultante.
    void imprimir_resultados() const; // Método para imprimir os resultados na saída padrão.
    void salvar_resultados(const std::string &filename) const; // Método para salvar os resultados em um arquivo.

private:
    int nx, ny;                             // Número de nós na direção x e y.
    int N;                                  // Total de nós na malha (nx * ny).
    thrust::host_vector<double> K;          // Matriz de rigidez (densa) na CPU.
    thrust::host_vector<double> f;          // Vetor de cargas na CPU.
    thrust::host_vector<double> u;          // Vetor solução na CPU.
    thrust::host_vector<int> contorno;      // Vetor para armazenar os índices das condições de contorno.
    thrust::host_vector<int> elementos;     // Vetor para armazenar os elementos.
    thrust::host_vector<double> nos;        // Vetor para armazenar as coordenadas dos nós.
    double hx, hy;                          // Tamanho dos elementos em x e y.

    // Vetores para armazenar dados na GPU.
    thrust::device_vector<double> d_K;
    thrust::device_vector<double> d_f;
    thrust::device_vector<double> d_nos;
    thrust::device_vector<int> d_elementos;
    thrust::device_vector<int> d_contorno;

    double fonte(double x, double y); // Função que define o termo fonte.
    void resolver_sistema(); // Método para resolver o sistema linear usando a biblioteca Eigen.
};

FEMSolver2D::FEMSolver2D(int nx, int ny)
    : nx(nx), ny(ny), N((nx + 1) * (ny + 1)), K(N * N, 0.0),
      f(N, 0.0), u(N, 0.0), nos((nx + 1) * (ny + 1) * 2, 0.0), elementos(2 * nx * ny * 3), d_K(N * N, 0.0), d_f(N, 0.0) {
    hx = 1.0 / nx; // Calcula o tamanho dos elementos na direção x.
    hy = 1.0 / ny; // Calcula o tamanho dos elementos na direção y.
    
    auto start = std::chrono::high_resolution_clock::now(); // Marca o tempo inicial para a construção da malha.
    construir_malha(); // Chama o método para construir a malha.
    auto end = std::chrono::high_resolution_clock::now(); // Marca o tempo final.
    std::chrono::duration<double, std::milli> diff = end - start; // Calcula a duração em milissegundos.
    std::cout << "Tempo de construção da malha: " << std::scientific << diff.count() << " ms\n"; // Imprime o tempo gasto.
}

void FEMSolver2D::construir_malha() {
    // Geração dos nós
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            int idx = j * (nx + 1) + i; // Calcula o índice do nó atual.
            nos[idx * 2 + 0] = i * hx; // Define a coordenada x do nó.
            nos[idx * 2 + 1] = j * hy; // Define a coordenada y do nó.
        }
    }

    d_nos = nos; // Copia os dados dos nós para a GPU.

    // Geração dos elementos
    int index = 0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int n1 = j * (nx + 1) + i; // Nó inferior esquerdo do quadrado.
            int n2 = n1 + 1; // Nó inferior direito do quadrado.
            int n3 = n1 + (nx + 1); // Nó superior esquerdo do quadrado.
            int n4 = n3 + 1; // Nó superior direito do quadrado.

            elementos[index++] = n1; // Primeiro triângulo: inferior esquerdo.
            elementos[index++] = n3; // Primeiro triângulo: superior esquerdo.
            elementos[index++] = n2; // Primeiro triângulo: inferior direito.

            elementos[index++] = n2; // Segundo triângulo: inferior direito.
            elementos[index++] = n3; // Segundo triângulo: superior esquerdo.
            elementos[index++] = n4; // Segundo triângulo: superior direito.
        }
    }

    d_elementos = elementos; // Copia os dados dos elementos para a GPU.
}

void FEMSolver2D::construir_matriz_e_vetor() {
    double pesoDeGauss = 0.5; // Peso associado ao ponto de Gauss.
    thrust::device_vector<double> d_pontoDeGauss = {1.0 / 3, 1.0 / 3}; // Ponto de Gauss no centro do triângulo.

    int blockSize = 256; // Define o tamanho do bloco de threads.
    int gridSize = (elementos.size() / 3 + blockSize - 1) / blockSize; // Calcula o número de blocos necessários.

    construirMatrizEVetorKernel<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_elementos.data()), // Ponteiro para os elementos na GPU.
        thrust::raw_pointer_cast(d_nos.data()), // Ponteiro para os nós na GPU.
        thrust::raw_pointer_cast(d_K.data()), // Ponteiro para a matriz de rigidez na GPU.
        thrust::raw_pointer_cast(d_f.data()), // Ponteiro para o vetor de forças na GPU.
        elementos.size() / 3, N, pesoDeGauss, // Número de elementos, número total de nós e peso de Gauss.
        thrust::raw_pointer_cast(d_pontoDeGauss.data())); // Ponteiro para o ponto de Gauss na GPU.

    cudaDeviceSynchronize(); // Sincroniza a GPU com a CPU para garantir que todos os cálculos estejam concluídos.
}

void FEMSolver2D::aplicar_condicoes_contorno() {
    for (int i = 0; i <= nx; ++i) {
        contorno.push_back(i);                 // Borda inferior.
        contorno.push_back(ny * (nx + 1) + i); // Borda superior.
    }
    for (int j = 1; j < ny; ++j) {
        contorno.push_back(j * (nx + 1));      // Borda esquerda.
        contorno.push_back(j * (nx + 1) + nx); // Borda direita.
    }

    d_contorno = contorno; // Copia os índices de contorno para a GPU.

    int blockSize = 256; // Define o tamanho do bloco de threads.
    int gridSize = (contorno.size() + blockSize - 1) / blockSize; // Calcula o número de blocos necessários.

    aplicarCondicoesContornoKernel<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_K.data()), // Ponteiro para a matriz de rigidez na GPU.
        thrust::raw_pointer_cast(d_f.data()), // Ponteiro para o vetor de forças na GPU.
        thrust::raw_pointer_cast(d_contorno.data()), // Ponteiro para o vetor de contorno na GPU.
        contorno.size(), N); // Tamanho do vetor de contorno e número total de nós.

    cudaDeviceSynchronize(); // Sincroniza a GPU com a CPU para garantir que todos os cálculos estejam concluídos.
}

double FEMSolver2D::fonte(double x, double y) {
    return 2 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y); // Função que representa o termo fonte.
}

void FEMSolver2D::resolver() {
    thrust::copy(d_K.begin(), d_K.end(), K.begin()); // Copia a matriz de rigidez da GPU para a CPU.
    thrust::copy(d_f.begin(), d_f.end(), f.begin()); // Copia o vetor de forças da GPU para a CPU.
    resolver_sistema(); // Chama o método para resolver o sistema linear.
}

void FEMSolver2D::resolver_sistema() {
    Eigen::SparseMatrix<double> K_eigen(N, N); // Criação de uma matriz esparsa Eigen para armazenar a matriz de rigidez.
    Eigen::VectorXd f_eigen(N); // Criação de um vetor Eigen para armazenar o vetor de forças.
    Eigen::VectorXd u_eigen(N); // Criação de um vetor Eigen para armazenar o vetor solução.

    // Transferir a matriz K e o vetor f para Eigen
    std::vector<Eigen::Triplet<double>> tripletList; // Lista de triplas para construir a matriz esparsa.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (K[i * N + j] != 0) {
                tripletList.push_back(Eigen::Triplet<double>(i, j, K[i * N + j])); // Armazena apenas os valores não nulos.
            }
        }
        f_eigen(i) = f[i]; // Copia o vetor de forças para o vetor Eigen.
    }
    K_eigen.setFromTriplets(tripletList.begin(), tripletList.end()); // Constrói a matriz esparsa a partir das triplas.
    K_eigen.makeCompressed(); // Comprime a matriz esparsa para otimizar operações.

    // Resolver o sistema de equações lineares
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver; // Cria um objeto de solver esparso LU.
    solver.compute(K_eigen); // Realiza a decomposição LU da matriz de rigidez.
    if (solver.info() != Eigen::Success) {
        std::cerr << "Decomposição falhou" << std::endl; // Mensagem de erro se a decomposição falhar.
        return;
    }
    u_eigen = solver.solve(f_eigen); // Resolve o sistema linear.
    if (solver.info() != Eigen::Success) {
        std::cerr << "Solução falhou" << std::endl; // Mensagem de erro se a solução falhar.
        return;
    }

    // Copiar os resultados de volta para u
    for (int i = 0; i < N; ++i) {
        u[i] = u_eigen(i); // Copia o vetor solução para o vetor padrão.
    }
}

void FEMSolver2D::imprimir_resultados() const {
    for (int i = 0; i < ny + 1; ++i) {
        for (int j = 0; j < nx + 1; ++j) {
            std::cout << u[i * (nx + 1) + j] << " "; // Imprime os valores da solução.
        }
        std::cout << std::endl; // Nova linha para a próxima linha de nós.
    }
}

void FEMSolver2D::salvar_resultados(const std::string &filename) const {
    std::ofstream outfile(filename); // Cria um objeto de saída de arquivo.
    outfile << std::setprecision(20); // Define a precisão para a saída dos valores.
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            outfile << u[j * (nx + 1) + i] << " "; // Escreve os valores da solução no arquivo.
        }
        outfile << std::endl; // Nova linha para a próxima linha de nós.
    }
    outfile.close(); // Fecha o arquivo.
}

int main(int argc, char *argv[]) {
    int nx = argc > 1 ? atoi(argv[1]) : 100; // Número de nós na direção x, por padrão 100.
    int ny = argc > 2 ? atoi(argv[2]) : 100; // Número de nós na direção y, por padrão 100.
    FEMSolver2D solver(nx, ny); // Cria um objeto solver com o número de nós especificado.

    auto start = std::chrono::high_resolution_clock::now(); // Marca o tempo inicial para a construção da matriz e vetor.
    solver.construir_matriz_e_vetor(); // Chama o método para construir a matriz de rigidez e vetor de forças.
    auto end = std::chrono::high_resolution_clock::now(); // Marca o tempo final.
    std::chrono::duration<double, std::milli> diff = end - start; // Calcula a duração em milissegundos.
    std::cout << "\033[0;31mTempo para construir matriz e vetor: " << std::scientific << diff.count() << " ms\033[0m\n"; // Imprime o tempo gasto.

    start = std::chrono::high_resolution_clock::now(); // Marca o tempo inicial para a aplicação das condições de contorno.
    solver.aplicar_condicoes_contorno(); // Chama o método para aplicar as condições de contorno.
    end = std::chrono::high_resolution_clock::now(); // Marca o tempo final.
    diff = end - start; // Calcula a duração em milissegundos.
    std::cout << "Tempo para aplicar condições de contorno: " << std::scientific << diff.count() << " ms\n"; // Imprime o tempo gasto.

    start = std::chrono::high_resolution_clock::now(); // Marca o tempo inicial para resolver o sistema.
    solver.resolver(); // Chama o método para resolver o sistema linear.
    end = std::chrono::high_resolution_clock::now(); // Marca o tempo final.
    diff = end - start; // Calcula a duração em milissegundos.
    std::cout << "Tempo para resolver o sistema: " << std::scientific << diff.count() << " ms\n"; // Imprime o tempo gasto.

    // solver.imprimir_resultados(); // (Comentado) Imprime os resultados na saída padrão.
    solver.salvar_resultados("resultados.txt"); // Salva os resultados no arquivo "resultados.txt".
    return 0; // Retorna 0 indicando sucesso na execução.
}
