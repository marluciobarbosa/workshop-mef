#include <openacc.h> // Inclui a biblioteca OpenACC para diretivas de paralelização.

#include <Eigen/Dense>  // Inclui a biblioteca Eigen para operações com matrizes densas.
#include <Eigen/Sparse> // Inclui a biblioteca Eigen para operações com matrizes esparsas.
#include <cassert>      // Inclui a biblioteca assert para verificações durante o desenvolvimento.
#include <chrono>       // Inclui a biblioteca chrono para medição de tempo.
#include <cmath>        // Inclui a biblioteca cmath para funções matemáticas padrão.
#include <fstream>      // Inclui a biblioteca fstream para operações de leitura e escrita em arquivos.
#include <iomanip>      // Inclui a biblioteca iomanip para manipulação de formatação de saída.
#include <iostream>     // Inclui a biblioteca iostream para operações de entrada e saída padrão.
#include <vector>       // Inclui a biblioteca vector para utilizar vetores dinâmicos.

// Classe para o solucionador do MEF em 2D
class FEMSolver2D
{
  public:
    FEMSolver2D(int nx, int ny);                               // Construtor que inicializa o solucionador com
                                                               // número de nós nx e ny.
    ~FEMSolver2D();                                            // Destrutor para liberar memória alocada.
    void construir_malha();                                    // Método para construir a malha de elementos finitos.
    void construir_matriz_e_vetor();                           // Método para construir a matriz de
                                                               // rigidez e o vetor de carga.
    void aplicar_condicoes_contorno();                         // Método para aplicar condições de
                                                               // contorno Dirichlet.
    void resolver();                                           // Método para resolver o sistema linear resultante.
    void imprimir_resultados() const;                          // Método para imprimir os resultados na saída padrão.
    void salvar_resultados(const std::string &filename) const; // Método para salvar os resultados em um arquivo.

    int nx, ny;     // Número de nós na direção x e y.
    int n;          // Total de nós na malha (nx * ny).
    double *K;      // Ponteiro para matriz de rigidez.
    double *f;      // Ponteiro para vetor de cargas.
    double *u;      // Ponteiro para vetor solução.
    int *elementos; // Ponteiro para armazenar os elementos.
    double *nos;    // Ponteiro para armazenar as coordenadas dos nós.
    double hx, hy;  // Tamanho dos elementos em x e y.

    double fonte(double x, double y); // Função que define o termo fonte.
    void resolver_sistema();          // Método para resolver o sistema linear usando a
                                      // biblioteca Eigen.
};

FEMSolver2D::FEMSolver2D(int nx, int ny) : nx(nx), ny(ny), n((nx + 1) * (ny + 1))
{
    hx = 1.0 / nx;                      // Calcula o tamanho dos elementos na direção x.
    hy = 1.0 / ny;                      // Calcula o tamanho dos elementos na direção y.
    K = new double[n * n]();            // Aloca e inicializa a matriz de rigidez.
    f = new double[n]();                // Aloca e inicializa o vetor de cargas.
    u = new double[n]();                // Aloca e inicializa o vetor solução.
    elementos = new int[6 * nx * ny](); // Aloca e inicializa o vetor de elementos.
    nos = new double[2 * n]();          // Aloca e inicializa o vetor de nós.

    auto start = std::chrono::high_resolution_clock::now();       // Marca o tempo inicial
                                                                  // para a construção da
                                                                  // malha.
    construir_malha();                                            // Chama o método para construir a malha.
    auto end = std::chrono::high_resolution_clock::now();         // Marca o tempo final.
    std::chrono::duration<double, std::milli> diff = end - start; // Calcula a duração em milissegundos.
    std::cout << "Tempo de construção da malha: " << std::scientific << diff.count()
              << " ms\n"; // Imprime o tempo gasto.
}

FEMSolver2D::~FEMSolver2D()
{
    delete[] K;         // Libera a memória alocada para a matriz de rigidez.
    delete[] f;         // Libera a memória alocada para o vetor de cargas.
    delete[] u;         // Libera a memória alocada para o vetor solução.
    delete[] elementos; // Libera a memória alocada para os elementos.
    delete[] nos;       // Libera a memória alocada para os nós.
}

void FEMSolver2D::construir_malha()
{
    // Geração dos nós
    for (int j = 0; j <= ny; ++j)
    {
        for (int i = 0; i <= nx; ++i)
        {
            int idx = j * (nx + 1) + i; // Calcula o índice do nó atual.
            nos[2 * idx] = i * hx;      // Define a coordenada x do nó.
            nos[2 * idx + 1] = j * hy;  // Define a coordenada y do nó.
        }
    }

    // Geração dos elementos
    int elem_idx = 0;
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            int n1 = j * (nx + 1) + i; // Nó inferior esquerdo do quadrado.
            int n2 = n1 + 1;           // Nó inferior direito do quadrado.
            int n3 = n1 + (nx + 1);    // Nó superior esquerdo do quadrado.
            int n4 = n3 + 1;           // Nó superior direito do quadrado.

            elementos[elem_idx++] = n1; // Primeiro triângulo: inferior esquerdo.
            elementos[elem_idx++] = n3; // Primeiro triângulo: superior esquerdo.
            elementos[elem_idx++] = n2; // Primeiro triângulo: inferior direito.
            elementos[elem_idx++] = n2; // Segundo triângulo: inferior direito.
            elementos[elem_idx++] = n3; // Segundo triângulo: superior esquerdo.
            elementos[elem_idx++] = n4; // Segundo triângulo: superior direito.
        }
    }
}

void FEMSolver2D::construir_matriz_e_vetor()
{
// Certifique-se de que todos os dados necessários estão copiados para o
// acelerador
#pragma acc enter data copyin(this, this->nx, this->ny, this->n, this->K[ : n * n], this->f[ : n],                     \
                              this->elementos[ : 6 * nx * ny], this->nos[ : 2 * n])
    // Funções de forma lineares para elementos triangulares
    auto N = [](double xi, double eta, double *N_) {
        // Funções de forma para o triângulo.
        N_[0] = 1 - xi - eta;
        N_[1] = xi;
        N_[2] = eta;
    };
    const double dN_dxi[2][3] = {{-1, 1, 0}, {-1, 0, 1}}; // Derivadas das funções de forma nas coordenadas locais.
    const double pontoDeGauss[2] = {1.0 / 3, 1.0 / 3};    // Ponto de Gauss no centro do triângulo.
    const double pesoDeGauss = 1.0 / 2.0;                 // Peso associado ao ponto de Gauss.

#pragma acc parallel loop present(this, this->nx, this->ny, this->n, this->K[ : n * n], this->f[ : n],                 \
                                  this->elementos[ : 6 * nx * ny], this->nos[ : 2 * n])
    for (int e = 0; e < 2 * nx * ny; ++e)
    {                                // Itera sobre cada elemento.
        double coords_e[3][2];       // Array para armazenar as coordenadas dos nós do
                                     // elemento atual.
        double J[2][2] = {{0}};      // Matriz Jacobiana do elemento.
        double invJ[2][2];           // Matriz inversa de J.
        double dN_dxy[2][3] = {{0}}; // Derivadas das funções de forma nas coordenadas globais.
        double K_e[3][3] = {{0}};    // Matriz de rigidez local do elemento.
        double F_e[3] = {0};         // Vetor de forças local do elemento.

        // Calcular coords_e
        for (int i = 0; i < 3; ++i)
        {
            int node = elementos[3 * e + i];    // Índice do nó do elemento.
            coords_e[i][0] = nos[2 * node];     // Coordenada x do nó.
            coords_e[i][1] = nos[2 * node + 1]; // Coordenada y do nó.
        }

        // Calcular J
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                for (int k = 0; k < 3; ++k)
                {
                    J[i][j] += dN_dxi[i][k] * coords_e[k][j]; // Cálculo da matriz Jacobiana.
                }
            }
        }

        double detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0]; // Determinante da matriz Jacobiana.
        invJ[0][0] = J[1][1] / detJ;                         // Cálculo da matriz inversa de J.
        invJ[0][1] = -J[0][1] / detJ;
        invJ[1][0] = -J[1][0] / detJ;
        invJ[1][1] = J[0][0] / detJ;

        // Calcular dN_dxy
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                for (int k = 0; k < 2; ++k)
                {
                    dN_dxy[i][j] += invJ[i][k] * dN_dxi[k][j]; // Transformação das derivadas para
                                                               // coordenadas globais.
                }
            }
        }

        double N_e[3];
        N(pontoDeGauss[0], pontoDeGauss[1],
          N_e); // Funções de forma no ponto de Gauss.

        // Calcular K_e
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                for (int k = 0; k < 2; ++k)
                {
                    K_e[i][j] += dN_dxy[k][i] * dN_dxy[k][j] * detJ * pesoDeGauss; // Cálculo da matriz de rigidez local
                                                                                   // usando integração numérica.
                }
            }
        }

        // Calcular F_e
        for (int i = 0; i < 3; ++i)
        {
            double x = 0.0, y = 0.0; // Inicializa as coordenadas para o cálculo
                                     // do ponto de integração.
            for (int j = 0; j < 3; ++j)
            {
                x += N_e[j] * coords_e[j][0]; // Calcula a coordenada x do ponto
                                              // de integração.
                y += N_e[j] * coords_e[j][1]; // Calcula a coordenada y do ponto
                                              // de integração.
            }
            F_e[i] = N_e[i] * fonte(x, y) * detJ * pesoDeGauss; // Cálculo do vetor de forças local.
        }

        // Atualizar K e f
        for (int i = 0; i < 3; ++i)
        {
            int row = elementos[3 * e + i]; // Índice da linha na matriz global.
#pragma acc atomic update                   // Diretriz OpenACC para garantir operação atômica de
                                            // atualização.
            f[row] += F_e[i];               // Atualiza o vetor de forças global.
            for (int j = 0; j < 3; ++j)
            {
                int col = elementos[3 * e + j]; // Índice da coluna na matriz global.
#pragma acc atomic update                       // Diretriz OpenACC para garantir operação atômica de
                                                // atualização.
                K[row * n + col] += K_e[i][j];  // Atualiza a matriz de rigidez global.
            }
        }
    }

// Certifique-se de que os dados atualizados sejam copiados de volta para o
// host
#pragma acc exit data copyout(this->K[ : n * n], this->f[ : n])
}

void FEMSolver2D::aplicar_condicoes_contorno()
{
    std::vector<int> contorno; // Vetor para armazenar os índices dos nós de contorno.

    for (int i = 0; i <= nx; ++i)
    {
        contorno.push_back(i);                 // Borda inferior.
        contorno.push_back(ny * (nx + 1) + i); // Borda superior.
    }
    for (int j = 1; j < ny; ++j)
    {
        contorno.push_back(j * (nx + 1));      // Borda esquerda.
        contorno.push_back(j * (nx + 1) + nx); // Borda direita.
    }

    for (int idx : contorno)
    { // Para cada nó na borda.
        for (int j = 0; j < n; ++j)
        {
            K[idx * n + j] = 0.0; // Zera a linha correspondente na matriz de rigidez.
        }
        K[idx * n + idx] = 1.0; // Define 1 na diagonal para garantir solução única.
        f[idx] = 0.0;           // Define o vetor de forças para 0 nesse nó (condição de
                                // Dirichlet nula).
    }
}

void FEMSolver2D::resolver()
{
    // Resolver o sistema no lado do host
    resolver_sistema(); // Chama a função para resolver o sistema linear.
}

double FEMSolver2D::fonte(double x, double y)
{
    return 2 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y); // Função que representa o termo fonte.
}

void FEMSolver2D::resolver_sistema()
{
    Eigen::Map<Eigen::MatrixXd> K_eigen(K, n, n); // Mapeia a matriz de rigidez para uma matriz Eigen.
    Eigen::Map<Eigen::VectorXd> f_eigen(f, n);    // Mapeia o vetor de forças para um vetor Eigen.
    Eigen::VectorXd u_eigen(n);                   // Criação de um vetor Eigen para armazenar o vetor solução.

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver; // Cria um objeto de solver esparso LU.
    solver.compute(K_eigen.sparseView());                // Realiza a decomposição LU da matriz de rigidez.
    if (solver.info() != Eigen::Success)
    {
        std::cerr << "Decomposição falhou" << std::endl; // Mensagem de erro se a decomposição falhar.
        return;
    }
    u_eigen = solver.solve(f_eigen); // Resolve o sistema linear.
    if (solver.info() != Eigen::Success)
    {
        std::cerr << "Solução falhou" << std::endl; // Mensagem de erro se a solução falhar.
        return;
    }

    // Copiar o resultado de volta para o vetor u
    for (int i = 0; i < n; ++i)
    {
        u[i] = u_eigen(i); // Copia o vetor solução para o vetor padrão.
    }
}

void FEMSolver2D::imprimir_resultados() const
{
    for (int i = 0; i < ny + 1; ++i)
    {
        for (int j = 0; j < nx + 1; ++j)
        {
            std::cout << u[i * (nx + 1) + j] << " "; // Imprime os valores da solução.
        }
        std::cout << std::endl; // Nova linha para a próxima linha de nós.
    }
}

void FEMSolver2D::salvar_resultados(const std::string &filename) const
{
    std::ofstream outfile(filename);  // Cria um objeto de saída de arquivo.
    outfile << std::setprecision(20); // Define a precisão para a saída dos valores.
    for (int j = 0; j <= ny; ++j)
    {
        for (int i = 0; i <= nx; ++i)
        {
            outfile << u[j * (nx + 1) + i] << " "; // Escreve os valores da solução no arquivo.
        }
        outfile << std::endl; // Nova linha para a próxima linha de nós.
    }
    outfile.close(); // Fecha o arquivo.
}

int main(int argc, char *argv[])
{
    int nx = argc > 1 ? atoi(argv[1]) : 100; // Número de nós na direção x, por padrão 100.
    int ny = argc > 2 ? atoi(argv[2]) : 100; // Número de nós na direção y, por padrão 100.
    FEMSolver2D solver(nx, ny);              // Cria um objeto solver com o número de nós especificado.

    auto start =
        std::chrono::high_resolution_clock::now(); // Marca o tempo inicial para a construção da matriz e vetor.
    solver.construir_matriz_e_vetor();             // Chama o método para construir a matriz
                                                   // de rigidez e vetor de forças.

    auto end = std::chrono::high_resolution_clock::now();         // Marca o tempo final.
    std::chrono::duration<double, std::milli> diff = end - start; // Calcula a duração em milissegundos.
    std::cout << "\033[0;31mTempo para construir matriz e vetor: " << std::scientific << diff.count()
              << " ms\033[0m\n"; // Imprime o tempo gasto.

    start = std::chrono::high_resolution_clock::now(); // Marca o tempo inicial
                                                       // para a aplicação das
                                                       // condições de contorno.
    solver.aplicar_condicoes_contorno();               // Chama o método para aplicar as
                                                       // condições de contorno.
    end = std::chrono::high_resolution_clock::now();   // Marca o tempo final.
    diff = end - start;                                // Calcula a duração em milissegundos.
    std::cout << "Tempo para aplicar condições de contorno: " << std::scientific << diff.count()
              << " ms\n"; // Imprime o tempo gasto.

    start = std::chrono::high_resolution_clock::now(); // Marca o tempo inicial
                                                       // para resolver o sistema.
    solver.resolver();                                 // Chama o método para resolver o sistema linear.

    end = std::chrono::high_resolution_clock::now(); // Marca o tempo final.
    diff = end - start;                              // Calcula a duração em milissegundos.
    std::cout << "Tempo para resolver o sistema: " << std::scientific << diff.count()
              << " ms\n"; // Imprime o tempo gasto.

    solver.salvar_resultados("resultados.txt"); // Salva os resultados no arquivo "resultados.txt".
    return 0;                                   // Retorna 0 indicando sucesso na execução.
}