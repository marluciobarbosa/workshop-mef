#include <Kokkos_Core.hpp> // Inclui a biblioteca Kokkos para computação paralela e abstração de memória.
#include <Eigen/Dense>     // Inclui a biblioteca Eigen para operações com matrizes densas.
#include <Eigen/Sparse>    // Inclui a biblioteca Eigen para operações com matrizes esparsas.
#include <cassert>         // Inclui a biblioteca assert para verificações durante o desenvolvimento.
#include <chrono>          // Inclui a biblioteca chrono para medição de tempo.
#include <cmath>           // Inclui a biblioteca cmath para funções matemáticas padrão.
#include <fstream>         // Inclui a biblioteca fstream para operações de leitura e escrita em arquivos.
#include <iomanip>         // Inclui a biblioteca iomanip para manipulação de formatação de saída.
#include <iostream>        // Inclui a biblioteca iostream para operações de entrada e saída padrão.
#include <vector>          // Inclui a biblioteca vector para utilizar vetores dinâmicos.

// Define um alias para o espaço de memória associado a um espaço de execução específico.
template <typename ExecutionSpace>
using MemorySpace = typename ExecutionSpace::memory_space;

// Define tipos de Views Kokkos para diferentes dimensões e espaços de execução.

template <typename ExecutionSpace>
using viewDouble2D = typename Kokkos::View<double **, MemorySpace<ExecutionSpace>>; // View 2D para armazenar dados do tipo double.

template <typename ExecutionSpace>
using host_viewDouble2D = typename viewDouble2D<ExecutionSpace>::HostMirror; // View 2D para host.

template <typename ExecutionSpace>
using viewDouble1D = typename Kokkos::View<double *, MemorySpace<ExecutionSpace>>; // View 1D para armazenar dados do tipo double.

template <typename ExecutionSpace>
using host_viewDouble1D = typename viewDouble1D<ExecutionSpace>::HostMirror; // View 1D para host.

template <typename ExecutionSpace>
using viewInt2D = typename Kokkos::View<int **, MemorySpace<ExecutionSpace>>; // View 2D para armazenar dados do tipo int.

template <typename ExecutionSpace>
using host_viewInt2D = typename viewInt2D<ExecutionSpace>::HostMirror; // View 2D para host.

template <typename ExecutionSpace>
using viewInt1D = typename Kokkos::View<double *, MemorySpace<ExecutionSpace>>; // View 1D para armazenar dados do tipo int.

template <typename ExecutionSpace>
using host_viewInt1D = typename viewInt1D<ExecutionSpace>::HostMirror; // View 1D para host.

// Classe para o solucionador do MEF em 2D utilizando Kokkos.
template <typename ExecutionSpace>
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
  int nx, ny; // Número de nós na direção x e y.
  int N;      // Total de nós na malha (nx * ny).

  // Usando Kokkos::View para armazenamento de dados
  viewDouble2D<ExecutionSpace> K;    // Matriz de rigidez.
  viewDouble1D<ExecutionSpace> f;    // Vetor de cargas.
  viewDouble1D<ExecutionSpace> u;    // Vetor solução.

  viewInt2D<ExecutionSpace> elementos; // Elementos.
  viewDouble2D<ExecutionSpace> nos; // Nós.

  double hx, hy; // Tamanho dos elementos em x e y.

  double fonte(double x, double y); // Função que define o termo fonte.
  void resolver_sistema(); // Método para resolver o sistema linear usando a biblioteca Eigen.

  // Função inline Kokkos para calcular as funções de forma
  static KOKKOS_INLINE_FUNCTION void calcular_forma(double xi, double eta, double* N) {
    N[0] = 1 - xi - eta;
    N[1] = xi;
    N[2] = eta;
  }
};

template <typename ExecutionSpace> 
FEMSolver2D<ExecutionSpace>::FEMSolver2D(int nx, int ny)
    : nx(nx), ny(ny), N((nx + 1) * (ny + 1)), 
      K("K", N, N), f("f", N), u("u", N), nos("nos", (nx + 1) * (ny + 1), 2),
      elementos("elementos", 2 * nx * ny, 3) { // Inicializa os views Kokkos.
  hx = 1.0 / nx; // Calcula o tamanho dos elementos na direção x.
  hy = 1.0 / ny; // Calcula o tamanho dos elementos na direção y.
  
  auto start = std::chrono::high_resolution_clock::now(); // Marca o tempo inicial para a construção da malha.
  construir_malha(); // Chama o método para construir a malha.
  auto end = std::chrono::high_resolution_clock::now(); // Marca o tempo final.
  std::chrono::duration<double, std::milli> diff = end - start; // Calcula a duração em milissegundos.
  std::cout << "Tempo de construção da malha: " << std::scientific << diff.count() << " ms\n"; // Imprime o tempo gasto.
}

template <typename ExecutionSpace> 
void FEMSolver2D<ExecutionSpace>::construir_malha() {
  
  // As capturas implícitas de this são substituídas por capturas explícitas das variáveis necessárias.
  auto nx_local = nx;
  auto ny_local = ny;
  auto hx_local = hx;
  auto hy_local = hy;
  auto nos_local = nos;
  auto elementos_local = elementos;
    
  // Geração dos nós usando Kokkos
  Kokkos::parallel_for("construir_malha", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny_local + 1, nx_local + 1}), 
  KOKKOS_LAMBDA(int j, int i) {
    int idx = j * (nx_local + 1) + i; // Calcula o índice do nó atual.
    nos_local(idx, 0) = i * hx_local; // Define a coordenada x do nó.
    nos_local(idx, 1) = j * hy_local; // Define a coordenada y do nó.
  });

  // Geração dos elementos
  Kokkos::parallel_for("construir_elementos", ny_local * nx_local, KOKKOS_LAMBDA(int index) {
    int j = index / nx_local; // Calcula o índice da linha do elemento.
    int i = index % nx_local; // Calcula o índice da coluna do elemento.
    int n1 = j * (nx_local + 1) + i; // Nó inferior esquerdo do quadrado.
    int n2 = n1 + 1; // Nó inferior direito do quadrado.
    int n3 = n1 + (nx_local + 1); // Nó superior esquerdo do quadrado.
    int n4 = n3 + 1; // Nó superior direito do quadrado.

    elementos_local(2 * index, 0) = n1; // Primeiro triângulo: inferior esquerdo.
    elementos_local(2 * index, 1) = n3; // Primeiro triângulo: superior esquerdo.
    elementos_local(2 * index, 2) = n2; // Primeiro triângulo: inferior direito.

    elementos_local(2 * index + 1, 0) = n2; // Segundo triângulo: inferior direito.
    elementos_local(2 * index + 1, 1) = n3; // Segundo triângulo: superior esquerdo.
    elementos_local(2 * index + 1, 2) = n4; // Segundo triângulo: superior direito.
  });
}

template <typename ExecutionSpace> 
void FEMSolver2D<ExecutionSpace>::construir_matriz_e_vetor() {
  // Derivadas das funções de forma lineares 
  const double dN_dxi[2][3] = {{-1, 1, 0}, {-1, 0, 1}}; // Derivadas das funções de forma nas coordenadas locais.

  // Integração numérica (1 ponto de Gauss)
  const double pontoDeGauss[2] = {1.0 / 3, 1.0 / 3}; // Ponto de Gauss no centro do triângulo.
  const double pesoDeGauss = 1.0 / 2.0; // Peso associado ao ponto de Gauss.

  auto nos_local = nos;
  auto K_local = K;
  auto f_local = f;
  auto elementos_local = elementos;

  Kokkos::parallel_for("construir_matriz_e_vetor", elementos_local.extent(0), KOKKOS_LAMBDA(size_t e) {
    double coords_e[3][2]; // Array para armazenar as coordenadas dos nós do elemento atual.
    for (int i = 0; i < 3; ++i) {
      coords_e[i][0] = nos_local(elementos_local(e, i), 0); // Coordenada x do nó i do elemento.
      coords_e[i][1] = nos_local(elementos_local(e, i), 1); // Coordenada y do nó i do elemento.
    }

    double J[2][2] = {{0, 0}, {0, 0}}; // Matriz Jacobiana do elemento.
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 3; ++k) {
          J[i][j] += dN_dxi[i][k] * coords_e[k][j]; // Cálculo da matriz Jacobiana.
        }
      }
    }

    double detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0]; // Determinante da matriz Jacobiana.
    double invJ[2][2] = {{J[1][1] / detJ, -J[0][1] / detJ},
                         {-J[1][0] / detJ, J[0][0] / detJ}}; // Matriz inversa de J.

    double dN_dxy[2][3] = {{0, 0, 0}, {0, 0, 0}}; // Derivadas das funções de forma nas coordenadas globais.
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 2; ++k) {
          dN_dxy[i][j] += invJ[i][k] * dN_dxi[k][j]; // Transformação das derivadas para coordenadas globais.
        }
      }
    }

    double N_e[3]; // Array para armazenar os valores das funções de forma no ponto de Gauss.
    calcular_forma(pontoDeGauss[0], pontoDeGauss[1], N_e);

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
      Kokkos::atomic_add(&f_local(elementos_local(e, i)), F_e[i]); // Soma o vetor de forças local ao vetor de forças global usando operação atômica.
      for (int j = 0; j < 3; ++j) {
        Kokkos::atomic_add(&K_local(elementos_local(e, i), elementos_local(e, j)), K_e[i][j]); // Soma a matriz de rigidez local à matriz de rigidez global usando operação atômica.
      }
    }
  });
}

template <typename ExecutionSpace> 
void FEMSolver2D<ExecutionSpace>::aplicar_condicoes_contorno() {
  auto K_local = K;
  auto f_local = f;
  const auto nx_local = nx;
  const auto ny_local = ny;
  const auto N_local = N;

  Kokkos::parallel_for("aplicar_condicoes_contorno", N, KOKKOS_LAMBDA(const int idx) {
    int i = idx % (nx_local + 1); // Calcula o índice da coluna do nó.
    int j = idx / (nx_local + 1); // Calcula o índice da linha do nó.

    // Verifica se o nó está na borda.
    if (i == 0 || i == nx_local || j == 0 || j == ny_local) {
      for (int k = 0; k < N_local; ++k) {
        K_local(idx, k) = 0.0; // Zera a linha correspondente na matriz de rigidez.
      }
      K_local(idx, idx) = 1.0; // Define 1 na diagonal para garantir solução única.
      f_local(idx) = 0.0; // Define o vetor de forças para 0 nesse nó (condição de Dirichlet nula).
    }
  });
}

template <typename ExecutionSpace> 
void FEMSolver2D<ExecutionSpace>::resolver() { resolver_sistema(); } // Método que chama a função de resolver o sistema linear.

template <typename ExecutionSpace> 
double FEMSolver2D<ExecutionSpace>::fonte(double x, double y) {
  return 2 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y); // Função que representa o termo fonte.
}

template <typename ExecutionSpace>
void FEMSolver2D<ExecutionSpace>::resolver_sistema() {
  Eigen::SparseMatrix<double> K_eigen(N, N); // Criação de uma matriz esparsa Eigen para armazenar a matriz de rigidez.
  Eigen::VectorXd f_eigen(N); // Criação de um vetor Eigen para armazenar o vetor de forças.
  Eigen::VectorXd u_eigen(N); // Criação de um vetor Eigen para armazenar o vetor solução.

  // Definindo explicitamente o tipo do mirror de K
  host_viewDouble2D<ExecutionSpace> host_K = Kokkos::create_mirror_view(K);

  // Definindo explicitamente o tipo do mirror de f
  host_viewDouble1D<ExecutionSpace> host_f = Kokkos::create_mirror_view(f);

  Kokkos::deep_copy(host_K, K); // Copia os dados da matriz de rigidez da GPU para o host.
  Kokkos::deep_copy(host_f, f); // Copia os dados do vetor de forças da GPU para o host.

  // Preenchendo a matriz K_eigen e o vetor f_eigen com os valores de K e f
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (host_K(i, j) != 0) {
        K_eigen.insert(i, j) = host_K(i, j); // Armazena apenas os valores não nulos.
      }
    }
    f_eigen(i) = host_f(i); // Copia o vetor de forças para o vetor Eigen.
  }

  K_eigen.makeCompressed(); // Comprime a matriz esparsa para otimizar operações.

  // Usando Eigen para resolver o sistema linear
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

  // Copiando os valores de volta para o Kokkos::View u
  auto h_u = Kokkos::create_mirror_view(u);

  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    h_u(i) = u_eigen(i); // Copia o vetor solução para o vetor padrão.
  }

  Kokkos::deep_copy(u, h_u); // Copia o vetor solução do host para a GPU.
}

template <typename ExecutionSpace> 
void FEMSolver2D<ExecutionSpace>::imprimir_resultados() const {
  auto h_u = Kokkos::create_mirror_view(u); // Cria um espelho do vetor solução na CPU.
  Kokkos::deep_copy(h_u, u); // Copia os dados do vetor solução da GPU para o host.

  for (int i = 0; i < ny + 1; ++i) {
    for (int j = 0; j < nx + 1; ++j) {
      std::cout << h_u(i * (nx + 1) + j) << " "; // Imprime os valores da solução.
    }
    std::cout << std::endl; // Nova linha para a próxima linha de nós.
  }
}

template <typename ExecutionSpace> 
void FEMSolver2D<ExecutionSpace>::salvar_resultados(const std::string &filename) const {
  auto h_u = Kokkos::create_mirror_view(u); // Cria um espelho do vetor solução na CPU.
  Kokkos::deep_copy(h_u, u); // Copia os dados do vetor solução da GPU para o host.

  std::ofstream outfile(filename); // Cria um objeto de saída de arquivo.
  outfile << std::setprecision(20); // Define a precisão para a saída dos valores.
  for (int j = 0; j <= ny; ++j) {
    for (int i = 0; i <= nx; ++i) {
      outfile << h_u(j * (nx + 1) + i) << " "; // Escreve os valores da solução no arquivo.
    }
    outfile << std::endl; // Nova linha para a próxima linha de nós.
  }
  outfile.close(); // Fecha o arquivo.
}

// Define o espaço de execução padrão para o dispositivo
typedef Kokkos::DefaultExecutionSpace device_execution_space;
// Define um alias para a classe FEMSolver2D com o espaço de execução padrão do dispositivo
typedef FEMSolver2D<device_execution_space> FEMSolver;

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv); // Inicializa a biblioteca Kokkos.
  {
    int nx = argc > 1 ? atoi(argv[1]) : 100; // Número de nós na direção x, por padrão 100.
    int ny = argc > 2 ? atoi(argv[2]) : 100; // Número de nós na direção y, por padrão 100.
    FEMSolver solver(nx, ny); // Cria um objeto solver com o número de nós especificado.

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
  }
  Kokkos::finalize(); // Finaliza a biblioteca Kokkos.
  return 0; // Retorna 0 indicando sucesso na execução.
}
