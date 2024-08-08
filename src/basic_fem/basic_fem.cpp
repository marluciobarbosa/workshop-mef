#include <Eigen/Dense>     // Inclui a biblioteca Eigen para operações com matrizes densas.
#include <Eigen/Sparse>    // Inclui a biblioteca Eigen para operações com matrizes esparsas.
#include <cassert>         // Inclui a biblioteca assert para verificações durante o desenvolvimento.
#include <chrono>          // Inclui a biblioteca chrono para medição de tempo.
#include <cmath>           // Inclui a biblioteca cmath para funções matemáticas padrão.
#include <fstream>         // Inclui a biblioteca fstream para operações de leitura e escrita em arquivos.
#include <iomanip>         // Inclui a biblioteca iomanip para manipulação de formatação de saída.
#include <iostream>        // Inclui a biblioteca iostream para operações de entrada e saída padrão.
#include <vector>          // Inclui a biblioteca vector para utilizar vetores dinâmicos.

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
  int nx, ny;                              // Número de nós na direção x e y.
  int N;                                   // Total de nós na malha (nx * ny).
  std::vector<std::vector<double>> K;      // Matriz de rigidez (densa).
  std::vector<double> f;                   // Vetor de cargas.
  std::vector<double> u;                   // Vetor solução.
  std::vector<std::vector<int>> elementos; // Vetor para armazenar os elementos (cada elemento é um vetor de índices de nós).
  std::vector<std::vector<double>> nos;    // Vetor para armazenar as coordenadas dos nós.
  double hx, hy;                           // Tamanho dos elementos em x e y.

  double fonte(double x, double y); // Função que define o termo fonte.
  void resolver_sistema(); // Método para resolver o sistema linear usando a biblioteca Eigen.
};

FEMSolver2D::FEMSolver2D(int nx, int ny)
    : nx(nx), ny(ny), N((nx + 1) * (ny + 1)), K(N, std::vector<double>(N, 0.0)),
      f(N, 0.0), u(N, 0.0) {
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
  nos.resize((nx + 1) * (ny + 1), std::vector<double>(2, 0.0)); // Redimensiona o vetor de nós para armazenar as coordenadas de cada nó.
  for (int j = 0; j <= ny; ++j) {
    for (int i = 0; i <= nx; ++i) {
      int idx = j * (nx + 1) + i; // Calcula o índice do nó atual.
      nos[idx][0] = i * hx; // Define a coordenada x do nó.
      nos[idx][1] = j * hy; // Define a coordenada y do nó.
    }
  }

  // Geração dos elementos
  elementos.reserve(2 * nx * ny); // Reserva espaço para todos os elementos (dois triângulos por célula quadrada).
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      int n1 = j * (nx + 1) + i; // Nó inferior esquerdo do quadrado.
      int n2 = n1 + 1; // Nó inferior direito do quadrado.
      int n3 = n1 + (nx + 1); // Nó superior esquerdo do quadrado.
      int n4 = n3 + 1; // Nó superior direito do quadrado.

      elementos.push_back({n1, n3, n2}); // Primeiro triângulo: inferior esquerdo, superior esquerdo, inferior direito.
      elementos.push_back({n2, n3, n4}); // Segundo triângulo: inferior direito, superior esquerdo, superior direito.
    }
  }
}

void FEMSolver2D::construir_matriz_e_vetor() {

  // Funções de forma lineares para elementos triangulares
  auto forma = [](double xi, double eta) {
    return std::vector<double>{1 - xi - eta, xi, eta}; // Funções de forma para elemento triangular linear.
  };
  // Derivadas das funções de forma lineares 
  std::vector<std::vector<double>> dN_dxi = {{-1, 1, 0}, {-1, 0, 1}}; // Derivadas parciais das funções de forma em relação às coordenadas locais xi e eta.

  // Integração numérica (1 ponto de Gauss)
  std::vector<double> pontoDeGauss = {1.0 / 3, 1.0 / 3}; // Ponto de Gauss no centro do triângulo.
  double pesoDeGauss = 1.0 / 2.0; // Peso associado ao ponto de Gauss.

  for (const auto &elemento : elementos) {
    std::vector<std::vector<double>> coords_e(3, std::vector<double>(2, 0.0)); // Coordenadas dos nós do elemento atual.
    for (int i = 0; i < 3; ++i) {
      coords_e[i][0] = nos[elemento[i]][0]; // Coordenada x do nó i do elemento.
      coords_e[i][1] = nos[elemento[i]][1]; // Coordenada y do nó i do elemento.
    }

    std::vector<std::vector<double>> J(2, std::vector<double>(2, 0.0)); // Matriz Jacobiana do elemento.
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 3; ++k) {
          J[i][j] += dN_dxi[i][k] * coords_e[k][j]; // Cálculo da matriz Jacobiana.
        }
      }
    }

    double detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0]; // Determinante da matriz Jacobiana.
    std::vector<std::vector<double>> invJ = {{J[1][1] / detJ, -J[0][1] / detJ},
                                             {-J[1][0] / detJ, J[0][0] / detJ}}; // Matriz inversa de J.

    std::vector<std::vector<double>> dN_dxy(2, std::vector<double>(3, 0.0)); // Derivadas das funções de forma nas coordenadas globais.
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 2; ++k) {
          dN_dxy[i][j] += invJ[i][k] * dN_dxi[k][j]; // Transformação das derivadas para coordenadas globais.
        }
      }
    }

    std::vector<double> N_e = forma(pontoDeGauss[0], pontoDeGauss[1]); // Valores das funções de forma no ponto de Gauss.

    std::vector<std::vector<double>> K_e(3, std::vector<double>(3, 0.0)); // Matriz de rigidez local do elemento.
    std::vector<double> F_e(3, 0.0); // Vetor de forças local do elemento.

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 2; ++k) {
          K_e[i][j] += dN_dxy[k][i] * dN_dxy[k][j] * detJ * pesoDeGauss; // Cálculo da matriz de rigidez local usando integração numérica.
        }
      }
    }

    for (int i = 0; i < 3; ++i) {
      double x = 0.0, y = 0.0;
      for (int j = 0; j < 3; ++j) {
        x += N_e[j] * coords_e[j][0]; // Calcula a coordenada x do ponto de integração.
        y += N_e[j] * coords_e[j][1]; // Calcula a coordenada y do ponto de integração.
      }
      F_e[i] = N_e[i] * fonte(x, y) * detJ * pesoDeGauss; // Cálculo do vetor de forças local.
    }

    for (int i = 0; i < 3; ++i) {
      f[elemento[i]] += F_e[i]; // Soma o vetor de forças local ao vetor de forças global.
      for (int j = 0; j < 3; ++j) {
        K[elemento[i]][elemento[j]] += K_e[i][j]; // Soma a matriz de rigidez local à matriz de rigidez global.
      }
    }
  }
}

void FEMSolver2D::aplicar_condicoes_contorno() {
  std::vector<int> contorno; // Vetor para armazenar os índices dos nós na borda.

  for (int i = 0; i <= nx; ++i) {
    contorno.push_back(i);                 // Borda inferior.
    contorno.push_back(ny * (nx + 1) + i); // Borda superior.
  }
  for (int j = 1; j < ny; ++j) {
    contorno.push_back(j * (nx + 1));      // Borda esquerda.
    contorno.push_back(j * (nx + 1) + nx); // Borda direita.
  }

  for (int idx : contorno) { // Para cada nó na borda.
    for (int j = 0; j < N; ++j) {
      K[idx][j] = 0.0; // Zera a linha correspondente na matriz de rigidez.
    }
    K[idx][idx] = 1.0; // Define um 1 na diagonal para garantir solução única.
    f[idx] = 0.0; // Define o vetor de forças para 0 nesse nó (condição de Dirichlet nula).
  }
}

void FEMSolver2D::resolver() { resolver_sistema(); } // Método que chama a função de resolver o sistema linear.

double FEMSolver2D::fonte(double x, double y) {
  return 2 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y); // Função que representa o termo fonte.
}

void FEMSolver2D::resolver_sistema() {
  Eigen::SparseMatrix<double> K_eigen(N, N); // Criação de uma matriz esparsa Eigen para armazenar a matriz de rigidez.
  Eigen::VectorXd f_eigen(N); // Criação de um vetor Eigen para armazenar o vetor de forças.
  Eigen::VectorXd u_eigen(N); // Criação de um vetor Eigen para armazenar o vetor solução.

  std::vector<Eigen::Triplet<double>> tripletList; // Lista de triplas para construir a matriz esparsa.
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (K[i][j] != 0) {
        tripletList.push_back(Eigen::Triplet<double>(i, j, K[i][j])); // Armazena apenas os valores não nulos.
      }
    }
    f_eigen(i) = f[i]; // Copia o vetor de forças para o vetor Eigen.
  }
  K_eigen.setFromTriplets(tripletList.begin(), tripletList.end()); // Constrói a matriz esparsa a partir das triplas.
  K_eigen.makeCompressed(); // Comprime a matriz esparsa para otimizar operações.

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