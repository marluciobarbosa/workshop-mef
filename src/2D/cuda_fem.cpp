#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

// Kernel CUDA para calcular a carga
__kernel__ double carga(double x, double y) {
  // Exemplo de função arbitrária: f(x, y) = x + y
  return 1.0;
}

// Kernel CUDA para construir a matriz e o vetor
__global__ void construir_matriz_e_vetor_kernel(double *K, double *f, int nx,
                                                int ny, double hx, double hy) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (j < nx && i < ny) {
    int n = j * ny + i;
    if (j > 0)
      K[n * N + (n - ny)] = -1 / (hx * hx);
    if (j < nx - 1)
      K[n * N + (n + ny)] = -1 / (hx * hx);
    if (i > 0)
      K[n * N + (n - 1)] = -1 / (hy * hy);
    if (i < ny - 1)
      K[n * N + (n + 1)] = -1 / (hy * hy);
    K[n * N + n] = 2 / (hx * hx) + 2 / (hy * hy);
    f[n] = carga(i * hx, j * hy); // x = i * hx;y = j * hy;
  }
}

// Kernel CUDA para resolver o sistema
__global__ void resolver_sistema_kernel(double *K, double *f, double *u,
                                        int N) {
  // Implementação simplificada - pode não ser eficiente para grandes sistemas
  for (int i = 0; i < N; ++i) {
    double pivot = K[i * N + i];
    for (int j = i + 1; j < N; ++j) {
      double ratio = K[j * N + i] / pivot;
      for (int k = i; k < N; ++k) {
        K[j * N + k] -= ratio * K[i * N + k];
      }
      f[j] -= ratio * f[i];
    }
  }

  for (int i = N - 1; i >= 0; --i) {
    u[i] = f[i];
    for (int j = i + 1; j < N; ++j) {
      u[i] -= K[i * N + j] * u[j];
    }
    u[i] /= K[i * N + i];
  }
}

class FEMSolver2D {
public:
  FEMSolver2D(int nx, int ny);
  void construir_matriz_e_vetor();
  void aplicar_condicoes_contorno();
  void resolver();
  void imprimir_resultados() const;
  void salvar_resultados(const std::string &filename) const;

private:
  int nx, ny;                         // Número de nós na direção x e y
  int N;                              // Total de nós na malha (nx * ny)
  std::vector<std::vector<double>> K; // Matriz de rigidez (esparsa)
  std::vector<double> f;              // Vetor de cargas
  std::vector<double> u;              // Vetor solução
  double hx, hy;                      // Tamanho dos elementos em x e y
  double *d_A, *d_f, *d_u; // Ponteiros para a matriz e o vetor na GPU

  void resolver_sistema();
};

FEMSolver2D::FEMSolver2D(int nx, int ny)
    : nx(nx), ny(ny), N(nx * ny), K(N, std::vector<double>(N, 0.0)), f(N, 0.0),
      u(N, 0.0) {
  hx = 1.0 / (nx - 1);
  hy = 1.0 / (ny - 1);
}

void FEMSolver2D::construir_matriz_e_vetor() {
  cudaMalloc(&d_A, N * N * sizeof(double));
  cudaMalloc(&d_f, N * sizeof(double));

  dim3 blockDim(16, 16);
  dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
               (ny + blockDim.y - 1) / blockDim.y);
  construir_matriz_e_vetor_kernel<<<gridDim, blockDim>>>(d_A, d_f, nx, ny, hx,
                                                         hy);

  cudaMemcpy(K.data(), d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(f.data(), d_f, N * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_f);
}

void FEMSolver2D::aplicar_condicoes_contorno() {
  std::vector<int> boundary;

  // Adicionar índices das bordas
  for (int i = 0; i < nx; ++i) {
    boundary.push_back(i);                 // Borda inferior
    boundary.push_back((ny - 1) * nx + i); // Borda superior
  }
  for (int j = 1; j < ny - 1; ++j) {
    boundary.push_back(j * nx);          // Borda esquerda
    boundary.push_back(j * nx + nx - 1); // Borda direita
  }

  // Aplicar condições de contorno
  for (int idx : boundary) {
    for (int j = 0; j < N; ++j) {
      K[idx][j] = 0.0;
    }
    K[idx][idx] = 1.0;
    f[idx] = 0.0;
  }
}

void FEMSolver2D::resolver() { resolver_sistema(); }

// void FEMSolver2D::resolver_sistema() {
//   cudaMalloc(&d_A, N * N * sizeof(double));
//   cudaMalloc(&d_f, N * sizeof(double));
//   cudaMalloc(&d_u, N * sizeof(double));

//   cudaMemcpy(d_A, K.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_f, f.data(), N * sizeof(double), cudaMemcpyHostToDevice);

//   resolver_sistema_kernel<<<1, 1>>>(d_A, d_f, d_u, N);

//   cudaMemcpy(u.data(), d_u, N * sizeof(double), cudaMemcpyDeviceToHost);

//   cudaFree(d_A);
//   cudaFree(d_f);
//   cudaFree(d_u);
// }

void FEMSolver2D::resolver_sistema() {
  Eigen::SparseMatrix<double> K_eigen(N, N);
  Eigen::VectorXd f_eigen(N);
  Eigen::VectorXd u_eigen(N);

  // Preenchendo a matriz K_eigen e o vetor f_eigen com os valores de K e f
  std::vector<Eigen::Triplet<double>> tripletList;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (K[i][j] != 0) {
        tripletList.push_back(Eigen::Triplet<double>(i, j, K[i][j]));
      }
    }
    f_eigen(i) = f[i];
  }
  K_eigen.setFromTriplets(tripletList.begin(), tripletList.end());
  K_eigen.makeCompressed();

  // Usando Eigen para resolver o sistema linear
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(K_eigen);
  if (solver.info() != Eigen::Success) {
    std::cerr << "Decomposição falhou" << std::endl;
    return;
  }
  u_eigen = solver.solve(f_eigen);
  if (solver.info() != Eigen::Success) {
    std::cerr << "Solução falhou" << std::endl;
    return;
  }

  // Copiando os valores de volta para o vetor u
  for (int i = 0; i < N; ++i) {
    u[i] = u_eigen(i);
  }
}

void FEMSolver2D::imprimir_resultados() const {
  for (int i = 0; i < ny; ++i) {
    for (int j = 0; j < nx; ++j) {
      std::cout << u[i * nx + j] << " ";
    }
    std::cout << std::endl;
  }
}

void FEMSolver2D::salvar_resultados(const std::string &filename) const {
  std::ofstream outfile(filename);
  // Definindo a precisão para 20 casas decimais (ajuste conforme necessário)
  outfile << std::setprecision(20);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      outfile << u[j * nx + i] << " ";
    }
    outfile << std::endl;
  }
  outfile.close();
}

int main() {
  int nx = 250; // Número de nós na direção x
  int ny = 250; // Número de nós na direção y
  FEMSolver2D solver(nx, ny);

  auto start = std::chrono::high_resolution_clock::now();
  solver.construir_matriz_e_vetor();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << "Tempo para construir matriz e vetor: " << diff.count()
            << " s\n";

  start = std::chrono::high_resolution_clock::now();
  solver.aplicar_condicoes_contorno();
  end = std::chrono::high_resolution_clock::now();
  diff = end - start;
  std::cout << "Tempo para aplicar condições de contorno: " << diff.count()
            << " s\n";

  start = std::chrono::high_resolution_clock::now();
  solver.resolver();
  end = std::chrono::high_resolution_clock::now();
  diff = end - start;
  std::cout << "Tempo para resolver o sistema: " << diff.count() << " s\n";

  //   solver.imprimir_resultados();
  solver.salvar_resultados("resultados.txt");
  return 0;
}