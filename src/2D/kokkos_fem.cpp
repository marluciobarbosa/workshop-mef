#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Kokkos_Core.hpp>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

class FEMSolver2D {
public:
  FEMSolver2D(int nx, int ny);
  void construir_matriz_e_vetor();
  void aplicar_condicoes_contorno();
  void resolver();
  void imprimir_resultados() const;
  void salvar_resultados(const std::string &filename) const;

private:
  int nx, ny;                // Número de nós na direção x e y
  int N;                     // Total de nós na malha (nx * ny)
  Kokkos::View<double **> K; // Matriz de rigidez (esparsa)
  Kokkos::View<double *> f;  // Vetor de cargas
  Kokkos::View<double *> u;  // Vetor solução
  double hx, hy;             // Tamanho dos elementos em x e y

  KOKKOS_FUNCTION double carga(double x, double y) const;
  void resolver_sistema() const;
};

FEMSolver2D::FEMSolver2D(int nx, int ny)
    : nx(nx), ny(ny), N(nx * ny), K("K", N, N), f("f", N), u("u", N) {
  hx = 1.0 / (nx - 1);
  hy = 1.0 / (ny - 1);
}

void FEMSolver2D::construir_matriz_e_vetor() {
  Kokkos::parallel_for(
      "construir", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nx, ny}),
      KOKKOS_LAMBDA(const int j, const int i) {
        int n = j * ny + i;
        if (j > 0)
          K(n, n - ny) = -1 / (hx * hx);
        if (j < nx - 1)
          K(n, n + ny) = -1 / (hx * hx);
        if (i > 0)
          K(n, n - 1) = -1 / (hy * hy);
        if (i < ny - 1)
          K(n, n + 1) = -1 / (hy * hy);
        K(n, n) = 2 / (hx * hx) + 2 / (hy * hy);
        f(n) = carga(i * hx, j * hy); // x = i * hx;y = j * hy;
      });
}

void FEMSolver2D::aplicar_condicoes_contorno() {
  Kokkos::parallel_for(
      "contorno", N, KOKKOS_LAMBDA(const int n) {
        int i = n / nx;
        int j = n % nx;
        if (i == 0 || i == ny - 1 || j == 0 || j == nx - 1) {
          for (int k = 0; k < N; ++k) {
            K(n, k) = 0.0;
          }
          K(n, n) = 1.0;
          f(n) = 0.0;
        }
      });
}

void FEMSolver2D::resolver() { resolver_sistema(); }

KOKKOS_FUNCTION double FEMSolver2D::carga(double x, double y) const {
  // Exemplo de função arbitrária: f(x, y) = x + y
  return 1.0;
}

// void FEMSolver2D::resolver_sistema() const {
//   // Implementar um solucionador de sistema linear simples
//   // Nota: Para problemas maiores, um método mais eficiente seria necessário
//   for (int i = 0; i < N; ++i) {
//     double pivot = K(i, i);
//     assert(std::abs(pivot) > 1e-10);

//     Kokkos::parallel_for(
//         "eliminar", N - i - 1, KOKKOS_LAMBDA(const int j) {
//           int row = i + j + 1;
//           double ratio = K(row, i) / pivot;
//           for (int k = i; k < N; ++k) {
//             K(row, k) -= ratio * K(i, k);
//           }
//           f(row) -= ratio * f(i);
//         });
//   }

//   for (int i = N - 1; i >= 0; --i) {
//     u(i) = f(i);
//     for (int j = i + 1; j < N; ++j) {
//       u(i) -= K(i, j) * u(j);
//     }
//     u(i) /= K(i, i);
//   }
// }

void FEMSolver2D::resolver_sistema() const {
  Eigen::SparseMatrix<double> K_eigen(N, N);
  Eigen::VectorXd f_eigen(N);
  Eigen::VectorXd u_eigen(N);

  // Preenchendo a matriz K_eigen e o vetor f_eigen com os valores de K e f
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (K(i, j) != 0) {
        K_eigen.insert(i, j) = K(i, j);
      }
    }
    f_eigen(i) = f(i);
  }

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

  // Copiando os valores de volta para o Kokkos::View u
  auto h_u = Kokkos::create_mirror_view(u);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i) {
            h_u(i) = u_eigen(i); 
        }
    );
  Kokkos::fence();
  Kokkos::deep_copy(u, h_u);
}

void FEMSolver2D::imprimir_resultados() const {
  auto h_u = Kokkos::create_mirror_view(u);
  Kokkos::deep_copy(h_u, u);
  for (int i = 0; i < ny; ++i) {
    for (int j = 0; j < nx; ++j) {
      std::cout << h_u(i * nx + j) << " ";
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

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
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

    // solver.imprimir_resultados();
    solver.salvar_resultados("resultados.txt");
  }
  Kokkos::finalize();
  return 0;
}