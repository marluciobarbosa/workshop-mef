#!/bin/bash


# Otimiza perfomance para OpenMP 4.0 ou maior
export OMP_PROC_BIND=spread && export OMP_PLACES=threads

# Verifica se dois argumentos foram passados
if [ "$#" -ne 2 ]; then
    echo -e "\033[0;31mUso: $0 nx ny\033[0m"  # Vermelho para a mensagem de uso
    exit 1
fi

# Armazena os argumentos nx e ny
nx=$1
ny=$2
N=$(( ($nx + 1) * ($ny + 1) ))

# Caminho para o diretório de binários
BIN_DIR="./build/bin"

# Lista de executáveis para rodar
executables=("basic_fem" "openmp_fem" "openacc_fem" "cuda_fem" "kokkos_fem" "stdpar_fem")

# Executa cada binário com os argumentos nx e ny
for exe in ${executables[@]}; do
    if [ -f "${BIN_DIR}/${exe}" ]; then
        echo -e "\033[0;32mExecutando ${exe} com nx=${nx} e ny=${ny} (N = ${N})\033[0m"  # Verde para a mensagem de execução
        "${BIN_DIR}/${exe}" $nx $ny
        echo ""
    else
        echo -e "\033[0;31mExecutável não encontrado: ${BIN_DIR}/${exe}\033[0m"  # Vermelho para a mensagem de erro
        echo ""
    fi
done
