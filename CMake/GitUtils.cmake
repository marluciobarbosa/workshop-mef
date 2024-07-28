cmake_minimum_required(VERSION 3.18)

include(${CMAKE_CURRENT_LIST_DIR}/Utils.cmake)
include(CMakeParseArguments)

find_package(Git)
if(NOT GIT_FOUND)
    message(FATAL_ERROR "git não encontrado!")
endif()


# clonar um repositório git em um diretório no momento da configuração
# isso pode ser útil para incluir projetos da biblioteca cmake que contêm arquivos *.cmake
# A função também iniciará automaticamente os submódulos do git
#
# ATENÇÃO: os arquivos CMakeLists no repositório clonado NÃO serão criados automaticamente
#
# Por que não usar ExternalProject_Add, você pergunta? Porque precisamos executá-la no momento da configuração
#
# USO:
# git_clone(
# PROJECT_NAME <nome do projeto>
# GIT_URL <url>
# [GIT_TAG|GIT_BRANCH|GIT_COMMIT <symbol>]
# [DIRECTORY <dir>]
# [QUIET]
# )
#
#
# ARGUMENTOS:
# PROJECT_NAME
# nome do projeto que será usado nas variáveis de saída.
# Deve ser o mesmo que o nome do diretório/repo do git
#
# GIT_URL
# url para o repositório git
#
# GIT_TAG|GIT_BRANCH|GIT_COMMIT
# opcional
# a tag/branch/commit para checkout
# o padrão é master
#
# DIRETÓRIO
# opcional
# o diretório no qual o projeto será clonado
# o padrão é o diretório de compilação, semelhante ao ExternalProject (${CMAKE_BINARY_DIR})
#
# QUIET
# opcional
# Não imprime mensagens de status
#
# SOURCE_DIR_VARIABLE
# Opcional
# A variável será definida para conter o caminho para o diretório clonado.
# Se não for definida, o caminho será definido em <nome do projeto>_SOURCE_DIR
#
# CLONE_RESULT_VARIABLE
# opcional
# a variável será definida para conter o resultado do clone. TRUE - sucesso, FALSE - erro
# se não for definido, o resultado será definido em <nome do projeto>_CLONE_RESULT
#
#
#
# VARIÁVEIS DE SAÍDA:
#<nome do projeto>_SOURCE_DIR
# opcional, existe quando SOURCE_DIR_VARIABLE não está definida      
# Diretório de nível superior da fonte do projeto clonado
#
#<nome do projeto>_CLONE_RESULT
# opcional, existe quando CLONE_RESULT_VARIABLE não está definida      
# Resultado da função git_clone. TRUE - sucesso, FALSE - erro
#
#
# EXEMPLO:
# git_clone(
# PROJECT_NAME testProj
# GIT_URL https://github.com/test/test.git
# GIT_COMMIT a1b2c3
# DIRETÓRIO ${CMAKE_BINARY_DIR}
# QUIET
# )
#
# Incluir(${testProj_SOURCE_DIR}/cmake/myFancyLib.cmake) Traduzido com www.DeepL.com/Translator (versão gratuita)

function(git_clone)

    cmake_parse_arguments(
            PARGS                                                                                                         # prefix of output variables
            "QUIET"                                                                                                       # list of names of the boolean arguments (only defined ones will be true)
            "PROJECT_NAME;GIT_URL;GIT_TAG;GIT_BRANCH;GIT_COMMIT;DIRECTORY;SOURCE_DIR_VARIABLE;CLONE_RESULT_VARIABLE"      # list of names of mono-valued arguments
            ""                                                                                                            # list of names of multi-valued arguments (output variables are lists)
            ${ARGN}                                                                                                       # arguments of the function to parse, here we take the all original ones
    )                                                                                                                     # remaining unparsed arguments can be found in PARGS_UNPARSED_ARGUMENTS
    if(NOT PARGS_PROJECT_NAME)
        message(FATAL_ERROR "Você deve fornecer um nome de projeto")
    endif()

    if(NOT PARGS_GIT_URL)
        message(FATAL_ERROR "Você deve fornecer uma url do git")
    endif()

    if(NOT PARGS_DIRECTORY)
        set(PARGS_DIRECTORY ${CMAKE_BINARY_DIR})
    endif()

    if(NOT PARGS_SOURCE_DIR_VARIABLE)
        set(${PARGS_PROJECT_NAME}_SOURCE_DIR
                ${PARGS_DIRECTORY}/${PARGS_PROJECT_NAME}
                CACHE INTERNAL "" FORCE) # makes var visible everywhere because PARENT_SCOPE wouldn't include this scope
        
        set(SOURCE_DIR ${PARGS_PROJECT_NAME}_SOURCE_DIR)
    else()
        set(${PARGS_SOURCE_DIR_VARIABLE}
                ${PARGS_DIRECTORY}/${PARGS_PROJECT_NAME}
                CACHE INTERNAL "" FORCE) # makes var visible everywhere because PARENT_SCOPE wouldn't include this scope
        
        set(SOURCE_DIR ${PARGS_SOURCE_DIR_VARIABLE})
    endif()

    if(NOT PARGS_CLONE_RESULT_VARIABLE)   
        set(CLONE_RESULT ${PARGS_PROJECT_NAME}_CLONE_RESULT)
    else()
        set(CLONE_RESULT ${PARGS_CLONE_RESULT_VARIABLE})
    endif()    

    # check that only one of GIT_TAG xor GIT_BRANCH xor GIT_COMMIT was passed
    at_most_one(at_most_one_tag ${PARGS_GIT_TAG} ${PARGS_GIT_BRANCH} ${PARGS_GIT_COMMIT})

    if(NOT at_most_one_tag)
        message(FATAL_ERROR "você só pode fornecer uma das opções GIT_TAG, GIT_BRANCH ou GIT_COMMIT")
    endif()

    if(NOT PARGS_QUIET)
        message(STATUS "baixando/atualizando ${PARGS_PROJECT_NAME}")
    endif()

    # first clone the repo
    if(EXISTS ${${SOURCE_DIR}})
        if(NOT PARGS_QUIET)
            message(STATUS "Diretório ${PARGS_PROJECT_NAME} encontrado, puxando...")
        endif()

        execute_process(
                COMMAND             ${GIT_EXECUTABLE} pull origin master
                WORKING_DIRECTORY   ${${SOURCE_DIR}}
                RESULT_VARIABLE     git_result
                OUTPUT_VARIABLE     git_output)
        if(git_result EQUAL "0")
                execute_process(
                COMMAND             ${GIT_EXECUTABLE} submodule update --remote
                WORKING_DIRECTORY   ${${SOURCE_DIR}}
                RESULT_VARIABLE     git_result
                OUTPUT_VARIABLE     git_output)
                if(NOT git_result EQUAL "0")
                    set(${CLONE_RESULT} FALSE CACHE INTERNAL "" FORCE)
                    if(NOT PARGS_QUIET)
                        message(WARNING "Erro de atualização do submódulo ${PARGS_PROJECT_NAME}") #ToDo: maybe FATAL_ERROR?
                    endif()
                    return()
                endif()
        else()
            set(${CLONE_RESULT} FALSE CACHE INTERNAL "" FORCE)
            if(NOT PARGS_QUIET)
                message(WARNING "${PARGS_PROJECT_NAME} erro de extração")  #ToDo: maybe FATAL_ERROR?
            endif()
            return()
        endif()
    else()
        if(NOT PARGS_QUIET)
            message(STATUS "Diretório ${PARGS_PROJECT_NAME} não encontrado, clonando...")
        endif()

        execute_process(
                COMMAND             ${GIT_EXECUTABLE} clone ${PARGS_GIT_URL}  --depth 1 --recursive ${${SOURCE_DIR}}
                WORKING_DIRECTORY   ${PARGS_DIRECTORY}
                RESULT_VARIABLE     git_result
                OUTPUT_VARIABLE     git_output)
        if(NOT git_result EQUAL "0")
            set(${CLONE_RESULT} FALSE CACHE INTERNAL "" FORCE)
            if(NOT PARGS_QUIET)
                message(WARNING "Erro de clonagem de ${PARGS_PROJECT_NAME}")  #ToDo: maybe FATAL_ERROR?
            endif()
            return()
        endif()        
    endif()


    if(NOT PARGS_QUIET)
        message(STATUS "${git_output}")
    endif()

    # now checkout the right commit
    if(PARGS_GIT_TAG)
        execute_process(
                COMMAND             ${GIT_EXECUTABLE} fetch --all --tags --prune
                COMMAND             ${GIT_EXECUTABLE} checkout tags/${PARGS_GIT_TAG} -b tag_${PARGS_GIT_TAG}
                WORKING_DIRECTORY   ${${SOURCE_DIR}}
                RESULT_VARIABLE     git_result
                OUTPUT_VARIABLE     git_output)
    elseif(PARGS_GIT_BRANCH OR PARGS_GIT_COMMIT)
        execute_process(
                COMMAND             ${GIT_EXECUTABLE} checkout ${PARGS_GIT_BRANCH} ${PARGS_GIT_COMMIT}
                WORKING_DIRECTORY   ${${SOURCE_DIR}}
                RESULT_VARIABLE     git_result
                OUTPUT_VARIABLE     git_output)
    else()
        if(NOT PARGS_QUIET)
            message(STATUS "nenhuma tag especificada, padrão para master")
        endif()
        execute_process(
                COMMAND             ${GIT_EXECUTABLE} checkout master
                WORKING_DIRECTORY   ${${SOURCE_DIR}}
                RESULT_VARIABLE     git_result
                OUTPUT_VARIABLE     git_output)
    endif()
    if(NOT git_result EQUAL "0")
        set(${CLONE_RESULT} FALSE CACHE INTERNAL "" FORCE)
        if(NOT PARGS_QUIET)
            message(WARNING "${PARGS_PROJECT_NAME} ocorreu algum erro. ${git_output}")  #ToDo: maybe FATAL_ERROR?
        endif()
        return()
    else()
        set(${CLONE_RESULT} TRUE CACHE INTERNAL "" FORCE)
    endif()
    if(NOT PARGS_QUIET)
        message(STATUS "${git_output}")
    endif()
endfunction()
