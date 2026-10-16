cmake_minimum_required(VERSION 3.21)
project(synps LANGUAGES C CXX)

set(SYNPS_VERSION "1.0.0")

include(GNUInstallDirs)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# Os binarios ficam na raiz do diretorio de build (`build/synps`), que e o
# caminho que a documentacao e os projetos de exemplo citam.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

set(MLN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../MyeLiN" CACHE PATH
    "Raiz do SDK e dos contratos compartilhados do MyeLiN")
set(MLN_COMPILER_API "${CMAKE_BINARY_DIR}/src/compiler/libsyn_compiler_api.so" CACHE FILEPATH
    "Biblioteca do Compiler usada pelo comando synps build")
# Migrate build trees created before the compiler sources moved under src/.
if(MLN_COMPILER_API STREQUAL "${CMAKE_BINARY_DIR}/compiler/libsyn_compiler_api.so")
    set(MLN_COMPILER_API "${CMAKE_BINARY_DIR}/src/compiler/libsyn_compiler_api.so" CACHE FILEPATH
        "Biblioteca do Compiler usada pelo comando synps build" FORCE)
endif()
