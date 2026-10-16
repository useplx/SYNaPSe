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
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

set(MLN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../MyeLiN" CACHE PATH "MyeLiN source tree")
set(MLN_COMPILER_API "${CMAKE_BINARY_DIR}/src/compiler/libsyn_compiler_api.so" CACHE FILEPATH
    "Compiler library used by synps build")
if(MLN_COMPILER_API STREQUAL "${CMAKE_BINARY_DIR}/compiler/libsyn_compiler_api.so")
    set(MLN_COMPILER_API "${CMAKE_BINARY_DIR}/src/compiler/libsyn_compiler_api.so" CACHE FILEPATH
        "Compiler library used by synps build" FORCE)
endif()

option(SYN_BUILD_COMPILER "Build the MyeLiN compiler service" ON)
option(SYN_BUILD_TESTS "Build SYNaPSe tests" ON)
option(SYN_ENABLE_CUDA "Enable the IREE CUDA compiler target" OFF)
option(SYN_ENABLE_VULKAN "Enable the IREE Vulkan compiler target" ON)
option(SYN_ENABLE_ROCM "Enable the IREE ROCm compiler target" OFF)
option(SYN_ENABLE_METAL "Enable the IREE Metal compiler target" OFF)

set(MLN_BUILD_RUNTIME OFF CACHE BOOL "" FORCE)
set(MLN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
if(NOT TARGET mln_sdk)
    add_subdirectory("${MLN_ROOT}" MyeLiN EXCLUDE_FROM_ALL)
endif()

add_library(synps_core SHARED
    src/build/graph.cpp
    src/build/planner.cpp
    src/cli/command.cpp
    src/cli/compiler_api.cpp
    src/cli/compiler_plan.cpp
    src/cli/manifest.cpp
    src/cli/workspace.cpp
    src/config/project.cpp
    src/config/schema.cpp
    src/config/value.cpp
    src/error.cpp
    src/package/entrypoint.cpp
    src/package/lockfile.cpp
    src/package/resolver.cpp
    src/system/hash.cpp
    src/system/process.cpp
    src/toolchain/cpp.cpp
    src/toolchain/driver.cpp
    src/toolchain/go.cpp
    src/toolchain/rust.cpp
    src/toolchain/sdk.cpp
    src/toolchain/symbols.cpp
    src/toolchain/zig.cpp
)
target_include_directories(synps_core PUBLIC include)
target_include_directories(synps_core SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tomlplusplus/include")
target_link_libraries(synps_core PUBLIC mln_package PRIVATE dl)
target_compile_features(synps_core PUBLIC cxx_std_23)
target_compile_options(synps_core PRIVATE -Wall -Wextra -Wshadow)
target_compile_definitions(synps_core PRIVATE
    SYNPS_DEFAULT_MLN_COMPILER_API="${MLN_COMPILER_API}"
    SYNPS_DEFAULT_MLN_ROOT="${MLN_ROOT}"
    SYNPS_DEFAULT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    SYNPS_INSTALLED_COMPILER_API="../${CMAKE_INSTALL_LIBDIR}/libsyn_compiler_api${CMAKE_SHARED_LIBRARY_SUFFIX}"
    SYNPS_INSTALLED_SDK="../${CMAKE_INSTALL_DATADIR}/synps/sdk"
    SYNPS_IR_INCLUDES="${MLN_ROOT}/third_party/iree/third_party/llvm-project/mlir/include|${MLN_ROOT}/third_party/iree/third_party/llvm-project/llvm/include|${MLN_ROOT}/third_party/iree/third_party/stablehlo|${MLN_ROOT}/third_party/iree/compiler/src")

add_executable(synps
    src/main.cpp
    src/cli/add.cpp
    src/cli/build.cpp
    src/cli/clean.cpp
    src/cli/graph.cpp
    src/cli/help.cpp
    src/cli/init.cpp
    src/cli/inspect.cpp
    src/cli/install.cpp
    src/cli/new.cpp
    src/cli/remove.cpp
    src/cli/update.cpp
)
target_link_libraries(synps PRIVATE synps_core)
target_compile_definitions(synps PRIVATE SYNPS_VERSION="${SYNPS_VERSION}")
set_target_properties(synps PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

if(SYN_BUILD_COMPILER)
    set(IREE_ROOT "${MLN_ROOT}/third_party/iree" CACHE PATH "IREE source tree" FORCE)
    set(IREE_BUILD_COMPILER ON CACHE BOOL "" FORCE)
    set(IREE_BUILD_RUNTIME OFF CACHE BOOL "" FORCE)
    set(IREE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
    set(IREE_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(IREE_TARGET_BACKEND_DEFAULTS OFF CACHE BOOL "" FORCE)
    set(IREE_TARGET_BACKEND_LLVM_CPU ON CACHE BOOL "" FORCE)
    set(IREE_TARGET_BACKEND_VMVX ON CACHE BOOL "" FORCE)
    set(IREE_TARGET_BACKEND_CUDA ${SYN_ENABLE_CUDA} CACHE BOOL "" FORCE)
    set(IREE_TARGET_BACKEND_VULKAN_SPIRV ${SYN_ENABLE_VULKAN} CACHE BOOL "" FORCE)
    set(IREE_TARGET_BACKEND_ROCM ${SYN_ENABLE_ROCM} CACHE BOOL "" FORCE)
    set(IREE_TARGET_BACKEND_METAL_SPIRV ${SYN_ENABLE_METAL} CACHE BOOL "" FORCE)
    add_subdirectory("${IREE_ROOT}" third_party/iree EXCLUDE_FROM_ALL)
    add_subdirectory(src/compiler)
endif()

if(SYN_BUILD_TESTS)
    enable_testing()
    add_executable(synps-tests
        tests/main.cpp
        tests/build_graph.cpp
        tests/compiler_plan.cpp
        tests/config_project.cpp
        tests/hash.cpp
        tests/lockfile.cpp
        tests/package_resolver.cpp
        tests/toolchain_symbols.cpp
    )
    if(SYN_BUILD_COMPILER)
        target_sources(synps-tests PRIVATE tests/compiler_api.cpp)
    endif()
    target_link_libraries(synps-tests PRIVATE synps_core)
    add_test(NAME synps COMMAND synps-tests)
    add_test(NAME synps_version COMMAND synps --version)
    set_tests_properties(synps_version PROPERTIES PASS_REGULAR_EXPRESSION "synps 1.0.0")
endif()

install(TARGETS synps synps_core
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(SYN_BUILD_COMPILER)
    install(TARGETS syn_compiler_api
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/syn DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY "${MLN_ROOT}/include/mln"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/synps/sdk/myelin/include)
install(DIRECTORY "${MLN_ROOT}/src/sdk"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/synps/sdk/myelin/src)
install(DIRECTORY include/syn
    DESTINATION ${CMAKE_INSTALL_DATADIR}/synps/sdk/synapse/include)
if(SYN_BUILD_COMPILER)
    foreach(include_root IN ITEMS
        "${IREE_ROOT}/third_party/llvm-project/mlir/include/"
        "${IREE_ROOT}/third_party/llvm-project/llvm/include/"
        "${IREE_ROOT}/third_party/stablehlo/"
        "${IREE_ROOT}/compiler/src/"
        "${CMAKE_BINARY_DIR}/third_party/iree/llvm-project/tools/mlir/include/"
        "${CMAKE_BINARY_DIR}/third_party/iree/llvm-project/include/"
        "${CMAKE_BINARY_DIR}/third_party/iree/llvm-external-projects/stablehlo/"
        "${CMAKE_BINARY_DIR}/third_party/iree/compiler/src/")
        install(DIRECTORY "${include_root}"
            DESTINATION ${CMAKE_INSTALL_DATADIR}/synps/sdk/compiler/include
            FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.hpp"
            PATTERN "*.inc"
            PATTERN "*.def")
    endforeach()
endif()
