#pragma once

#include "syn/build/graph.hpp"
#include "syn/compiler/plan.hpp"
#include "syn/package/resolver.hpp"
#include "syn/toolchain/driver.hpp"

#include <filesystem>
#include <cstddef>
#include <cstdint>
#include <string>

namespace Syn::Build {
    namespace Planner {
        struct Options {
                std::filesystem::path build_directory;
                std::filesystem::path cache_directory;
                bool optimize = false;
                std::size_t jobs = 1;
                uint64_t weight_conversion_memory_bytes = uint64_t{2} * 1024 * 1024 * 1024;

                // Biblioteca do mln_compiler_api. Resolvida a partir do package
                // `myelin`; explicita aqui para o planner nao precisar conhecer o
                // layout interno de outro package.
                std::filesystem::path compiler_api;
        };

        // O DAG e o resultado, junto com os artefatos que o no final precisa
        // referenciar. Manter os dois juntos evita recalcular caminhos que o
        // planner ja decidiu.
        struct Plan {
                Graph graph;
                Compiler::Artifacts artifacts;
                std::filesystem::path package;
        };

        // Monta o grafo de build a partir da resolucao. Um no por build
        // package, mais o no final de empacotamento, que depende de todos os
        // outros.
        //
        // Os tres runtimes saem daqui como nos distintos e nao
        // intercambiaveis: um plugin de grafo compila para a maquina de build,
        // um provider nativo para o dispositivo alvo, os modulos portateis
        // para Wasm, e o grafo e emitido pelo Myelin dentro do no de
        // empacotamento.
        Plan run(const Package::Resolution &resolution, const Toolchain::Registry &toolchains,
                 const Options &options);
    } // namespace Planner
} // namespace Syn::Build
