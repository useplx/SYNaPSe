#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace Syn::Build {
    struct Node {
            std::string name;
            std::vector<std::string> dependencies;
            // Garante ordem de execucao sem incorporar o fingerprint inteiro do
            // predecessor. O consumidor deve listar como input o contrato exato
            // que usa, permitindo caches por dominio.
            std::vector<std::string> order_only_dependencies;
            std::vector<std::filesystem::path> inputs;
            std::vector<std::filesystem::path> outputs;
            std::vector<std::string> configuration;
            std::function<void()> action;
    };

    class Graph {
        public:
            void add(Node node);
            const std::map<std::string, Node, std::less<>> &nodes() const noexcept {
                return nodes_;
            }
            std::vector<std::string> order() const;

        private:
            std::map<std::string, Node, std::less<>> nodes_;
    };

    class Executor {
        public:
            struct Result {
                    std::size_t executed = 0;
                    std::size_t cached = 0;
            };

            explicit Executor(std::filesystem::path cache_directory, std::size_t jobs = 1)
                : cache_directory_(std::move(cache_directory)), jobs_(jobs == 0 ? 1 : jobs) {}
            Result run(const Graph &graph) const;

        private:
            std::filesystem::path cache_directory_;
            std::size_t jobs_ = 1;
    };
} // namespace Syn::Build
