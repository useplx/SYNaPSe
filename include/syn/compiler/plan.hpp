#pragma once

#include "syn/config/value.hpp"
#include "syn/package/resolver.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Syn::Compiler {
    namespace Runtime {
        struct Module {
                std::string runtime;
                std::string name;
                std::filesystem::path path;
        };
    }

    struct Artifacts {
            std::vector<Runtime::Module> modules;
            std::vector<std::pair<std::string, std::filesystem::path>> assets;

            std::vector<std::filesystem::path> plugins;

            std::vector<std::filesystem::path> weights;
    };

    namespace Plan {
        Config::Value::Array entrypoints(const Package::Resolution &resolution);
        Config::Value::Array hooks(const Package::Resolution &resolution);
        Config::Value full(const Package::Resolution &resolution, const Artifacts &artifacts);

        Config::Value graph(const Package::Resolution &resolution, const Artifacts &artifacts,
                            const std::filesystem::path &requirements_output = {},
                            const std::filesystem::path &parameters_package = {},
                            std::string_view graph_implementation = {});

        Config::Value weights(const Package::Resolution &resolution, const Artifacts &artifacts,
                              const std::filesystem::path &requirements,
                              std::size_t conversion_jobs = 1,
                              uint64_t conversion_memory_bytes = uint64_t{2} * 1024 * 1024 * 1024);

        Config::Value requirements(const std::filesystem::path &graph_cache);

        Config::Value
        merge_graphs(const std::vector<std::pair<std::string, std::filesystem::path>> &graphs);

        Config::Value package(const Package::Resolution &resolution, const Artifacts &artifacts,
                              const std::filesystem::path &graph_cache,
                              const std::filesystem::path &parameters_cache);

        void validate(const Config::Value &plan);
    }

    Config::Value weights_fingerprint(const Package::Resolution &resolution);

    std::optional<std::filesystem::path> tuning_spec(const Package::Resolution &resolution);
}
