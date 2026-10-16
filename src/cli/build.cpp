#include "syn/build/graph.hpp"
#include "syn/cli/command.hpp"
#include "syn/cli/workspace.hpp"
#include "syn/error.hpp"
#include "syn/toolchain/driver.hpp"

#include <algorithm>
#include <charconv>
#include <limits>
#include <print>
#include <thread>

namespace Syn::Cli {
    namespace {
        class Build final : public Command {
            public:
                Build()
                    : Command("build", "compila o projeto e produz o .mln", {},
                              {{"release", {}, "compila otimizado, sem simbolos de debug"},
                               {"jobs", "N", "executa ate N etapas independentes em paralelo"},
                               {"weight-memory-mib", "MiB",
                                "limit temporary memory used by weight conversion"}}) {}

                int run(const Invocation &invocation) const override {
                    const std::filesystem::path file = project_file(invocation);
                    const Package::Resolution resolution = resolve(file);
                    (void)bind_myelin(resolution);

                    Toolchain::Registry toolchains(resolution.root().project.toolchains());
                    std::size_t jobs =
                        std::clamp<std::size_t>(std::thread::hardware_concurrency(), 1, 8);
                    if (invocation.has("jobs")) {
                        const std::string_view value = invocation.option("jobs");
                        const auto parsed =
                            std::from_chars(value.data(), value.data() + value.size(), jobs);
                        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
                            jobs == 0 || jobs > 8)
                            Errors::raise(Error::Cli::Usage, signature());
                    }
                    uint64_t weight_memory_mib = 2048;
                    if (invocation.has("weight-memory-mib")) {
                        const std::string_view value = invocation.option("weight-memory-mib");
                        const auto parsed = std::from_chars(
                            value.data(), value.data() + value.size(), weight_memory_mib);
                        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
                            weight_memory_mib == 0 ||
                            weight_memory_mib > std::numeric_limits<uint64_t>::max() / 1024 / 1024)
                            Errors::raise(Error::Cli::Usage, signature());
                    }
                    Syn::Build::Planner::Options planner =
                        planner_options(file.parent_path(), invocation);
                    planner.jobs = jobs;
                    planner.weight_conversion_memory_bytes = weight_memory_mib * 1024 * 1024;
                    const Syn::Build::Planner::Plan built =
                        Syn::Build::Planner::run(resolution, toolchains, planner);
                    const Syn::Build::Executor::Result result =
                        Syn::Build::Executor(planner.cache_directory, jobs).run(built.graph);
                    std::println("{}", built.package.string());
                    std::println("{} etapas executadas, {} em cache", result.executed,
                                 result.cached);
                    return 0;
                }
        };

        const Build command;
        const int registered = [] {
            Commands::add(command);
            return 0;
        }();
    }
}
