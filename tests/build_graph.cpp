#include "test.hpp"

#include "syn/build/graph.hpp"
#include "syn/build/planner.hpp"
#include "syn/error.hpp"
#include "syn/toolchain/abi.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

SYNPS_TEST(static_library_is_not_treated_as_a_runtime_provider) {
    const auto requirements = Syn::Toolchain::Abi::requirements(
        Syn::Toolchain::Runtime::Native, Syn::Toolchain::Output::StaticLibrary);
    Syn::Testing::require(requirements.required_symbols.empty(),
                          "static input unexpectedly requires a runtime provider symbol");
    Syn::Testing::require(requirements.defines.empty(),
                          "static input unexpectedly receives provider build defines");
    Syn::Testing::require(requirements.position_independent,
                          "static input linked into a provider must remain relocatable");
}

SYNPS_TEST(compiler_headers_are_inputs_of_cpp_plugins) {
    const auto source = std::filesystem::path(__FILE__).parent_path().parent_path();
    const auto root = std::filesystem::temp_directory_path() / "synps-plugin-headers-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{root};
    std::ofstream(root / "graph.cpp") << "// Planner-only fixture.\n";
    std::ofstream(root / "orchestration.cpp") << "// Planner-only fixture.\n";
    std::ofstream(root / "synps.toml") << R"toml(
[package]
name = "fixture"
version = "0.1.0"
[entrypoint]
package = "fixture"
[package.graph]
language = "cpp"
sources = ["graph.cpp"]
exports = ["model"]
[package.orchestration]
language = "cpp"
sources = ["orchestration.cpp"]
[toolchains.fixture]
language = "cpp"
compiler = "c++"
[build.wasm]
toolchain = "fixture"
)toml";
    Syn::Toolchain::Myelin::set_root(source.parent_path() / "MyeLiN");
    const auto resolution = Syn::Package::Resolver({root / "cache"}).resolve(root / "synps.toml");
    Syn::Toolchain::Registry toolchains(resolution.root().project.toolchains());
    const auto plan = Syn::Build::Planner::run(resolution, toolchains,
                                               {
                                                   .build_directory = root / "build",
                                                   .cache_directory = root / "cache",
                                               });
    const auto &inputs = plan.graph.nodes().at("fixture:graph").inputs;
    Syn::Testing::require(std::ranges::find(inputs, source / "include" / "syn") != inputs.end(),
                          "implicitly included Compiler headers do not invalidate plugin cache");
    const auto &configuration = plan.graph.nodes().at("fixture:graph").configuration;
    Syn::Testing::require(
        std::ranges::find(configuration, "DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1") !=
            configuration.end(),
        "implicit IREE ABI definitions do not invalidate plugin cache");
}

SYNPS_TEST(build_graph_reuses_unchanged_outputs) {
    const auto root = std::filesystem::temp_directory_path() / "synps-build-graph-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    {
        std::ofstream input(root / "input");
        input << "source";
    }
    std::size_t executions = 0;
    Syn::Build::Graph graph;
    graph.add({
        .name = "compile",
        .dependencies = {},
        .inputs = {root / "input"},
        .outputs = {root / "output"},
        .configuration = {"c++20", "host"},
        .action =
            [&] {
                ++executions;
                std::ofstream(root / "output") << "artifact";
            },
    });
    const Syn::Build::Executor executor(root / "cache");
    const auto first = executor.run(graph);
    const auto second = executor.run(graph);
    Syn::Testing::require(first.executed == 1 && second.cached == 1,
                          "unchanged build node was not cached");
    Syn::Testing::require(executions == 1, "cached build action was executed again");
    std::filesystem::remove_all(root);
}

SYNPS_TEST(build_graph_detects_cycles) {
    Syn::Build::Graph graph;
    graph.add({.name = "one",
               .dependencies = {"two"},
               .inputs = {},
               .outputs = {},
               .configuration = {},
               .action = [] {}});
    graph.add({.name = "two",
               .dependencies = {"one"},
               .inputs = {},
               .outputs = {},
               .configuration = {},
               .action = [] {}});
    try {
        (void)graph.order();
        Syn::Testing::require(false, "build cycle was accepted");
    } catch (const Syn::Diagnostic &error) {
        Syn::Testing::require(error.error() == Syn::Error::Graph::Cycle,
                              "build cycle returned the wrong diagnostic");
    }
}

SYNPS_TEST(build_graph_runs_independent_nodes_in_parallel) {
    const auto root = std::filesystem::temp_directory_path() / "synps-parallel-graph-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    std::atomic<int> active = 0;
    std::atomic<int> peak = 0;
    Syn::Build::Graph graph;
    const auto independent = [&](std::string name) {
        const std::filesystem::path output = root / name;
        graph.add({
            .name = name,
            .outputs = {output},
            .action =
                [&, output] {
                    const int current = active.fetch_add(1) + 1;
                    int observed = peak.load();
                    while (observed < current && !peak.compare_exchange_weak(observed, current)) {
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(40));
                    std::ofstream(output) << "done";
                    active.fetch_sub(1);
                },
        });
    };
    independent("one");
    independent("two");
    graph.add({
        .name = "join",
        .dependencies = {"one", "two"},
        .outputs = {root / "join"},
        .action =
            [&] {
                Syn::Testing::require(active.load() == 0, "dependent node started too early");
                std::ofstream(root / "join") << "done";
            },
    });

    const auto result = Syn::Build::Executor(root / "cache", 2).run(graph);
    Syn::Testing::require(peak.load() == 2, "independent build nodes did not overlap");
    Syn::Testing::require(result.executed == 3, "parallel graph lost a build node");
    std::filesystem::remove_all(root);
}

SYNPS_TEST(runtime_change_reuses_graph_and_weights_core) {
    const auto root = std::filesystem::temp_directory_path() / "synapse-domain-cache-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    std::ofstream(root / "graph.cpp") << "graph-v1";
    std::ofstream(root / "runtime.cpp") << "runtime-v1";

    std::size_t graph_runs = 0;
    std::size_t runtime_runs = 0;
    std::size_t package_runs = 0;
    Syn::Build::Graph graph;
    graph.add({
        .name = "myelin:core",
        .inputs = {root / "graph.cpp"},
        .outputs = {root / "core"},
        .action =
            [&] {
                ++graph_runs;
                std::ofstream(root / "core") << "vmfb+weights";
            },
    });
    graph.add({
        .name = "runtime",
        .inputs = {root / "runtime.cpp"},
        .outputs = {root / "runtime.wasm"},
        .action =
            [&] {
                ++runtime_runs;
                std::ofstream(root / "runtime.wasm") << "wasm";
            },
    });
    graph.add({
        .name = "package",
        .dependencies = {"myelin:core", "runtime"},
        .outputs = {root / "application.mln"},
        .action =
            [&] {
                ++package_runs;
                std::ofstream(root / "application.mln") << "package";
            },
    });

    const Syn::Build::Executor executor(root / "cache");
    (void)executor.run(graph);
    std::ofstream(root / "runtime.cpp", std::ios::trunc) << "runtime-v2";
    const auto changed = executor.run(graph);
    Syn::Testing::require(graph_runs == 1 && runtime_runs == 2 && package_runs == 2,
                          "runtime change rebuilt the graph/weights core");
    Syn::Testing::require(changed.cached == 1 && changed.executed == 2,
                          "domain cache did not isolate runtime from graph compilation");
    std::filesystem::remove_all(root);
}

SYNPS_TEST(graph_change_reuses_parameters_when_contract_is_unchanged) {
    const auto root =
        std::filesystem::temp_directory_path() / "synapse-parameter-contract-cache-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    std::ofstream(root / "graph.cpp") << "graph-v1";

    std::size_t graph_runs = 0;
    std::size_t parameter_runs = 0;
    Syn::Build::Graph graph;
    graph.add({
        .name = "graph",
        .inputs = {root / "graph.cpp"},
        .outputs = {root / "graph.cache", root / "weights.requirements"},
        .action =
            [&] {
                ++graph_runs;
                std::ofstream(root / "graph.cache") << "vmfb-" << graph_runs;
                std::ofstream(root / "weights.requirements") << "same-contract";
            },
    });
    graph.add({
        .name = "parameters",
        .order_only_dependencies = {"graph"},
        .inputs = {root / "weights.requirements"},
        .outputs = {root / "parameters.cache"},
        .action =
            [&] {
                ++parameter_runs;
                std::ofstream(root / "parameters.cache") << "parameters";
            },
    });

    const Syn::Build::Executor executor(root / "cache");
    (void)executor.run(graph);
    std::ofstream(root / "graph.cpp", std::ios::trunc) << "graph-v2";
    const auto changed = executor.run(graph);
    Syn::Testing::require(graph_runs == 2 && parameter_runs == 1,
                          "graph-only change reconverted parameters");
    Syn::Testing::require(changed.executed == 1 && changed.cached == 1,
                          "order-only dependency polluted cache key");
    std::filesystem::remove_all(root);
}

SYNPS_TEST(compiler_api_content_invalidates_compiler_stages) {
    const auto root = std::filesystem::temp_directory_path() / "synapse-compiler-api-cache-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    std::ofstream(root / "libmyelin-compiler.so") << "compiler-v1";

    std::size_t graph_runs = 0;
    Syn::Build::Graph graph;
    graph.add({
        .name = "myelin:graph",
        .inputs = {root / "libmyelin-compiler.so"},
        .outputs = {root / "graph.cache"},
        .action =
            [&] {
                ++graph_runs;
                std::ofstream(root / "graph.cache") << "graph";
            },
    });

    const Syn::Build::Executor executor(root / "cache");
    (void)executor.run(graph);
    std::ofstream(root / "libmyelin-compiler.so", std::ios::trunc) << "compiler-v2";
    const auto changed = executor.run(graph);
    Syn::Testing::require(graph_runs == 2 && changed.executed == 1,
                          "compiler API replacement reused stale output");
    std::filesystem::remove_all(root);
}

SYNPS_TEST(build_graph_separates_configuration_fields) {
    const auto root = std::filesystem::temp_directory_path() / "synapse-fingerprint-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    std::size_t executions = 0;
    const auto build = [&](std::vector<std::string> configuration) {
        Syn::Build::Graph graph;
        graph.add({
            .name = "compile",
            .dependencies = {},
            .inputs = {},
            .outputs = {root / "output"},
            .configuration = std::move(configuration),
            .action =
                [&] {
                    ++executions;
                    std::ofstream(root / "output") << "artifact";
                },
        });
        return Syn::Build::Executor(root / "cache").run(graph);
    };

    build({"a\nconfigb"});
    const auto second = build({"a", "b"});
    Syn::Testing::require(executions == 2 && second.executed == 1,
                          "distinct configurations shared a fingerprint");
    std::filesystem::remove_all(root);
}
