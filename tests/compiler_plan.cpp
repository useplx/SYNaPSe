#include "test.hpp"

#include "syn/compiler/plan.hpp"
#include "syn/package/resolver.hpp"

#include <filesystem>
#include <fstream>

SYNPS_TEST(validation_policy_is_preserved_in_compiler_plan) {
    const auto root = std::filesystem::temp_directory_path() / "synapse-validation-plan-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{root};
    {
        std::ofstream manifest(root / "synps.toml");
        manifest << R"toml(
[package]
name = "application"
version = "1.0.0"

[entrypoint]
package = "application"

[validation]
compatibility = "strict"
numeric = "strict"
allow_fallback = false

[package.graph]
language = "cpp"
sources = ["graph.cpp"]
exports = ["model"]

[package.orchestration]
language = "cpp"
sources = ["orchestration.cpp"]
)toml";
    }

    const auto resolution = Syn::Package::Resolver({root / "cache"}).resolve(root / "synps.toml");
    Syn::Compiler::Artifacts artifacts;
    artifacts.modules.push_back({"wasm", "application:orchestration", root / "model.wasm"});
    const auto plan = Syn::Compiler::Plan::full(resolution, artifacts);
    const auto &validation = plan.require("validation");
    Syn::Testing::require(validation.require("compatibility").string() == "strict",
                          "compatibility enforcement was lost");
    Syn::Testing::require(validation.require("numeric").string() == "strict",
                          "numeric enforcement was lost");
    Syn::Testing::require(!validation.require("allow_fallback").boolean(),
                          "fallback prohibition was lost");
}

SYNPS_TEST(scheduling_intent_is_preserved_in_compiler_plan) {
    const auto root = std::filesystem::temp_directory_path() / "synapse-scheduling-plan-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{root};
    {
        std::ofstream manifest(root / "synps.toml");
        manifest << R"toml(
[package]
name = "application"
version = "1.0.0"

[entrypoint]
package = "application"

[scheduling]
favor = "max-concurrency"

[package.graph]
language = "cpp"
sources = ["graph.cpp"]
exports = ["model"]

[package.orchestration]
language = "cpp"
sources = ["orchestration.cpp"]
)toml";
    }

    const auto resolution = Syn::Package::Resolver({root / "cache"}).resolve(root / "synps.toml");
    Syn::Compiler::Artifacts artifacts;
    artifacts.modules.push_back({"wasm", "application:orchestration", root / "model.wasm"});
    const auto plan = Syn::Compiler::Plan::full(resolution, artifacts);
    Syn::Testing::require(plan.require("scheduling").require("favor").string() == "max-concurrency",
                          "scheduling intent was lost");
    const auto weights = Syn::Compiler::Plan::weights(
        resolution, artifacts, root / "requirements.bin", 6, 768 * 1024 * 1024);
    Syn::Testing::require(weights.require("conversion_jobs").integer() == 6,
                          "weight conversion parallelism was lost");
    Syn::Testing::require(weights.require("conversion_memory_bytes").integer() == 768 * 1024 * 1024,
                          "weight conversion memory budget was lost");
}

SYNPS_TEST(device_topology_is_preserved_in_compiler_plan) {
    const auto root = std::filesystem::temp_directory_path() / "synapse-topology-plan-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{root};
    {
        std::ofstream manifest(root / "synps.toml");
        manifest << R"toml(
[package]
name = "application"
version = "1.0.0"

[entrypoint]
package = "application"

[[topology.links]]
source = { device = "local", ordinal = 0 }
target = { device = "vulkan", ordinal = 0 }
transparent_access = true
unified_memory = false

[package.graph]
language = "cpp"
sources = ["graph.cpp"]
exports = ["model"]

[package.orchestration]
language = "cpp"
sources = ["orchestration.cpp"]
)toml";
    }

    const auto resolution = Syn::Package::Resolver({root / "cache"}).resolve(root / "synps.toml");
    Syn::Compiler::Artifacts artifacts;
    artifacts.modules.push_back({"wasm", "application:orchestration", root / "model.wasm"});
    const auto plan = Syn::Compiler::Plan::full(resolution, artifacts);
    const auto &link = plan.require("topology").require("links").array().front();
    Syn::Testing::require(link.require("source").require("device").string() == "local",
                          "topology source was lost");
    Syn::Testing::require(link.require("target").require("device").string() == "vulkan",
                          "topology target was lost");
    Syn::Testing::require(link.require("transparent_access").boolean(),
                          "topology access property was lost");
}

SYNPS_TEST(tuning_spec_is_selected_from_a_resolved_package) {
    const auto root = std::filesystem::temp_directory_path() / "synapse-tuning-plan-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "target");
    struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{root};
    {
        std::ofstream spec(root / "target/q6a.mlir");
        spec << "module attributes {transform.with_named_sequence} {}\n";
        std::ofstream dependency(root / "target/synps.toml");
        dependency << R"toml(
[package]
name = "q6a-tuning"
version = "1.0.0"
[tuning.default]
path = "q6a.mlir"
)toml";
        std::ofstream manifest(root / "synps.toml");
        manifest << R"toml(
[package]
name = "application"
version = "1.0.0"
[dependencies.q6a-tuning]
path = "target"
[myelin]
tuning = "q6a-tuning:default"
[entrypoint]
package = "application"
[package.graph]
language = "cpp"
sources = ["graph.cpp"]
exports = ["model"]
[package.orchestration]
language = "cpp"
sources = ["orchestration.cpp"]
)toml";
    }

    const auto resolution = Syn::Package::Resolver({root / "cache"}).resolve(root / "synps.toml");
    Syn::Compiler::Artifacts artifacts;
    artifacts.modules.push_back({"wasm", "application:orchestration", root / "model.wasm"});
    const auto selected = Syn::Compiler::tuning_spec(resolution);
    Syn::Testing::require(selected && *selected == root / "target/q6a.mlir",
                          "package tuning export was not resolved");
    const auto plan = Syn::Compiler::Plan::full(resolution, artifacts);
    Syn::Testing::require(plan.require("tuning_spec").string() == selected->string(),
                          "resolved tuning spec was not forwarded to compiler plan");
}

SYNPS_TEST(loading_policy_is_preserved_independently_from_codec_scheme) {
    const auto root = std::filesystem::temp_directory_path() / "synapse-loading-plan-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{root};
    {
        std::ofstream manifest(root / "synps.toml");
        manifest << R"toml(
[package]
name = "application"
version = "1.0.0"
[entrypoint]
package = "application"
[package.graph]
language = "cpp"
sources = ["graph.cpp"]
exports = ["model"]
[package.orchestration]
language = "cpp"
sources = ["orchestration.cpp"]
[policies."model"]
dtype = "f16"
[policies."model".loading]
mode = "streamed"
)toml";
    }
    const auto resolution = Syn::Package::Resolver({root / "cache"}).resolve(root / "synps.toml");
    Syn::Compiler::Artifacts artifacts;
    artifacts.modules.push_back({"wasm", "application:orchestration", root / "model.wasm"});
    const auto plan = Syn::Compiler::Plan::full(resolution, artifacts);
    Syn::Testing::require(
        plan.require("policies").require("model").require("loading").require("mode").string() ==
            "streamed",
        "loading policy was lost from resolved plan");
}

SYNPS_TEST(root_selects_dependency_entrypoint_without_repeating_physical_bindings) {
    const auto root = std::filesystem::temp_directory_path() / "synapse-composed-entrypoint-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "model");
    struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{root};
    {
        std::ofstream manifest(root / "model/synps.toml");
        manifest << R"toml(
[package]
name = "generator"
version = "1.0.0"

[package.graph]
language = "cpp"
sources = ["graph.cpp"]
exports = ["decoder"]

[package.orchestration]
language = "cpp"
sources = ["orchestration.cpp"]

)toml";
    }
    {
        std::ofstream manifest(root / "synps.toml");
        manifest << R"toml(
[package]
name = "application"
version = "1.0.0"
[entrypoint]
package = "chat"
[dependencies]
chat = { path = "model" }
)toml";
    }
    const auto resolution = Syn::Package::Resolver({root / "cache"}).resolve(root / "synps.toml");
    Syn::Compiler::Artifacts artifacts;
    artifacts.modules.push_back({"wasm", "generator:orchestration", root / "generation.wasm"});
    const auto plan = Syn::Compiler::Plan::full(resolution, artifacts);
    const auto &entries = plan.require("entrypoints").array();
    Syn::Testing::require(entries.size() == 1, "physical graph leaked as an entrypoint");
    Syn::Testing::require(entries[0].require("module").string() == "generator:orchestration",
                          "entrypoint did not derive its provider binding");
    Syn::Testing::require(entries[0].find("implementation") == nullptr,
                          "entrypoint retained a physical graph binding");
}

SYNPS_TEST(runtime_hook_derives_wasm_route_from_package_function) {
    const auto root = std::filesystem::temp_directory_path() / "synapse-runtime-hook-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{root};
    {
        std::ofstream manifest(root / "synps.toml");
        manifest << R"toml(
[package]
name = "application"
version = "1.0.0"
[entrypoint]
package = "application"
[package.graph]
language = "cpp"
sources = ["graph.cpp"]
exports = ["model"]
[package.orchestration]
language = "cpp"
sources = ["orchestration.cpp"]
[package.filter]
runtime = "wasm"
language = "cpp"
sources = ["filter.cpp"]
exports = ["forward"]
[[hooks]]
path = "chat"
function = "application:filter"
)toml";
    }
    const auto resolution = Syn::Package::Resolver({root / "cache"}).resolve(root / "synps.toml");
    Syn::Compiler::Artifacts artifacts;
    artifacts.modules.push_back({"wasm", "application:orchestration", root / "model.wasm"});
    artifacts.modules.push_back({"wasm", "application:filter", root / "filter.wasm"});
    const auto plan = Syn::Compiler::Plan::full(resolution, artifacts);
    const auto &hook = plan.require("hooks").array().front();
    Syn::Testing::require(hook.require("module").string() == "application:filter",
                          "runtime hook did not derive its Wasm module");
    Syn::Testing::require(hook.require("export").string() == "forward",
                          "runtime hook did not derive forward");
}

SYNPS_TEST(graph_hook_derives_compiler_export_from_package_function) {
    const auto root = std::filesystem::temp_directory_path() / "synapse-graph-hook-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{root};
    {
        std::ofstream manifest(root / "synps.toml");
        manifest << R"toml(
[package]
name = "application"
version = "1.0.0"
[entrypoint]
package = "application"
[package.graph]
language = "cpp"
sources = ["graph.cpp"]
exports = ["model"]
[package.orchestration]
language = "cpp"
sources = ["orchestration.cpp"]
[package.projection]
language = "cpp"
sources = ["projection.cpp"]
exports = ["projection"]
[[hooks]]
path = "model.layers.*.projection"
function = "application:projection"
)toml";
    }
    const auto resolution = Syn::Package::Resolver({root / "cache"}).resolve(root / "synps.toml");
    Syn::Compiler::Artifacts artifacts;
    artifacts.modules.push_back({"wasm", "application:orchestration", root / "model.wasm"});
    const auto plan = Syn::Compiler::Plan::graph(resolution, artifacts);
    const auto &hook = plan.require("hooks").array().front();
    Syn::Testing::require(hook.require("implementation").string() == "application:projection",
                          "graph hook did not derive its compiler export");
    Syn::Testing::require(hook.find("module") == nullptr,
                          "graph hook leaked into the runtime module table");
}

SYNPS_TEST(resolved_plan_routes_entrypoints_to_wasm_modules) {
    const auto root = std::filesystem::temp_directory_path() / "synapse-compiler-plan-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    {
        std::ofstream project(root / "synps.toml");
        project << R"toml(
[package]
name = "application"
version = "1.0.0"
[entrypoint]
package = "application"
[package.graph]
language = "cpp"
sources = ["graph.cpp"]
exports = ["decoder"]
[package.orchestration]
runtime = "wasm"
language = "cpp"
sources = ["orchestration.cpp"]
)toml";
    }

    const auto resolution = Syn::Package::Resolver({root / "cache"}).resolve(root / "synps.toml");
    Syn::Compiler::Artifacts artifacts;
    artifacts.modules.push_back({"wasm", "application:orchestration", root / "generation.wasm"});
    const Syn::Config::Value plan = Syn::Compiler::Plan::full(resolution, artifacts);
    const auto &document = plan.table();
    Syn::Testing::require(document.at("modules").array().size() == 1,
                          "runtime module was not emitted");
    Syn::Testing::require(document.at("entrypoints").array().at(0).require("module").string() ==
                              "application:orchestration",
                          "entrypoint lost its module binding");
    std::filesystem::remove_all(root);
}

SYNPS_TEST(graph_plan_only_contains_configuration_used_by_graph_facets) {
    const auto root = std::filesystem::temp_directory_path() / "synapse-facet-config-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "model");
    struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{root};
    {
        std::ofstream manifest(root / "model/synps.toml");
        manifest << R"toml(
[package]
name = "generator"
version = "1.0.0"

[package.graph]
language = "cpp"
sources = ["graph.cpp"]
exports = ["decoder"]

[package.orchestration]
language = "cpp"
sources = ["orchestration.cpp"]

[package.variables]
width = { type = "uint32", default = 64, define = "WIDTH", facets = ["graph"] }
limit = { type = "uint32", default = 8, define = "LIMIT", facets = ["orchestration"] }
)toml";
    }
    {
        std::ofstream manifest(root / "synps.toml");
        manifest << R"toml(
[package]
name = "application"
version = "1.0.0"

[entrypoint]
package = "generator"

[dependencies]
generator = { path = "model" }

[packages.generator]
width = 128
limit = 16
)toml";
    }

    const auto resolution = Syn::Package::Resolver({root / "cache"}).resolve(root / "synps.toml");
    Syn::Compiler::Artifacts artifacts;
    artifacts.modules.push_back({"wasm", "generator:orchestration", root / "generator.wasm"});
    const auto full = Syn::Compiler::Plan::full(resolution, artifacts);
    const auto graph = Syn::Compiler::Plan::graph(resolution, artifacts);
    const auto &full_config = full.require("package_config").require("generator");
    const auto &graph_config = graph.require("package_config").require("generator");

    Syn::Testing::require(full_config.find("width") != nullptr &&
                              full_config.find("limit") != nullptr,
                          "full plan discarded package configuration");
    Syn::Testing::require(graph_config.find("width") != nullptr,
                          "graph plan discarded graph configuration");
    Syn::Testing::require(graph_config.find("limit") == nullptr,
                          "orchestration configuration invalidated the graph plan");
}
