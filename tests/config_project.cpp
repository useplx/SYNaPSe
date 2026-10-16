#include "test.hpp"

#include "syn/config/project.hpp"
#include "syn/error.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <variant>

namespace {
    class Directory {
        public:
            Directory() {
                path_ = std::filesystem::temp_directory_path() /
                        ("synapse-test-" + std::to_string(++sequence_));
                std::filesystem::create_directories(path_);
            }
            ~Directory() { std::filesystem::remove_all(path_); }

            const std::filesystem::path &path() const noexcept { return path_; }

        private:
            inline static std::size_t sequence_ = 0;
            std::filesystem::path path_;
    };

    void write(const std::filesystem::path &path, std::string_view contents) {
        std::ofstream output(path);
        output << contents;
    }
}

SYNPS_TEST(loads_project_without_losing_mln_configuration) {
    Directory directory;
    write(directory.path() / "synps.toml", R"toml(
[package]
name = "assistant"
version = "0.1.0"

[dependencies]
qwen = { path = "../Qwen3.5" }

[toolchains.host-cpp]
language = "cpp"
compiler = "clang++"

[toolchains.target-cpp]
language = "cpp"
compiler = "aarch64-linux-gnu-clang++"
triple = "aarch64-linux-gnu"

[toolchains.wasi-cpp]
language = "cpp"
compiler = "wasm32-wasi-clang++"
triple = "wasm32-wasi"

[build.host]
toolchain = "host-cpp"

[build.native]
cpp = "target-cpp"

[build.wasm]
toolchain = "wasi-cpp"

[package.graph]
runtime = "iree"
language = "cpp"
sources = ["src/graph.cpp"]

[capabilities]
provides = ["application"]
requires = ["generator"]

[bindings]
generator = "qwen"

[policies."model.layers.*.mlp"]
dtype = "f16"

[[policies."model.layers.*.mlp".schemes]]
scope = "constant"
lifetime = "immutable"
format = "q4k"
)toml");

    const auto project = Syn::Config::Project::load(directory.path() / "synps.toml");
    Syn::Testing::require(project.metadata().name == "assistant", "project name was not parsed");
    Syn::Testing::require(project.layout().sources == "src", "default source layout changed");
    Syn::Testing::require(
        std::holds_alternative<Syn::Config::Dependency::Path>(project.dependencies().at("qwen")),
        "path dependency was not parsed");
    Syn::Testing::require(project.build_packages().at("graph").runtime == "iree",
                          "package runtime was not parsed");
    Syn::Testing::require(project.toolchain_for("iree", "cpp") == "host-cpp",
                          "host build toolchain was not parsed");
    Syn::Testing::require(project.toolchain_for("native", "cpp") == "target-cpp",
                          "native build toolchain was not parsed");
    Syn::Testing::require(project.toolchain_for("wasm", "cpp") == "wasi-cpp",
                          "wasm build toolchain was not parsed");
    Syn::Testing::require(project.needs().at(0) == "generator",
                          "required capability was not parsed");
    Syn::Testing::require(project.bindings().at("generator") == "qwen",
                          "capability binding was not parsed");

    const auto &policies = project.document().require("policies").table();
    const auto &policy = policies.at("model.layers.*.mlp").table();
    Syn::Testing::require(policy.at("dtype").string() == "f16", "policy dtype was lost");
    const auto &scheme = policy.at("schemes").array().at(0).table();
    Syn::Testing::require(scheme.at("format").string() == "q4k",
                          "quantization format reference was lost");
}

SYNPS_TEST(expands_environment_in_compiler_policy_flags) {
    Directory directory;
    write(directory.path() / "synps.toml", R"toml(
[package]
name = "targeted"
version = "1.0.0"

[policies."model"]
dtype = "f16"

[policies."model".backend]
device = "cuda"

[policies."model".backend.flags]
iree-cuda-target = "${SYNPS_TEST_CUDA_TARGET:-sm_60}"
)toml");

    setenv("SYNPS_TEST_CUDA_TARGET", "sm_86", 1);
    const auto project = Syn::Config::Project::load(directory.path() / "synps.toml");
    unsetenv("SYNPS_TEST_CUDA_TARGET");

    const auto &backend =
        project.document().require("policies").table().at("model").require("backend").table();
    Syn::Testing::require(backend.at("flags").table().at("iree-cuda-target").string() == "sm_86",
                          "compiler policy flag retained an unexpanded environment variable");
}

SYNPS_TEST(finds_project_from_a_child_directory) {
    Directory directory;
    std::filesystem::create_directories(directory.path() / "src/deep");
    write(directory.path() / "synps.toml", "[package]\nname = \"sample\"\nversion = \"1.0.0\"\n");
    Syn::Testing::require(Syn::Config::Project::find(directory.path() / "src/deep") ==
                              directory.path() / "synps.toml",
                          "project discovery did not walk parent directories");
}

SYNPS_TEST(loads_heterogeneous_package_without_rewriting_its_entrypoint) {
    Directory directory;
    write(directory.path() / "synps.toml", R"toml(
[package]
name = "generator"
version = "1.0.0"

[entrypoint]
package = "generator"

[package.orchestration]
language = "cpp"
sources = ["orchestration.cpp"]

[package.graph]
language = "cpp"
sources = ["graph.cpp"]

[package.native]
language = "rust"
sources = ["Cargo.toml"]
)toml");
    const auto project = Syn::Config::Project::load(directory.path() / "synps.toml");
    Syn::Testing::require(project.metadata().entrypoint == "generator",
                          "entrypoint package was not retained");
    Syn::Testing::require(project.build_packages().size() == 3,
                          "entrypoint metadata was parsed as a build component");
    Syn::Testing::require(project.build_packages().at("orchestration").runtime == "wasm",
                          "orchestration requires the Wasm toolchain domain");
    Syn::Testing::require(project.build_packages().at("graph").runtime == "iree",
                          "graph requires the compiler domain without explicit exports");
    Syn::Testing::require(project.build_packages().at("native").runtime == "native",
                          "native requires the native toolchain domain");
}

SYNPS_TEST(composition_package_does_not_require_an_entrypoint) {
    Directory directory;
    write(directory.path() / "synps.toml", R"toml(
[package]
name = "codec"
version = "1.0.0"

[package.graph]
language = "cpp"
sources = ["codec.cpp"]
)toml");
    const auto project = Syn::Config::Project::load(directory.path() / "synps.toml");
    Syn::Testing::require(project.metadata().entrypoint.empty(),
                          "composition-only packages must not acquire an implicit entrypoint");
}

SYNPS_TEST(loads_canonical_build_domains_as_package_artifacts) {
    Directory directory;
    write(directory.path() / "synps.toml", R"toml(
[package]
name = "sample"
version = "1.0.0"

[build.graph]
toolchain = "cpp"
namespace = "sample"
sources = ["graph.cpp"]
exports = ["forward"]

[build.orchestration]
toolchain = "rust"
sources = ["Cargo.toml"]

[build.native]
toolchain = "zig"
sources = ["native.zig"]
)toml");

    const auto project = Syn::Config::Project::load(directory.path() / "synps.toml");
    Syn::Testing::require(project.build_packages().at("graph").runtime == "iree",
                          "graph build did not select IREE");
    Syn::Testing::require(project.build_packages().at("graph").language == "cpp",
                          "graph toolchain family did not become its language");
    Syn::Testing::require(project.build_packages().at("orchestration").runtime == "wasm",
                          "orchestration build did not select Wasm");
    Syn::Testing::require(project.build_packages().at("native").runtime == "native",
                          "native build did not select a shared-library runtime");
}

SYNPS_TEST(loads_multiple_labeled_artifacts_per_build_domain) {
    Directory directory;
    write(directory.path() / "synps.toml", R"toml(
[package]
name = "sample"
version = "1.0.0"

[build.graph.encoder]
toolchain = "cpp"
namespace = "sample"
sources = ["encoder.cpp"]
exports = ["encode"]

[build.native.tokenizer]
toolchain = "cpp"
sources = ["tokenizer.cpp"]

[build.native.search.web]
toolchain = "rust"
sources = ["Cargo.toml"]

[build.driver.qnn]
toolchain = "cpp"
sources = ["driver.cpp"]
)toml");

    const auto project = Syn::Config::Project::load(directory.path() / "synps.toml");
    Syn::Testing::require(project.build_packages().at("encoder").runtime == "iree",
                          "labeled graph did not inherit the IREE runtime");
    Syn::Testing::require(project.build_packages().at("tokenizer").runtime == "native",
                          "labeled native artifact did not inherit its runtime");
    Syn::Testing::require(project.build_packages().at("search.web").language == "rust",
                          "nested build label did not preserve its toolchain family");
    Syn::Testing::require(project.build_packages().at("qnn").runtime == "driver",
                          "labeled driver artifact did not inherit its runtime");
}

SYNPS_TEST(loads_inline_dependency_configuration) {
    Directory directory;
    write(directory.path() / "synps.toml", R"toml(
[package]
name = "consumer"
version = "1.0.0"

[dependencies]
qwen = { path = "../Qwen3.5", config = { context = 2048 } }
)toml");

    const auto project = Syn::Config::Project::load(directory.path() / "synps.toml");
    const auto &configuration = project.dependency_configurations().at("qwen").table();
    Syn::Testing::require(configuration.at("context").integer() == 2048,
                          "inline dependency configuration was not retained");
}

SYNPS_TEST(loads_hierarchical_toolchains_with_inheritance) {
    Directory directory;
    write(directory.path() / "synps.toml", R"toml(
[package]
name = "toolchain-tree"
version = "1.0.0"

[toolchain.cpp]
compiler = "clang++"
flags = ["-Wall"]

[toolchain.cpp.graph]
flags = ["-fPIC"]

[toolchain.cpp.orchestration."*"]
compiler = "wasm32-wasi-clang++"
triple = "wasm32-wasi"
)toml");
    const auto project = Syn::Config::Project::load(directory.path() / "synps.toml");
    const auto &graph = project.toolchains().at("cpp.graph");
    Syn::Testing::require(graph.compiler == "clang++", "child did not inherit compiler");
    Syn::Testing::require(graph.flags == std::vector<std::string>{"-Wall", "-fPIC"},
                          "child did not inherit flags");
    Syn::Testing::require(project.toolchains().at("cpp.orchestration.*").triple == "wasm32-wasi",
                          "labeled toolchain was not retained");
}

SYNPS_TEST(derives_package_runtime_from_declared_contract) {
    Directory directory;
    write(directory.path() / "synps.toml", R"toml(
[package]
name = "sample"
version = "1.0.0"

[package.graph]
language = "cpp"
sources = ["graph.cpp"]
exports = ["model", "intercept"]

[package.driver]
runtime = "wasm"
language = "cpp"
sources = ["driver.cpp"]

[package.provider]
language = "cpp"
sources = ["provider.cpp"]

)toml");

    const auto project = Syn::Config::Project::load(directory.path() / "synps.toml");
    Syn::Testing::require(project.build_packages().at("graph").runtime == "iree",
                          "graph export did not derive host runtime");
    Syn::Testing::require(project.build_packages().at("graph").exports ==
                              std::vector<std::string>{"model", "intercept"},
                          "package did not retain all declared exports");
    Syn::Testing::require(project.build_packages().at("driver").runtime == "wasm",
                          "declared Wasm runtime was not retained");
    Syn::Testing::require(project.build_packages().at("provider").runtime == "native",
                          "unbound provider did not derive native runtime");
}

SYNPS_TEST(rejects_legacy_recipe_section) {
    Directory directory;
    write(directory.path() / "synps.toml", R"toml(
[package]
name = "legacy"
version = "1.0.0"

[recipe.graph]
language = "cpp"
sources = ["graph.cpp"]
)toml");

    bool rejected = false;
    try {
        (void)Syn::Config::Project::load(directory.path() / "synps.toml");
    } catch (const Syn::Diagnostic &error) {
        rejected = error.error() == Syn::Error::Section::Renamed &&
                   error.message().find("[recipe.<name>]") != std::string::npos;
    }
    Syn::Testing::require(rejected, "legacy recipe section did not produce migration diagnostic");
}

SYNPS_TEST(rejects_legacy_project_section) {
    Directory directory;
    write(directory.path() / "synps.toml", R"toml(
[project]
name = "legacy"
version = "1.0.0"
)toml");

    try {
        (void)Syn::Config::Project::load(directory.path() / "synps.toml");
    } catch (const Syn::Diagnostic &error) {
        Syn::Testing::require(error.error() == Syn::Error::Section::Renamed &&
                                  error.message().find("[project]") != std::string::npos,
                              "legacy project returned the wrong diagnostic");
        return;
    }
    Syn::Testing::require(false, "legacy project section was accepted");
}

SYNPS_TEST(rejects_legacy_config_schema_section) {
    Directory directory;
    write(directory.path() / "synps.toml", R"toml(
[package]
name = "legacy"
version = "1.0.0"

[config-schema.value]
type = "uint32"
)toml");

    try {
        (void)Syn::Config::Project::load(directory.path() / "synps.toml");
    } catch (const Syn::Diagnostic &error) {
        Syn::Testing::require(error.error() == Syn::Error::Section::Renamed &&
                                  error.message().find("[config-schema]") != std::string::npos,
                              "legacy config-schema returned the wrong diagnostic");
        return;
    }
    Syn::Testing::require(false, "legacy config-schema section was accepted");
}

SYNPS_TEST(loads_git_dependency_subdirectory) {
    Directory directory;
    write(directory.path() / "synps.toml", R"toml(
[package]
name = "consumer"
version = "1.0.0"

[dependencies]
f16 = { git = "https://github.com/useplx/Quantizations.git", version = "v1.0.0", subdir = "F16" }
)toml");

    const auto project = Syn::Config::Project::load(directory.path() / "synps.toml");
    const auto &dependency =
        std::get<Syn::Config::Dependency::Git>(project.dependencies().at("f16"));
    Syn::Testing::require(dependency.subdir == "F16", "Git dependency subdir was not retained");
}
