#include "test.hpp"

#include "syn/error.hpp"
#include "syn/package/resolver.hpp"

#include <filesystem>
#include <fstream>
#include <string_view>

namespace {
    class Tree {
        public:
            Tree() {
                root_ = std::filesystem::temp_directory_path() /
                        ("synapse-resolver-test-" + std::to_string(++sequence_));
                std::filesystem::create_directories(root_);
            }
            ~Tree() { std::filesystem::remove_all(root_); }

            std::filesystem::path add(std::string_view name, std::string_view contents) const {
                const auto directory = root_ / name;
                std::filesystem::create_directories(directory);
                std::ofstream output(directory / "synps.toml");
                output << contents;
                return directory;
            }
            const std::filesystem::path &root() const noexcept { return root_; }

        private:
            inline static std::size_t sequence_ = 0;
            std::filesystem::path root_;
    };
}

SYNPS_TEST(resolves_entrypoint_package_alias) {
    Tree tree;
    tree.add("model", R"toml(
[package]
name = "qwen3-vl"
version = "1.0.0"
)toml");
    const auto application = tree.add("application", R"toml(
[package]
name = "sensix"
version = "1.0.0"
[entrypoint]
package = "generator"
[dependencies]
generator = { path = "../model" }
)toml");
    const auto resolution =
        Syn::Package::Resolver({tree.root() / "cache"}).resolve(application / "synps.toml");
    const auto entrypoint = resolution.entrypoint();
    Syn::Testing::require(entrypoint.has_value(), "entrypoint was not resolved");
    Syn::Testing::require(entrypoint->path == "qwen3-vl",
                          "entrypoint did not use the selected package name");
    Syn::Testing::require(resolution.packages().at(entrypoint->package).project.metadata().name ==
                              "qwen3-vl",
                          "entrypoint resolved to the consumer instead of its owner");
}

SYNPS_TEST(rejects_unknown_entrypoint_package) {
    Tree tree;
    tree.add("codec", "[package]\nname = 'codec'\nversion = '1.0.0'\n");
    const auto application = tree.add("application", R"toml(
[package]
name = "application"
version = "1.0.0"
[entrypoint]
package = "missing"
[dependencies]
codec = { path = "../codec" }
)toml");
    try {
        (void)Syn::Package::Resolver({tree.root() / "cache"}).resolve(application / "synps.toml");
    } catch (const Syn::Diagnostic &error) {
        Syn::Testing::require(error.error() == Syn::Error::Entrypoint::Dependency::Unknown,
                              "unknown entrypoint package returned the wrong diagnostic");
        return;
    }
    Syn::Testing::require(false, "unknown entrypoint package was accepted");
}

SYNPS_TEST(resolves_transitive_packages_and_capabilities) {
    Tree tree;
    tree.add("tokenizer", R"toml(
[package]
name = "tokenizer"
version = "1.0.0"
)toml");
    tree.add("qwen", R"toml(
[package]
name = "qwen3-vl"
version = "1.0.0"
[dependencies]
tokenizer = { path = "../tokenizer" }
[capabilities]
provides = ["generator"]
)toml");
    const auto application = tree.add("application", R"toml(
[package]
name = "assistant"
version = "0.1.0"
[dependencies]
qwen = { path = "../qwen" }
[capabilities]
requires = ["generator"]
[bindings]
generator = "qwen"
)toml");

    const auto resolution =
        Syn::Package::Resolver({tree.root() / "cache"}).resolve(application / "synps.toml");
    Syn::Testing::require(resolution.packages().size() == 3, "transitive package was not resolved");
    const auto provider = resolution.bindings().at("generator");
    Syn::Testing::require(resolution.packages().at(provider).project.metadata().name == "qwen3-vl",
                          "explicit capability binding selected the wrong package");
}

SYNPS_TEST(rejects_dependency_cycles) {
    Tree tree;
    const auto first = tree.add("first", R"toml(
[package]
name = "first"
version = "1.0.0"
[dependencies]
second = { path = "../second" }
)toml");
    tree.add("second", R"toml(
[package]
name = "second"
version = "1.0.0"
[dependencies]
first = { path = "../first" }
)toml");

    try {
        Syn::Package::Resolver({tree.root() / "cache"}).resolve(first / "synps.toml");
        Syn::Testing::require(false, "dependency cycle was accepted");
    } catch (const Syn::Diagnostic &error) {
        Syn::Testing::require(error.error() == Syn::Error::Dependency::Cycle,
                              "dependency cycle returned the wrong diagnostic");
    }
}

SYNPS_TEST(rejects_ambiguous_capability_provider) {
    Tree tree;
    tree.add("one", R"toml(
[package]
name = "one"
version = "1.0.0"
[capabilities]
provides = ["embedder"]
)toml");
    tree.add("two", R"toml(
[package]
name = "two"
version = "1.0.0"
[capabilities]
provides = ["embedder"]
)toml");
    const auto application = tree.add("application", R"toml(
[package]
name = "application"
version = "1.0.0"
[dependencies]
one = { path = "../one" }
two = { path = "../two" }
[capabilities]
requires = ["embedder"]
)toml");

    try {
        Syn::Package::Resolver({tree.root() / "cache"}).resolve(application / "synps.toml");
        Syn::Testing::require(false, "ambiguous capability was accepted");
    } catch (const Syn::Diagnostic &error) {
        Syn::Testing::require(error.error() == Syn::Error::Capability::Ambiguous,
                              "ambiguity returned the wrong diagnostic");
    }
}

SYNPS_TEST(validates_package_configuration_before_build) {
    Tree tree;
    tree.add("rag", R"toml(
[package]
name = "rag"
version = "1.0.0"
[package.variables.top_k]
type = "uint32"
default = 5
)toml");
    const auto application = tree.add("application", R"toml(
[package]
name = "application"
version = "1.0.0"
[dependencies]
rag = { path = "../rag" }
[packages.rag]
top_k = "banana"
)toml");

    try {
        Syn::Package::Resolver({tree.root() / "cache"}).resolve(application / "synps.toml");
        Syn::Testing::require(false, "invalid package configuration was accepted");
    } catch (const Syn::Diagnostic &error) {
        Syn::Testing::require(error.location() == "rag.top_k",
                              "configuration diagnostic lost its package field");
        Syn::Testing::require(
            std::string(error.what()).find("expected uint32, received string") != std::string::npos,
            "configuration diagnostic did not report expected and received types");
    }
}

SYNPS_TEST(resolves_root_package_variable_defaults) {
    Tree tree;
    const auto application = tree.add("application", R"toml(
[package]
name = "application"
version = "1.0.0"

[package.variables]
width = { type = "uint32", default = 128, define = "APP_WIDTH", facets = ["graph"] }
enabled = { type = "bool", default = true, define = "APP_ENABLED", facets = ["orchestration"] }
)toml");

    const auto resolution =
        Syn::Package::Resolver({tree.root() / "cache"}).resolve(application / "synps.toml");
    const auto found = resolution.configurations().find("application");
    Syn::Testing::require(found != resolution.configurations().end(),
                          "root package configuration was not resolved");
    Syn::Testing::require(found->second.require("width").integer() == 128,
                          "root package integer default was lost");
    Syn::Testing::require(found->second.require("enabled").boolean(),
                          "root package boolean default was lost");
}
