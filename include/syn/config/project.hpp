#pragma once

#include "syn/config/value.hpp"

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace Syn::Config {
    struct Metadata {
            std::string name;
            std::string version;
            std::string description;

            std::string entrypoint;

            std::vector<std::filesystem::path> include_directories;
    };

    struct Layout {
            std::filesystem::path sources = "src";
            std::filesystem::path includes = "include";
            std::filesystem::path assets = "assets";
            std::filesystem::path tests = "tests";
            std::filesystem::path examples = "examples";
    };

    namespace Dependency {
        struct Path {
                std::filesystem::path path;
        };
        struct Git {
                std::string url;
                std::string revision;
                std::string version;
                std::filesystem::path subdir;
        };
        struct Registry {
                std::string version;
        };
        using Any = std::variant<Path, Git, Registry>;
    }

    struct Toolchain {
            std::string language;
            std::filesystem::path compiler;
            std::filesystem::path linker;
            std::filesystem::path archiver;
            std::string triple;
            std::vector<std::string> flags;
    };

    namespace Build {
        struct Domain {
                std::string toolchain;
                std::map<std::string, std::string, std::less<>> languages;
        };

        struct Package {
                std::string name;
                std::string runtime;
                bool runtime_inferred = false;
                std::string graph_namespace;
                std::string language;
                std::string toolchain;
                std::vector<std::filesystem::path> sources;
                std::vector<std::filesystem::path> includes;
                std::vector<std::string> libraries;
                std::vector<std::string> defines;
                std::vector<std::filesystem::path> library_directories;
                std::string artifact;
                std::vector<std::string> exports;
                std::vector<std::string> uses;
                std::vector<std::string> extends;
        };
    }

    class Project {
        public:
            static Project load(const std::filesystem::path &path);
            static std::filesystem::path find(const std::filesystem::path &start);

            const std::filesystem::path &file() const noexcept { return file_; }
            const std::filesystem::path &root() const noexcept { return root_; }
            const Metadata &metadata() const noexcept { return metadata_; }
            const Layout &layout() const noexcept { return layout_; }
            const std::map<std::string, Dependency::Any, std::less<>> &
            dependencies() const noexcept {
                return dependencies_;
            }
            const std::map<std::string, Value, std::less<>> &
            dependency_configurations() const noexcept {
                return dependency_configurations_;
            }
            const std::map<std::string, Toolchain, std::less<>> &toolchains() const noexcept {
                return toolchains_;
            }
            const std::map<std::string, Build::Domain, std::less<>> &
            build_domains() const noexcept {
                return build_domains_;
            }
            std::string_view toolchain_for(std::string_view runtime,
                                           std::string_view language) const noexcept;
            const std::map<std::string, Build::Package, std::less<>> &
            build_packages() const noexcept {
                return build_packages_;
            }
            const std::vector<std::string> &provides() const noexcept { return provides_; }
            const std::vector<std::string> &needs() const noexcept { return needs_; }
            const std::map<std::string, std::string, std::less<>> &bindings() const noexcept {
                return bindings_;
            }
            const Value &document() const noexcept { return document_; }

        private:
            std::filesystem::path file_;
            std::filesystem::path root_;
            Metadata metadata_;
            Layout layout_;
            std::map<std::string, Dependency::Any, std::less<>> dependencies_;
            std::map<std::string, Value, std::less<>> dependency_configurations_;
            std::map<std::string, Toolchain, std::less<>> toolchains_;
            std::map<std::string, Build::Domain, std::less<>> build_domains_;
            std::map<std::string, Build::Package, std::less<>> build_packages_;
            std::vector<std::string> provides_;
            std::vector<std::string> needs_;
            std::map<std::string, std::string, std::less<>> bindings_;
            Value document_;
    };
}
