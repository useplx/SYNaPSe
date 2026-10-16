#pragma once

#include "syn/config/project.hpp"

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace Syn::Package {
    class Resolution;

    struct Source {
            enum class Kind { Root, Path, Git };

            Kind kind = Kind::Path;
            std::string location;
            std::string revision;
    };

    namespace Dependency {
        struct Edge {
                std::string name;
                std::size_t index = 0;
        };
    }

    namespace Resolved {
        struct Package {
                Config::Project project;
                Source source;
                std::vector<Dependency::Edge> dependencies;
        };
    }

    class Resolver {
        public:
            struct Options {
                    std::filesystem::path cache_directory;
                    bool update_git = false;
            };

            class State;

            explicit Resolver(Options options);
            Resolution resolve(const std::filesystem::path &project_file);

        private:
            Options options_;
    };

    class Resolution {
        public:
            struct Entrypoint {
                    std::size_t package;
                    std::string path;
            };

            std::optional<Entrypoint> entrypoint() const;

            const std::vector<Resolved::Package> &packages() const noexcept { return packages_; }
            const Resolved::Package &root() const { return packages_.at(root_); }
            const std::map<std::string, std::size_t, std::less<>> &bindings() const noexcept {
                return bindings_;
            }
            const std::map<std::string, Config::Value, std::less<>> &
            configurations() const noexcept {
                return configurations_;
            }

        private:
            friend class Resolver;
            friend class Resolver::State;
            std::vector<Resolved::Package> packages_;
            std::size_t root_ = 0;
            std::map<std::string, std::size_t, std::less<>> bindings_;
            std::map<std::string, Config::Value, std::less<>> configurations_;
    };
}
