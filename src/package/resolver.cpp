#include "syn/package/resolver.hpp"

#include "syn/config/schema.hpp"
#include "syn/error.hpp"
#include "syn/system/process.hpp"

#include <algorithm>
#include <cstdlib>
#include <format>
#include <set>

namespace Syn::Package {
    namespace {
        std::string stable_key(std::string_view value) {
            std::uint64_t hash = 14695981039346656037ull;
            for (const unsigned char byte : value) {
                hash ^= byte;
                hash *= 1099511628211ull;
            }
            return std::format("{:016x}", hash);
        }

        std::string trim(std::string value) {
            while (!value.empty() && (value.back() == '\n' || value.back() == '\r'))
                value.pop_back();
            return value;
        }
    }

    class Resolver::State {
        public:
            State(Resolution &resolution, const Options &options)
                : resolution_(resolution), options_(options) {}

            std::size_t add(const std::filesystem::path &file, Source source) {
                const std::filesystem::path canonical = std::filesystem::weakly_canonical(file);
                const std::string key = canonical.string();
                if (const auto found = indices_.find(key); found != indices_.end()) {
                    if (active_.contains(key)) {
                        std::string chain;
                        for (const std::string &item : stack_) {
                            if (!chain.empty())
                                chain += " -> ";
                            chain += item;
                        }
                        chain += " -> " + key;
                        Errors::at(Error::Dependency::Cycle, key, chain);
                    }
                    return found->second;
                }

                if (!std::filesystem::is_regular_file(canonical))
                    Errors::at(Error::Package::Manifest::Missing, canonical.string());
                const std::size_t index = resolution_.packages_.size();
                indices_.emplace(key, index);
                active_.insert(key);
                stack_.push_back(key);
                resolution_.packages_.push_back(
                    {Config::Project::load(canonical), std::move(source), {}});

                const auto dependencies = resolution_.packages_[index].project.dependencies();
                for (const auto &[name, dependency] : dependencies) {
                    const std::size_t target = std::visit(
                        [&](const auto &item) {
                            return resolve_dependency(resolution_.packages_[index].project, name,
                                                      item);
                        },
                        dependency);
                    resolution_.packages_[index].dependencies.push_back({name, target});
                }
                stack_.pop_back();
                active_.erase(key);
                return index;
            }

            void bind_capabilities() {
                std::map<std::string, std::vector<std::size_t>, std::less<>> providers;
                for (std::size_t index = 0; index < resolution_.packages_.size(); ++index)
                    for (const std::string &capability :
                         resolution_.packages_[index].project.provides())
                        providers[capability].push_back(index);

                const Config::Project &root = resolution_.root().project;
                for (const Resolved::Package &package : resolution_.packages_) {
                    for (const std::string &capability : package.project.needs()) {
                        if (resolution_.bindings_.contains(capability))
                            continue;
                        const auto available = providers.find(capability);
                        if (available == providers.end() || available->second.empty())
                            Errors::at(Error::Capability::Unprovided, "capabilities.requires",
                                       package.project.metadata().name, capability);

                        if (const auto explicit_binding = root.bindings().find(capability);
                            explicit_binding != root.bindings().end()) {
                            const std::size_t selected =
                                find_package(explicit_binding->second, available->second);
                            resolution_.bindings_.emplace(capability, selected);
                        } else if (available->second.size() == 1) {
                            resolution_.bindings_.emplace(capability, available->second.front());
                        } else {
                            Errors::at(Error::Capability::Ambiguous, "bindings." + capability,
                                       capability);
                        }
                    }
                }
            }

            void resolve_configurations() {
                const Config::Project &root = resolution_.root().project;
                resolution_.configurations_.emplace(
                    root.metadata().name,
                    Config::Schema::resolve(root, nullptr, root.metadata().name));
                const Config::Value *package_values = root.document().find("packages");
                const Config::Value::Table empty;
                const Config::Value::Table &overrides =
                    package_values ? package_values->table() : empty;
                for (const Dependency::Edge &edge : resolution_.root().dependencies) {
                    const Config::Value *value = nullptr;
                    const auto inline_found = root.dependency_configurations().find(edge.name);
                    const auto table_found = overrides.find(edge.name);
                    if (inline_found != root.dependency_configurations().end() &&
                        table_found != overrides.end())
                        Errors::at(Error::Dependency::Source::Ambiguous,
                                   "dependencies." + edge.name + ".config");
                    if (table_found != overrides.end())
                        value = &table_found->second;
                    else if (inline_found != root.dependency_configurations().end())
                        value = &inline_found->second;
                    resolution_.configurations_.emplace(
                        edge.name,
                        Config::Schema::resolve(resolution_.packages_.at(edge.index).project, value,
                                                edge.name));
                }
                for (const auto &[name, value] : overrides) {
                    const bool dependency = std::any_of(
                        resolution_.root().dependencies.begin(),
                        resolution_.root().dependencies.end(),
                        [&](const Dependency::Edge &edge) { return edge.name == name; });
                    if (!dependency)
                        Errors::at(Error::Configuration::Unknown::Dependency, "packages." + name);
                }
            }

        private:
            Resolution &resolution_;
            const Options &options_;
            std::map<std::string, std::size_t, std::less<>> indices_;
            std::set<std::string, std::less<>> active_;
            std::vector<std::string> stack_;

            std::size_t find_package(const std::string &name,
                                     const std::vector<std::size_t> &candidates) const {
                for (const std::size_t index : candidates)
                    if (resolution_.packages_[index].project.metadata().name == name)
                        return index;
                for (const Dependency::Edge &edge : resolution_.root().dependencies)
                    if (edge.name == name && std::find(candidates.begin(), candidates.end(),
                                                       edge.index) != candidates.end())
                        return edge.index;
                Errors::raise(Error::Capability::Not::Provided, name);
            }

            std::size_t resolve_dependency(const Config::Project &owner, const std::string &,
                                           const Config::Dependency::Path &dependency) {
                const auto root = std::filesystem::weakly_canonical(owner.root() / dependency.path);
                return add(root / "synps.toml", {Source::Kind::Path, root.string(), {}});
            }

            std::size_t resolve_dependency(const Config::Project &, const std::string &name,
                                           const Config::Dependency::Registry &) {
                Errors::at(Error::Registry::Unavailable, "dependencies." + name, name);
            }

            std::size_t resolve_dependency(const Config::Project &, const std::string &name,
                                           const Config::Dependency::Git &dependency) {
                const std::string selector =
                    !dependency.revision.empty()
                        ? dependency.revision
                        : (!dependency.version.empty() ? dependency.version : "HEAD");
                const std::filesystem::path directory =
                    options_.cache_directory / "git" / stable_key(dependency.url + "\n" + selector);
                std::filesystem::create_directories(directory.parent_path());
                if (!std::filesystem::exists(directory / ".git")) {
                    System::run_checked(
                        {"git",
                         {"clone", "--no-checkout", dependency.url, directory.string()},
                         {},
                         {}},
                        "cloning package `" + name + "`");
                } else if (options_.update_git) {
                    System::run_checked({"git", {"fetch", "--tags", "--prune"}, directory, {}},
                                        "updating package `" + name + "`");
                }
                System::run_checked({"git", {"checkout", "--detach", selector}, directory, {}},
                                    "checking out package `" + name + "`");
                const std::string revision =
                    trim(System::run_checked({"git", {"rev-parse", "HEAD"}, directory, {}},
                                             "resolving package revision")
                             .output);
                return add(directory / "synps.toml", {Source::Kind::Git, dependency.url, revision});
            }
    };

    Resolver::Resolver(Options options) : options_(std::move(options)) {
        if (options_.cache_directory.empty()) {
            if (const char *home = std::getenv("HOME"))
                options_.cache_directory = std::filesystem::path(home) / ".cache/synapse";
            else
                options_.cache_directory = ".synps-cache";
        }
    }

    Resolution Resolver::resolve(const std::filesystem::path &project_file) {
        Resolution result;
        State state(result, options_);
        result.root_ =
            state.add(project_file,
                      {Source::Kind::Root, std::filesystem::absolute(project_file).string(), {}});
        state.bind_capabilities();
        state.resolve_configurations();
        (void)result.entrypoint();
        return result;
    }
}
