#include "syn/compiler/plan.hpp"

#include "syn/error.hpp"

#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>

namespace Syn::Compiler {
    std::optional<std::filesystem::path> tuning_spec(const Package::Resolution &resolution) {
        const Config::Value *myelin = resolution.root().project.document().find("myelin");
        const Config::Value *selected =
            myelin && myelin->is_table() ? myelin->find("tuning") : nullptr;
        if (!selected)
            return std::nullopt;
        const std::string reference(selected->string());
        const std::size_t separator = reference.find(':');
        if (separator == std::string::npos || separator == 0 || separator + 1 == reference.size())
            throw std::runtime_error("myelin.tuning must be 'package:spec': " + reference);
        const std::string owner = reference.substr(0, separator);
        const std::string name = reference.substr(separator + 1);
        const auto package = std::ranges::find_if(
            resolution.packages(), [&](const Package::Resolved::Package &candidate) {
                return candidate.project.metadata().name == owner;
            });
        if (package == resolution.packages().end())
            throw std::runtime_error("tuning package is not resolved: " + owner);
        const Config::Value *exports = package->project.document().find("tuning");
        const Config::Value *spec = exports && exports->is_table() ? exports->find(name) : nullptr;
        const Config::Value *path = spec && spec->is_table() ? spec->find("path") : nullptr;
        if (!path)
            throw std::runtime_error("tuning spec is not exported: " + reference);
        const std::filesystem::path declared(path->string());
        return declared.is_absolute() ? declared : package->project.root() / declared;
    }

    namespace {
        using Config::Value;

        const Value *section(const Value &document, std::string_view name) {
            const Value *found = document.find(name);
            return found && found->is_table() ? found : nullptr;
        }

        const Value *manifest_section(const Package::Resolution &resolution,
                                      std::string_view name) {
            return section(resolution.root().project.document(), name);
        }

        Value::Array manifest_array(const Package::Resolution &resolution, std::string_view name) {
            const Value *found = resolution.root().project.document().find(name);
            if (!found)
                return {};
            if (!found->is_array())
                Errors::at(Error::Compiler::Plan::Manifest::Array::Expected, std::string(name));
            return found->array();
        }

        Value::Array compiler_array(const Package::Resolution &resolution, std::string_view name) {
            const Value *myelin = section(resolution.root().project.document(), "myelin");
            if (!myelin)
                return {};
            const Value *found = myelin->find(name);
            if (!found)
                return {};
            if (!found->is_array())
                Errors::at(Error::Compiler::Plan::Options::Array::Expected,
                           "myelin." + std::string(name));
            return found->array();
        }

        Value graph_configurations(const Package::Resolution &resolution) {
            Value::Table configurations;
            for (const auto &[name, configured] : resolution.configurations()) {
                const auto package = std::ranges::find_if(
                    resolution.packages(), [&name](const Package::Resolved::Package &candidate) {
                        return candidate.project.metadata().name == name;
                    });
                if (package == resolution.packages().end())
                    continue;

                const Value *declaration = package->project.document().find("package");
                const Value *schema = declaration ? declaration->find("variables") : nullptr;
                if (schema == nullptr || !schema->is_table()) {
                    configurations.emplace(name, configured);
                    continue;
                }

                Value::Table fields;
                for (const auto &[field, value] : configured.table()) {
                    const Value *rule = schema->find(field);
                    const Value *facets = rule ? rule->find("facets") : nullptr;
                    if (facets == nullptr) {
                        fields.emplace(field, value);
                        continue;
                    }

                    const bool used_by_graph =
                        std::ranges::any_of(facets->array(), [&package](const Value &candidate) {
                            const auto build =
                                package->project.build_packages().find(candidate.string());
                            return build != package->project.build_packages().end() &&
                                   build->second.runtime == "iree";
                        });
                    if (used_by_graph)
                        fields.emplace(field, value);
                }
                if (!fields.empty())
                    configurations.emplace(name, Value(std::move(fields)));
            }
            return Value(std::move(configurations));
        }

        Value file_entry(const std::string &name, const std::filesystem::path &path) {
            Value::Table entry;
            entry.emplace("name", Value(name));
            entry.emplace("path", Value(path.string()));
            return Value(std::move(entry));
        }

        Value derived_entrypoint(const Package::Resolution &resolution,
                                 const Package::Resolution::Entrypoint &selected) {
            const auto &package = resolution.packages().at(selected.package);
            const auto &project = package.project;
            const std::string &package_name = project.metadata().name;

            Value::Table entry;
            entry.emplace("name", Value(selected.path));
            const auto orchestration = project.build_packages().find("orchestration");
            if (orchestration != project.build_packages().end() &&
                orchestration->second.runtime == "wasm") {
                entry.emplace("runtime", Value("wasm"));
                entry.emplace("module", Value(package_name + ":orchestration"));
                entry.emplace("export", Value("forward"));
                return Value(std::move(entry));
            }
            const auto native = project.build_packages().find("native");
            if (native != project.build_packages().end() && native->second.runtime == "native") {
                entry.emplace("runtime", Value("native"));
                entry.emplace("module", Value("entrypoint"));
                entry.emplace("export", Value("forward"));
                return Value(std::move(entry));
            }
            const auto graph = project.build_packages().find("graph");
            if (graph != project.build_packages().end() && graph->second.runtime == "iree") {
                entry.emplace("runtime", Value("graph"));
                entry.emplace("module", Value(package_name + ":graph"));

                entry.emplace("export", Value("*"));
                return Value(std::move(entry));
            }
            Errors::at(Error::Entrypoint::Path::Unknown, "entrypoint.package", package_name,
                       selected.path);
            return Value(std::move(entry));
        }

        Value::Array normalized_hooks(const Package::Resolution &resolution) {
            Value::Array result = manifest_array(resolution, "hooks");
            for (Value &item : result) {
                Value::Table &hook = std::get<Value::Table>(item.data);
                if (const Value *target = item.find("package")) {
                    const std::string_view name = target->string();
                    const bool root = name == resolution.root().project.metadata().name;
                    const bool dependency =
                        std::ranges::any_of(resolution.root().dependencies,
                                            [name](const Package::Dependency::Edge &edge) {
                                                return edge.name == name;
                                            });
                    if (!root && !dependency)
                        Errors::at(Error::Hook::Package::Unknown, "hooks.package", name);
                }
                const Value *function = item.find("function");
                if (function == nullptr)
                    continue;
                const std::string reference(function->string());
                const std::size_t separator = reference.find(':');
                if (separator == std::string::npos)
                    Errors::at(Error::Hook::Function::Format::Invalid, "hooks.function", reference);

                const std::string_view owner(reference.data(), separator);
                const std::string_view facet(reference.data() + separator + 1,
                                             reference.size() - separator - 1);
                const Package::Resolved::Package *package = nullptr;
                if (owner == resolution.root().project.metadata().name) {
                    package = &resolution.root();
                } else {
                    const auto dependency =
                        std::ranges::find_if(resolution.root().dependencies,
                                             [owner](const Package::Dependency::Edge &edge) {
                                                 return edge.name == owner;
                                             });
                    if (dependency != resolution.root().dependencies.end())
                        package = &resolution.packages().at(dependency->index);
                }
                if (package == nullptr)
                    Errors::at(Error::Hook::Package::Unknown, "hooks.function", owner);

                const auto build = package->project.build_packages().find(facet);
                if (build != package->project.build_packages().end() &&
                    build->second.runtime == "wasm") {
                    hook["module"] = Value(reference);
                    hook["export"] = Value("forward");
                    continue;
                }

                const auto compiler_export = std::ranges::find_if(
                    package->project.build_packages(), [facet](const auto &entry) {
                        return std::ranges::find(entry.second.exports, facet) !=
                               entry.second.exports.end();
                    });
                if (compiler_export != package->project.build_packages().end()) {
                    hook["implementation"] = Value(reference);
                    continue;
                }

                Errors::at(Error::Hook::Function::Format::Invalid, "hooks.function", reference);
            }
            return result;
        }
    }

    Config::Value::Array Plan::entrypoints(const Package::Resolution &resolution) {
        const auto selected = resolution.entrypoint();
        return selected ? Value::Array{derived_entrypoint(resolution, *selected)} : Value::Array{};
    }

    Config::Value::Array Plan::hooks(const Package::Resolution &resolution) {
        return normalized_hooks(resolution);
    }

    Config::Value Plan::full(const Package::Resolution &resolution, const Artifacts &artifacts) {
        Value::Table plan;
        plan.emplace("version", Value(int64_t{1}));

        Value::Array plugins;
        for (const std::filesystem::path &plugin : artifacts.plugins) {
            Value::Table entry;
            entry.emplace("path", Value(plugin.string()));
            plugins.push_back(Value(std::move(entry)));
        }
        plan.emplace("plugins", Value(std::move(plugins)));

        plan.emplace("policies", manifest_section(resolution, "policies") != nullptr
                                     ? *manifest_section(resolution, "policies")
                                     : Value(Value::Table{}));

        plan.emplace("formats", Value(compiler_array(resolution, "formats")));
        plan.emplace("entrypoints", Value(Plan::entrypoints(resolution)));
        plan.emplace("hooks", Value(Plan::hooks(resolution)));
        plan.emplace("validation", manifest_section(resolution, "validation") != nullptr
                                       ? *manifest_section(resolution, "validation")
                                       : Value(Value::Table{}));
        plan.emplace("scheduling", manifest_section(resolution, "scheduling") != nullptr
                                       ? *manifest_section(resolution, "scheduling")
                                       : Value(Value::Table{}));
        plan.emplace("topology", manifest_section(resolution, "topology") != nullptr
                                     ? *manifest_section(resolution, "topology")
                                     : Value(Value::Table{}));
        if (const auto spec = tuning_spec(resolution))
            plan.emplace("tuning_spec", Value(spec->string()));

        Value::Table configurations;
        for (const auto &[name, value] : resolution.configurations())
            configurations.emplace(name, value);
        plan.emplace("package_config", Value(std::move(configurations)));

        Value::Array weights;
        for (const std::filesystem::path &path : artifacts.weights) {
            Value::Table entry;
            entry.emplace("path", Value(path.string()));
            weights.push_back(Value(std::move(entry)));
        }
        plan.emplace("weights", Value(std::move(weights)));

        Value::Array modules;
        for (const Runtime::Module &module : artifacts.modules) {
            Value::Table entry = file_entry(module.name, module.path).table();
            entry.emplace("runtime", Value(module.runtime));
            modules.emplace_back(std::move(entry));
        }
        plan.emplace("modules", Value(std::move(modules)));

        Value::Array assets;
        for (const auto &[name, path] : artifacts.assets)
            assets.push_back(file_entry(name, path));
        plan.emplace("assets", Value(std::move(assets)));

        Value result(std::move(plan));
        Plan::validate(result);
        return result;
    }

    Config::Value Plan::graph(const Package::Resolution &resolution, const Artifacts &artifacts,
                              const std::filesystem::path &requirements_output,
                              const std::filesystem::path &parameters_package,
                              std::string_view graph_implementation) {
        Value result = Plan::full(resolution, artifacts);
        Value::Table &table = std::get<Value::Table>(result.data);
        table["mode"] = Value("graph");
        table["package_config"] = graph_configurations(resolution);
        table["entrypoints"] = Value(Value::Array{});
        Value::Array graph_hooks;
        for (const Value &hook : table.at("hooks").array())
            if (hook.find("implementation") != nullptr)
                graph_hooks.push_back(hook);
        table["hooks"] = Value(std::move(graph_hooks));
        if (!requirements_output.empty())
            table["requirements_output"] = Value(requirements_output.string());
        if (!parameters_package.empty())
            table["parameters_package"] = Value(parameters_package.string());
        if (!graph_implementation.empty())
            table["graph_implementation"] = Value(std::string(graph_implementation));
        table["modules"] = Value(Value::Array{});
        table["assets"] = Value(Value::Array{});
        return result;
    }

    Config::Value Plan::weights(const Package::Resolution &resolution, const Artifacts &artifacts,
                                const std::filesystem::path &requirements,
                                std::size_t conversion_jobs, uint64_t conversion_memory_bytes) {
        Value result = Plan::full(resolution, artifacts);
        Value::Table &table = std::get<Value::Table>(result.data);
        table["mode"] = Value("weights");
        table["requirements"] = Value(requirements.string());
        table["conversion_jobs"] = Value(static_cast<int64_t>(conversion_jobs));
        table["conversion_memory_bytes"] = Value(static_cast<int64_t>(
            std::min<uint64_t>(conversion_memory_bytes, static_cast<uint64_t>(INT64_MAX))));
        table["modules"] = Value(Value::Array{});
        table["assets"] = Value(Value::Array{});
        return result;
    }

    Config::Value weights_fingerprint(const Package::Resolution &resolution) {
        Value::Table result;
        result.emplace("formats", Value(compiler_array(resolution, "formats")));

        Value::Table policies;
        if (const Value *configured = manifest_section(resolution, "policies")) {
            for (const auto &[path, policy] : configured->table()) {
                Value::Table relevant;
                if (const Value *schemes = policy.find("schemes"))
                    relevant.emplace("schemes", *schemes);
                policies.emplace(path, Value(std::move(relevant)));
            }
        }
        result.emplace("policies", Value(std::move(policies)));

        result.emplace("package_config", graph_configurations(resolution));
        return Value(std::move(result));
    }

    Config::Value Plan::requirements(const std::filesystem::path &graph_cache) {
        Value::Table result;
        result.emplace("version", Value(int64_t{1}));
        result.emplace("mode", Value("requirements"));
        result.emplace("base_package", Value(graph_cache.string()));
        return Value(std::move(result));
    }

    Config::Value
    Plan::merge_graphs(const std::vector<std::pair<std::string, std::filesystem::path>> &graphs) {
        Value::Table result;
        result.emplace("version", Value(int64_t{1}));
        result.emplace("mode", Value("merge-graphs"));
        Value::Array entries;
        for (const auto &[name, path] : graphs)
            entries.push_back(file_entry(name, path));
        result.emplace("graphs", Value(std::move(entries)));
        return Value(std::move(result));
    }

    Config::Value Plan::package(const Package::Resolution &resolution, const Artifacts &artifacts,
                                const std::filesystem::path &graph_cache,
                                const std::filesystem::path &parameters_cache) {
        Value result = Plan::full(resolution, artifacts);
        Value::Table &table = std::get<Value::Table>(result.data);
        table["mode"] = Value("package");
        table["base_package"] = Value(graph_cache.string());
        table["parameters_package"] = Value(parameters_cache.string());
        return result;
    }

    void Plan::validate(const Config::Value &plan) {
        const Value::Table &root = plan.table();

        const auto entrypoints = root.find("entrypoints");
        if (entrypoints == root.end() || entrypoints->second.array().empty())
            Errors::at(Error::Entrypoint::Missing, "entrypoints");

        std::set<std::string> names;
        std::set<std::pair<std::string, std::string>> exports;
        std::map<std::string, std::string, std::less<>> modules;
        for (const Value &module : root.at("modules").array())
            modules.emplace(std::string(module.require("name", "modules").string()),
                            std::string(module.require("runtime", "modules").string()));
        for (const Value &entrypoint : entrypoints->second.array()) {
            const Value &name = entrypoint.require("name", "entrypoints");
            if (!names.insert(std::string(name.string())).second)
                Errors::at(Error::Entrypoint::Duplicated, "entrypoints", name.string());

            const std::string module{entrypoint.require("module", "entrypoints").string()};
            const std::string runtime{entrypoint.require("runtime", "entrypoints").string()};
            if (runtime == "wasm" && (!modules.contains(module) || modules.at(module) != "wasm"))
                Errors::at(Error::Entrypoint::Module::Unknown, "entrypoints.module", module);

            if (const Value *exported = entrypoint.find("export")) {
                if (!exports.insert({module, std::string(exported->string())}).second)
                    Errors::at(Error::Entrypoint::Export::Conflict, "entrypoints",
                               exported->string(), module);
            }
        }

        for (const Value &hook : root.at("hooks").array()) {
            const Value &path = hook.require("path", "hooks");
            if (hook.find("implementation") == nullptr && hook.find("module") == nullptr)
                Errors::at(Error::Hook::Function::Missing, "hooks", path.string());
            if (const Value *module = hook.find("module"))
                if (!modules.contains(std::string(module->string())) ||
                    modules.at(std::string(module->string())) != "wasm")
                    Errors::at(Error::Hook::Module::Unknown, "hooks.module", module->string());
        }

        for (const char *group : {"assets"}) {
            std::set<std::string> seen;
            for (const Value &entry : root.at(group).array()) {
                const std::string name{entry.require("name").string()};
                if (!seen.insert(name).second)
                    Errors::raise(Error::Name::Duplicated, group, name);
            }
        }
    }
}
