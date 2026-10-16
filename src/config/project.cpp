#include "syn/config/project.hpp"
#include "syn/error.hpp"

#include <toml++/toml.h>

#include <algorithm>
#include <cstdlib>
#include <set>
#include <system_error>

namespace Syn::Config {
    namespace {
        Value convert(const toml::node &node) {
            if (const auto *table = node.as_table()) {
                Value::Table result;
                for (const auto &[key, item] : *table)
                    result.emplace(std::string(key.str()), convert(item));
                return result;
            }
            if (const auto *array = node.as_array()) {
                Value::Array result;
                result.reserve(array->size());
                for (const toml::node &item : *array)
                    result.push_back(convert(item));
                return result;
            }
            if (node.is_boolean())
                if (const auto value = node.value<bool>())
                    return *value;
            if (const auto value = node.value<int64_t>())
                return *value;
            if (const auto value = node.value<double>())
                return *value;
            if (const auto value = node.value<std::string>())
                return *value;
            Errors::raise(Error::Value::Unsupported);
        }

        std::string expand(std::string value);

        std::string expand(std::string value) {
            if (value.starts_with("~/"))
                if (const char *home = std::getenv("HOME"))
                    value = std::string(home) + value.substr(1);
            std::string result;
            for (std::size_t i = 0; i < value.size(); ++i) {
                if (value[i] != '$' || i + 1 >= value.size() || value[i + 1] != '{') {
                    result.push_back(value[i]);
                    continue;
                }
                const std::size_t close = value.find('}', i + 2);
                if (close == std::string::npos) {
                    result.append(value.substr(i));
                    break;
                }
                std::string name = value.substr(i + 2, close - i - 2);
                std::string fallback;
                if (const std::size_t dash = name.find(":-"); dash != std::string::npos) {
                    fallback = name.substr(dash + 2);
                    name = name.substr(0, dash);
                }
                const char *found = std::getenv(name.c_str());
                if (found && *found)
                    result.append(found);
                else if (!fallback.empty())
                    result.append(expand(std::move(fallback)));
                else
                    Errors::raise(Error::Environment::Variable::Unset, name);
                i = close;
            }
            return result;
        }

        void expand(Value &value) {
            if (auto *text = std::get_if<std::string>(&value.data)) {
                *text = expand(std::move(*text));
                return;
            }
            if (auto *array = std::get_if<Value::Array>(&value.data)) {
                for (Value &item : *array)
                    expand(item);
                return;
            }
            if (auto *table = std::get_if<Value::Table>(&value.data))
                for (auto &[key, item] : *table) {
                    (void)key;
                    expand(item);
                }
        }

        std::string text(const toml::table &table, std::string_view key, std::string_view location,
                         bool required = false) {
            const toml::node *node = table.get(key);
            if (!node) {
                if (required)
                    Errors::at(Error::Required::Missing,
                               std::string(location) + "." + std::string(key), "field");
                return {};
            }
            if (const auto value = node->value<std::string>())
                return *value;
            Errors::at(Error::Manifest::Field::String::Expected,
                       std::string(location) + "." + std::string(key));
        }

        std::vector<std::string> strings(const toml::table &table, std::string_view key,
                                         std::string_view location) {
            const toml::node *node = table.get(key);
            if (!node)
                return {};
            const auto *array = node->as_array();
            if (!array)
                Errors::at(Error::Manifest::Field::Array::Expected,
                           std::string(location) + "." + std::string(key));
            std::vector<std::string> result;
            result.reserve(array->size());
            for (std::size_t index = 0; index < array->size(); ++index) {
                const auto value = (*array)[index].value<std::string>();
                if (!value)
                    Errors::at(Error::Manifest::Field::Array::Element::String::Expected,
                               std::string(location) + "." + std::string(key) + "[" +
                                   std::to_string(index) + "]");
                result.push_back(*value);
            }
            return result;
        }

        const toml::table &required_table(const toml::table &parent, std::string_view key) {
            const toml::node *node = parent.get(key);
            if (!node)
                Errors::at(Error::Required::Missing, std::string(key), "table");
            const auto *result = node->as_table();
            if (!result)
                Errors::at(Error::Manifest::Table::Expected, std::string(key));
            return *result;
        }
    }

    Project Project::load(const std::filesystem::path &path) {
        toml::table parsed;
        try {
            parsed = toml::parse_file(path.string());
        } catch (const toml::parse_error &error) {
            Errors::at(Error::Toml::Invalid, path.string(), error.description());
        }

        Project result;
        result.file_ = std::filesystem::absolute(path).lexically_normal();
        result.root_ = result.file_.parent_path();
        result.document_ = convert(parsed);
        expand(result.document_);

        if (parsed.contains("project"))
            Errors::at(Error::Section::Renamed, "project", "[project]", "[package]");
        if (parsed.contains("config-schema"))
            Errors::at(Error::Section::Renamed, "config-schema", "[config-schema]",
                       "[package.variables]");

        const toml::table &package_declaration = required_table(parsed, "package");
        result.metadata_.name = text(package_declaration, "name", "package", true);
        result.metadata_.version = text(package_declaration, "version", "package", true);
        result.metadata_.description = text(package_declaration, "description", "package");
        if (const auto *entrypoint = parsed["entrypoint"].as_table())
            result.metadata_.entrypoint = text(*entrypoint, "package", "entrypoint", true);
        for (std::string value : strings(package_declaration, "include_directories", "package"))
            result.metadata_.include_directories.emplace_back(std::move(value));

        if (const auto *layout = parsed["layout"].as_table()) {
            if (const std::string value = text(*layout, "sources", "layout"); !value.empty())
                result.layout_.sources = value;
            if (const std::string value = text(*layout, "includes", "layout"); !value.empty())
                result.layout_.includes = value;
            if (const std::string value = text(*layout, "assets", "layout"); !value.empty())
                result.layout_.assets = value;
            if (const std::string value = text(*layout, "tests", "layout"); !value.empty())
                result.layout_.tests = value;
            if (const std::string value = text(*layout, "examples", "layout"); !value.empty())
                result.layout_.examples = value;
        }

        if (const auto *dependencies = parsed["dependencies"].as_table()) {
            for (const auto &[raw_name, node] : *dependencies) {
                const std::string name(raw_name.str());
                if (const auto version = node.value<std::string>()) {
                    result.dependencies_.emplace(name, Dependency::Registry{*version});
                    continue;
                }
                const auto *table = node.as_table();
                if (!table)
                    Errors::at(Error::Dependency::Source::Invalid, "dependencies." + name);
                const std::string path_value = text(*table, "path", "dependencies." + name);
                const std::string git = text(*table, "git", "dependencies." + name);
                const std::string version = text(*table, "version", "dependencies." + name);
                if (const toml::node *configuration = table->get("config")) {
                    if (!configuration->is_table())
                        Errors::at(Error::Manifest::Table::Expected,
                                   "dependencies." + name + ".config");
                    result.dependency_configurations_.emplace(name, convert(*configuration));
                }
                if ((!path_value.empty()) + (!git.empty()) +
                        (!version.empty() && path_value.empty() && git.empty()) !=
                    1)
                    Errors::at(Error::Dependency::Source::Ambiguous, "dependencies." + name);
                if (!path_value.empty())
                    result.dependencies_.emplace(name, Dependency::Path{path_value});
                else if (!git.empty())
                    result.dependencies_.emplace(
                        name,
                        Dependency::Git{git, text(*table, "rev", "dependencies." + name), version});
                else
                    result.dependencies_.emplace(name, Dependency::Registry{version});
            }
        }

        if (const auto *toolchains = parsed["toolchains"].as_table()) {
            for (const auto &[raw_name, node] : *toolchains) {
                const std::string name(raw_name.str());
                const auto *table = node.as_table();
                if (!table)
                    Errors::at(Error::Toolchain::Table::Expected, "toolchains." + name);
                Toolchain toolchain;
                toolchain.language = text(*table, "language", "toolchains." + name, true);
                toolchain.compiler = expand(text(*table, "compiler", "toolchains." + name));
                toolchain.linker = expand(text(*table, "linker", "toolchains." + name));
                toolchain.archiver = expand(text(*table, "archiver", "toolchains." + name));
                toolchain.triple = text(*table, "triple", "toolchains." + name);
                for (std::string flag : strings(*table, "flags", "toolchains." + name))
                    toolchain.flags.push_back(expand(std::move(flag)));
                result.toolchains_.emplace(name, std::move(toolchain));
            }
        }

        if (const auto *families = parsed["toolchain"].as_table()) {
            const auto walk = [&](const auto &self, const toml::table &table,
                                  const std::string &name, Toolchain inherited) -> void {
                const std::string location = "toolchain." + name;
                if (const std::string value = text(table, "compiler", location); !value.empty())
                    inherited.compiler = expand(value);
                if (const std::string value = text(table, "linker", location); !value.empty())
                    inherited.linker = expand(value);
                if (const std::string value = text(table, "archiver", location); !value.empty())
                    inherited.archiver = expand(value);
                if (const std::string value = text(table, "triple", location); !value.empty())
                    inherited.triple = value;
                for (std::string flag : strings(table, "flags", location))
                    inherited.flags.push_back(expand(std::move(flag)));
                const bool configured = !inherited.compiler.empty() || !inherited.linker.empty() ||
                                        !inherited.archiver.empty();
                if (configured)
                    result.toolchains_[name] = inherited;
                for (const auto &[raw_child, node] : table) {
                    const std::string child(raw_child.str());
                    if (child == "compiler" || child == "linker" || child == "archiver" ||
                        child == "triple" || child == "flags" || child == "language")
                        continue;
                    const auto *child_table = node.as_table();
                    if (!child_table)
                        Errors::at(Error::Toolchain::Table::Expected, location + "." + child);
                    self(self, *child_table, name + "." + child, inherited);
                }
            };
            for (const auto &[raw_language, node] : *families) {
                const std::string language(raw_language.str());
                const auto *table = node.as_table();
                if (!table)
                    Errors::at(Error::Toolchain::Table::Expected, "toolchain." + language);
                Toolchain base;
                base.language = language;
                walk(walk, *table, language, std::move(base));
            }
        }

        if (const auto *build = parsed["build"].as_table()) {
            const auto parse_recipe = [&](const toml::table &table, const std::string &name,
                                          const std::string &runtime, const std::string &location) {
                Build::Package package;
                package.name = name;
                package.runtime = runtime;
                package.runtime_inferred = true;
                const std::string selected = text(table, "toolchain", location);
                package.language = text(table, "language", location);
                if (package.language.empty())
                    package.language = selected;
                else if (selected != package.language)
                    package.toolchain = selected;
                if (package.language.empty())
                    Errors::at(Error::Required::Missing, location + ".toolchain", "field");
                package.graph_namespace = text(table, "namespace", location);
                package.artifact = text(table, "artifact", location);
                for (std::string value : strings(table, "sources", location))
                    package.sources.emplace_back(std::move(value));
                for (std::string value : strings(table, "includes", location))
                    package.includes.emplace_back(std::move(value));
                package.libraries = strings(table, "libraries", location);
                package.defines = strings(table, "defines", location);
                for (std::string value : strings(table, "library_directories", location))
                    package.library_directories.emplace_back(std::move(value));
                package.exports = strings(table, "exports", location);
                package.uses = strings(table, "uses", location);
                package.extends = strings(table, "extends", location);
                if (!result.build_packages_.emplace(name, std::move(package)).second)
                    Errors::at(Error::Name::Duplicated, location, name);
            };

            for (const auto &[raw_name, node] : *build) {
                const std::string written_name(raw_name.str());
                const auto *table = node.as_table();
                if (!table)
                    Errors::at(Error::Build::Domain::Table::Expected, "build." + written_name);

                const bool recipe = table->contains("sources") || table->contains("language") ||
                                    table->contains("libraries") || table->contains("exports");
                if (recipe) {
                    if (written_name != "graph" && written_name != "orchestration" &&
                        written_name != "native" && written_name != "driver")
                        Errors::at(Error::Build::Domain::Unknown, "build." + written_name);
                    const std::string runtime =
                        written_name == "graph"
                            ? "iree"
                            : (written_name == "orchestration"
                                   ? "wasm"
                                   : (written_name == "driver" ? "driver" : "native"));
                    parse_recipe(*table, written_name, runtime, "build." + written_name);
                    continue;
                }

                const bool labeled_recipes =
                    !table->empty() && std::ranges::all_of(*table, [](const auto &entry) {
                        return entry.second.is_table();
                    });
                if (labeled_recipes &&
                    (written_name == "graph" || written_name == "orchestration" ||
                     written_name == "native" || written_name == "driver")) {
                    const std::string runtime =
                        written_name == "graph"
                            ? "iree"
                            : (written_name == "orchestration"
                                   ? "wasm"
                                   : (written_name == "driver" ? "driver" : "native"));
                    const auto walk_recipes = [&](const auto &self, const toml::table &parent,
                                                  const std::string &label,
                                                  const std::string &location) -> void {
                        for (const auto &[raw_child, child_node] : parent) {
                            const auto *child = child_node.as_table();
                            if (!child)
                                Errors::at(Error::Package::Table::Expected,
                                           location + "." + std::string(raw_child.str()));
                            const std::string child_name(raw_child.str());
                            const std::string child_label =
                                label.empty() ? child_name : label + "." + child_name;
                            const std::string child_location = location + "." + child_name;
                            const bool child_recipe =
                                child->contains("sources") || child->contains("language") ||
                                child->contains("libraries") || child->contains("exports");
                            if (child_recipe)
                                parse_recipe(*child, child_label, runtime, child_location);
                            else
                                self(self, *child, child_label, child_location);
                        }
                    };
                    walk_recipes(walk_recipes, *table, "", "build." + written_name);
                    continue;
                }

                const std::string runtime = written_name == "host" ? "iree" : written_name;
                if (runtime != "iree" && runtime != "native" && runtime != "wasm")
                    Errors::at(Error::Build::Domain::Unknown, "build." + written_name);
                Build::Domain domain;
                domain.toolchain = text(*table, "toolchain", "build." + written_name);
                for (const auto &[raw_language, selected] : *table) {
                    const std::string language(raw_language.str());
                    if (language == "toolchain")
                        continue;
                    const auto name = selected.value<std::string>();
                    if (!name)
                        Errors::at(Error::Build::Toolchain::Name::Expected,
                                   "build." + written_name + "." + language);
                    domain.languages.emplace(language, *name);
                }
                if (domain.toolchain.empty() && domain.languages.empty())
                    Errors::at(Error::Build::Domain::Empty, "build." + written_name);
                const auto validate = [&](const std::string &name, const std::string &location) {
                    if (!result.toolchains_.contains(name))
                        Errors::at(Error::Build::Toolchain::Unknown, location, name);
                };
                if (!domain.toolchain.empty())
                    validate(domain.toolchain, "build." + written_name + ".toolchain");
                for (const auto &[language, name] : domain.languages)
                    validate(name, "build." + written_name + "." + language);
                if (!result.build_domains_.emplace(runtime, std::move(domain)).second)
                    Errors::at(Error::Build::Domain::Duplicated, "build." + written_name);
            }
        }

        if (parsed.contains("recipe"))
            Errors::at(Error::Section::Renamed, "recipe", "[recipe.<name>]", "[package.<name>]");

        if (const auto *packages = parsed["package"].as_table()) {
            for (const auto &[raw_name, node] : *packages) {
                const std::string name(raw_name.str());
                if (name == "name" || name == "version" || name == "description" ||
                    name == "include_directories" || name == "variables" || name == "entrypoint")
                    continue;
                const auto *table = node.as_table();
                if (!table)
                    Errors::at(Error::Package::Table::Expected, "package." + name);
                Build::Package package;
                package.name = name;
                package.runtime = text(*table, "runtime", "package." + name);
                package.graph_namespace = text(*table, "namespace", "package." + name);
                package.language = text(*table, "language", "package." + name, true);
                package.toolchain = text(*table, "toolchain", "package." + name);
                package.artifact = text(*table, "artifact", "package." + name);
                for (std::string value : strings(*table, "sources", "package." + name))
                    package.sources.emplace_back(std::move(value));
                for (std::string value : strings(*table, "includes", "package." + name))
                    package.includes.emplace_back(std::move(value));
                package.libraries = strings(*table, "libraries", "package." + name);
                package.defines = strings(*table, "defines", "package." + name);
                for (std::string value : strings(*table, "library_directories", "package." + name))
                    package.library_directories.emplace_back(std::move(value));
                package.exports = strings(*table, "exports", "package." + name);
                package.uses = strings(*table, "uses", "package." + name);
                package.extends = strings(*table, "extends", "package." + name);
                result.build_packages_.emplace(name, std::move(package));
            }
        }

        for (auto &[name, package] : result.build_packages_) {
            if (!package.runtime.empty())
                continue;
            package.runtime_inferred = true;
            if (name == "orchestration")
                package.runtime = "wasm";
            else if (name == "native")
                package.runtime = "native";
            else if (name == "driver")
                package.runtime = "driver";
            else if (name == "graph" || !package.exports.empty())
                package.runtime = "iree";
            else
                package.runtime = "native";
        }

        if (const auto *capabilities = parsed["capabilities"].as_table()) {
            result.provides_ = strings(*capabilities, "provides", "capabilities");
            result.needs_ = strings(*capabilities, "requires", "capabilities");
        }
        if (const auto *bindings = parsed["bindings"].as_table()) {
            for (const auto &[raw_capability, node] : *bindings) {
                const auto package = node.value<std::string>();
                if (!package)
                    Errors::at(Error::Binding::Package::Name::Expected,
                               "bindings." + std::string(raw_capability.str()));
                result.bindings_.emplace(std::string(raw_capability.str()), *package);
            }
        }
        return result;
    }

    std::string_view Project::toolchain_for(std::string_view runtime,
                                            std::string_view language) const noexcept {
        if (runtime == "driver")
            runtime = "native";
        const auto found = build_domains_.find(runtime);
        if (found == build_domains_.end())
            return {};
        const auto selected = found->second.languages.find(language);
        if (selected != found->second.languages.end())
            return selected->second;
        return found->second.toolchain;
    }

    std::filesystem::path Project::find(const std::filesystem::path &start) {
        std::filesystem::path current = std::filesystem::absolute(start);
        if (!std::filesystem::is_directory(current))
            current = current.parent_path();
        while (true) {
            const std::filesystem::path candidate = current / "synps.toml";
            if (std::filesystem::is_regular_file(candidate))
                return candidate;
            if (current == current.root_path())
                break;
            current = current.parent_path();
        }
        Errors::at(Error::Manifest::Not::Found, start.string());
    }
}
