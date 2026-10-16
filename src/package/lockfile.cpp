#include "syn/package/lockfile.hpp"

#include "syn/error.hpp"
#include "syn/system/hash.hpp"

#include <toml++/toml.h>

#include <fstream>

namespace Syn::Package {
    namespace {
        std::string source_name(Source::Kind kind) {
            switch (kind) {
            case Source::Kind::Root:
                return "root";
            case Source::Kind::Path:
                return "path";
            case Source::Kind::Git:
                return "git";
            }
            return "unknown";
        }

        std::string required_text(const toml::table &table, std::string_view key,
                                  std::string_view location) {
            if (const auto value = table[key].value<std::string>())
                return *value;
            Errors::at(Error::Lockfile::Field::String::Expected,
                       std::string(location) + "." + std::string(key));
        }
    }

    Lockfile Lockfile::create(const Resolution &resolution) {
        Lockfile result;
        result.packages_.reserve(resolution.packages().size());
        for (const Resolved::Package &package : resolution.packages()) {
            Locked::Package locked{
                package.project.metadata().name,
                package.project.metadata().version,
                source_name(package.source.kind),
                package.source.location,
                package.source.revision,
                System::hash_tree(package.project.root()),
                {},
            };
            for (const Dependency::Edge &dependency : package.dependencies)
                locked.dependencies.push_back({
                    dependency.name,
                    resolution.packages().at(dependency.index).project.metadata().name,
                });
            result.packages_.push_back(std::move(locked));
        }
        return result;
    }

    Lockfile Lockfile::load(const std::filesystem::path &path) {
        toml::table document;
        try {
            document = toml::parse_file(path.string());
        } catch (const toml::parse_error &error) {
            Errors::at(Error::Toml::Invalid, path.string(), error.description());
        }
        if (document["lock"].value_or<int64_t>(0) != 1)
            Errors::at(Error::Lockfile::Version::Unsupported, path.string());
        const auto *packages = document["package"].as_array();
        if (!packages)
            Errors::at(Error::Lockfile::Package::Array::Expected, path.string());

        Lockfile result;
        for (std::size_t index = 0; index < packages->size(); ++index) {
            const auto *table = (*packages)[index].as_table();
            const std::string location = "package[" + std::to_string(index) + "]";
            if (!table)
                Errors::at(Error::Lockfile::Package::Table::Expected, location);
            Locked::Package package{
                required_text(*table, "name", location),
                required_text(*table, "version", location),
                required_text(*table, "source", location),
                required_text(*table, "location", location),
                table->get("revision") ? required_text(*table, "revision", location)
                                       : std::string{},
                required_text(*table, "content_hash", location),
                {},
            };
            if (const auto *dependencies = (*table)["dependencies"].as_array()) {
                for (std::size_t dependency_index = 0; dependency_index < dependencies->size();
                     ++dependency_index) {
                    const auto *dependency = (*dependencies)[dependency_index].as_table();
                    if (!dependency)
                        Errors::at(Error::Lockfile::Dependency::Table::Expected,
                                   location + ".dependencies[" + std::to_string(dependency_index) +
                                       "]");
                    package.dependencies.push_back({
                        required_text(*dependency, "name", location + ".dependencies"),
                        required_text(*dependency, "package", location + ".dependencies"),
                    });
                }
            }
            result.packages_.push_back(std::move(package));
        }
        return result;
    }

    void Lockfile::write(const std::filesystem::path &path) const {
        toml::table document;
        document.insert("lock", 1);
        toml::array packages;
        for (const Locked::Package &package : packages_) {
            toml::table item;
            item.insert("name", package.name);
            item.insert("version", package.version);
            item.insert("source", package.source);
            item.insert("location", package.location);
            if (!package.revision.empty())
                item.insert("revision", package.revision);
            item.insert("content_hash", package.content_hash);
            toml::array dependencies;
            for (const Locked::Dependency &dependency : package.dependencies) {
                toml::table value;
                value.insert("name", dependency.name);
                value.insert("package", dependency.package);
                dependencies.push_back(std::move(value));
            }
            item.insert("dependencies", std::move(dependencies));
            packages.push_back(std::move(item));
        }
        document.insert("package", std::move(packages));

        if (!path.parent_path().empty())
            std::filesystem::create_directories(path.parent_path());
        std::ofstream output(path);
        if (!output)
            Errors::at(Error::File::Write::Failed, path.string());
        output << document << '\n';
    }
}
