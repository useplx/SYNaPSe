#include "syn/cli/manifest.hpp"

#include "syn/error.hpp"

#include <toml++/toml.h>

#include <fstream>

namespace Syn::Cli::Manifest {
    namespace {
        constexpr auto FLAGS =
            toml::toml_formatter::default_flags & ~toml::format_flags::allow_literal_strings;

        toml::table read(const std::filesystem::path &file) {
            try {
                return toml::parse_file(file.string());
            } catch (const toml::parse_error &error) {
                Errors::at(Error::Toml::Invalid, file.string(), error.description());
            }
        }

        void write(const std::filesystem::path &file, const toml::table &document) {
            if (!file.parent_path().empty())
                std::filesystem::create_directories(file.parent_path());
            std::ofstream output(file, std::ios::trunc);
            if (!output)
                Errors::at(Error::File::Write::Failed, file.string());
            output << toml::toml_formatter{document, FLAGS} << '\n';
        }

        toml::table source_of(const Dependency &dependency) {
            toml::table source;
            if (!dependency.path.empty())
                source.insert("path", dependency.path);
            else
                source.insert("git", dependency.git);
            source.is_inline(true);
            return source;
        }

        toml::table &dependencies_of(toml::table &document) {
            if (auto *found = document["dependencies"].as_table())
                return *found;
            document.insert_or_assign("dependencies", toml::table{});
            return *document["dependencies"].as_table();
        }
    }

    void create(const std::filesystem::path &file, std::string_view name, std::string_view version,
                const Dependency *dependency) {
        toml::table package;
        package.insert("name", name);
        package.insert("version", version);

        toml::table document;
        document.insert("package", std::move(package));
        if (dependency != nullptr) {
            toml::table dependencies;
            dependencies.insert(dependency->name, source_of(*dependency));
            document.insert("dependencies", std::move(dependencies));
        }
        write(file, document);
    }

    void add(const std::filesystem::path &file, const Dependency &dependency) {
        toml::table document = read(file);
        toml::table &dependencies = dependencies_of(document);
        if (dependencies.contains(dependency.name))
            Errors::at(Error::Dependency::Exists, "dependencies", dependency.name);
        dependencies.insert(dependency.name, source_of(dependency));
        write(file, document);
    }

    void remove(const std::filesystem::path &file, std::string_view name) {
        toml::table document = read(file);
        auto *dependencies = document["dependencies"].as_table();
        if (dependencies == nullptr || dependencies->erase(name) == 0)
            Errors::at(Error::Dependency::Unknown, "dependencies", name);
        write(file, document);
    }
}
