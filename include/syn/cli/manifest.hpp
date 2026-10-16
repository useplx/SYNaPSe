#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace Syn::Cli::Manifest {

    struct Dependency {
            std::string name;
            std::string path;
            std::string git;
    };

    void create(const std::filesystem::path &file, std::string_view name, std::string_view version,
                const Dependency *dependency = nullptr);

    void add(const std::filesystem::path &file, const Dependency &dependency);

    void remove(const std::filesystem::path &file, std::string_view name);
}
