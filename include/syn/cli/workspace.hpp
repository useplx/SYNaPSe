#pragma once

#include "syn/build/planner.hpp"
#include "syn/cli/command.hpp"
#include "syn/package/resolver.hpp"

#include <filesystem>

namespace Syn::Cli {

    std::filesystem::path project_file(const Invocation &invocation);

    Package::Resolution resolve(const std::filesystem::path &file);

    std::filesystem::path bind_myelin(const Package::Resolution &resolution);

    Build::Planner::Options planner_options(const std::filesystem::path &root,
                                            const Invocation &invocation);

    int install(const Invocation &invocation);
}
