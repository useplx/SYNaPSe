#include "syn/cli/workspace.hpp"

#include "syn/config/project.hpp"
#include "syn/error.hpp"
#include "syn/package/lockfile.hpp"
#include "syn/toolchain/abi.hpp"
#include "syn/toolchain/driver.hpp"
#include "syn/toolchain/sdk.hpp"

#include <cstdlib>
#include <print>

namespace Syn::Cli {
    namespace {
        std::filesystem::path compiler_api() {
            if (const char *configured = std::getenv("MLN_COMPILER_API");
                configured != nullptr && *configured != '\0')
                return configured;

#if defined(SYNPS_INSTALLED_COMPILER_API)
            std::error_code error;
            const std::filesystem::path executable =
                std::filesystem::read_symlink("/proc/self/exe", error);
            if (!error) {
                const std::filesystem::path installed =
                    (executable.parent_path() / SYNPS_INSTALLED_COMPILER_API).lexically_normal();
                if (std::filesystem::is_regular_file(installed, error))
                    return installed;
            }
#endif

            return SYNPS_DEFAULT_MLN_COMPILER_API;
        }
    }

    std::filesystem::path project_file(const Invocation &invocation) {
        if (invocation.has("project")) {
            const std::filesystem::path path = invocation.option("project");
            return std::filesystem::absolute(path).lexically_normal();
        }
        return Config::Project::find(std::filesystem::current_path());
    }

    Package::Resolution resolve(const std::filesystem::path &file) {
        return Package::Resolver({}).resolve(file);
    }

    std::filesystem::path bind_myelin(const Package::Resolution &) {
        const char *configured = std::getenv("MLN_ROOT");
        std::filesystem::path root = configured != nullptr && *configured != '\0'
                                         ? std::filesystem::path(configured)
                                         : Toolchain::Sdk::myelin();
        root = std::filesystem::absolute(root).lexically_normal();
        if (!std::filesystem::is_regular_file(root / "src/sdk/buffer.cpp") ||
            !std::filesystem::exists(root / "include" / "mln"))
            Errors::at(Error::Myelin::Root::Unresolved, root.string());
        Toolchain::Myelin::set_root(root);
        return root;
    }

    Build::Planner::Options planner_options(const std::filesystem::path &root,
                                            const Invocation &invocation) {
        Build::Planner::Options planner;
        planner.build_directory = root / "build";
        planner.cache_directory = root / "build" / ".cache";
        planner.optimize = invocation.has("release");
        planner.compiler_api = compiler_api();
        return planner;
    }

    int install(const Invocation &invocation) {
        const std::filesystem::path file = project_file(invocation);
        const Package::Resolution resolution = resolve(file);
        Package::Lockfile::create(resolution).write(file.parent_path() / "synps.lock");
        std::println("{} packages resolvidos -> synps.lock", resolution.packages().size());
        return 0;
    }
}
