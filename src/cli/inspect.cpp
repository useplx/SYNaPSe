#include "syn/cli/command.hpp"
#include "syn/cli/workspace.hpp"
#include "syn/compiler/plan.hpp"
#include "syn/config/value.hpp"
#include "syn/toolchain/driver.hpp"

#include <print>

namespace Syn::Cli {
    namespace {
        class Inspect final : public Command {
            public:
                Inspect() : Command("inspect", "plano resolvido que o Compiler recebe", {}, {}) {}

                int run(const Invocation &invocation) const override {
                    const std::filesystem::path file = project_file(invocation);
                    const Package::Resolution resolution = resolve(file);
                    (void)bind_myelin(resolution);

                    Toolchain::Registry toolchains(resolution.root().project.toolchains());
                    const Syn::Build::Planner::Options options =
                        planner_options(file.parent_path(), invocation);
                    const Syn::Build::Planner::Plan built =
                        Syn::Build::Planner::run(resolution, toolchains, options);

                    std::println("{}",
                                 Config::json(Compiler::Plan::full(resolution, built.artifacts)));
                    return 0;
                }
        };

        const Inspect command;
        const int registered = [] {
            Commands::add(command);
            return 0;
        }();
    }
}
