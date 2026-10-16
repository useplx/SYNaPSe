#include "syn/cli/command.hpp"
#include "syn/cli/workspace.hpp"

#include <print>
#include <string>

namespace Syn::Cli {
    namespace {
        class Graph final : public Command {
            public:
                Graph() : Command("graph", "dependency and capability graph") {}

                int run(const Invocation &invocation) const override {
                    const Package::Resolution resolution = resolve(project_file(invocation));
                    const auto &packages = resolution.packages();

                    const auto print = [&](this auto &&self, std::size_t index,
                                           std::string_view prefix, bool last) -> void {
                        const Package::Resolved::Package &package = packages.at(index);
                        const auto &provides = package.project.provides();
                        std::string capabilities;
                        for (const std::string &provide : provides) {
                            capabilities += capabilities.empty() ? "  [" : ", ";
                            capabilities += provide;
                        }
                        if (!capabilities.empty())
                            capabilities += ']';
                        std::println(
                            "{}{}{}{}", prefix, prefix.empty() ? "" : (last ? "└── " : "├── "),
                            package.project.metadata().name, capabilities);

                        const std::string child =
                            prefix.empty() ? std::string{}
                                           : std::string(prefix) + (last ? "    " : "│   ");
                        for (std::size_t i = 0; i < package.dependencies.size(); ++i)
                            self(package.dependencies[i].index,
                                 prefix.empty() ? std::string_view{} : std::string_view(child),
                                 i + 1 == package.dependencies.size());
                    };
                    print(0, "", true);

                    if (!resolution.bindings().empty()) {
                        std::println("\ncapabilities:");
                        for (const auto &[capability, index] : resolution.bindings())
                            std::println("  {} -> {}", capability,
                                         packages.at(index).project.metadata().name);
                    }
                    return 0;
                }
        };

        const Graph command;
        const int registered = [] {
            Commands::add(command);
            return 0;
        }();
    }
}
