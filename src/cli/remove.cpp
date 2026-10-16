#include "syn/cli/command.hpp"
#include "syn/cli/manifest.hpp"
#include "syn/cli/workspace.hpp"

#include <print>

namespace Syn::Cli {
    namespace {
        class Remove final : public Command {
            public:
                Remove()
                    : Command("remove", "remove uma dependencia",
                              {{"pkg", "nome da dependencia a remover "
                                       "de [dependencies]"}}) {}

                int run(const Invocation &invocation) const override {
                    const std::string_view name = invocation.argument("pkg");
                    Manifest::remove(project_file(invocation), name);
                    std::println("dependencia `{}` removida", name);
                    return 0;
                }
        };

        const Remove command;
        const int registered = [] {
            Commands::add(command);
            return 0;
        }();
    }
}
