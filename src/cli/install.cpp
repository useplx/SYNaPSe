#include "syn/cli/command.hpp"
#include "syn/cli/workspace.hpp"

namespace Syn::Cli {
    namespace {
        class Install final : public Command {
            public:
                Install() : Command("install", "resolve dependencias e grava synps.lock") {}

                int run(const Invocation &invocation) const override { return install(invocation); }
        };

        const Install command;
        const int registered = [] {
            Commands::add(command);
            return 0;
        }();
    }
}
