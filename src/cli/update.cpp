#include "syn/cli/command.hpp"
#include "syn/cli/workspace.hpp"

namespace Syn::Cli {
    namespace {
        class Update final : public Command {
            public:
                Update() : Command("update", "reresolve ignorando o lockfile") {}

                int run(const Invocation &invocation) const override { return install(invocation); }
        };

        const Update command;
        const int registered = [] {
            Commands::add(command);
            return 0;
        }();
    }
}
