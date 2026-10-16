#include "syn/cli/command.hpp"

#include <cstdio>
#include <print>

namespace Syn::Cli {
    namespace {
        class Help final : public Command {
            public:
                Help()
                    : Command("help", "descreve os comandos e seus argumentos",
                              {{"comando", "comando a detalhar; sem ele, lista todos", false}}) {}

                int run(const Invocation &invocation) const override {
                    const std::string_view name = invocation.argument("comando");
                    if (name.empty()) {
                        std::print("{}", Commands::usage());
                        return 0;
                    }
                    const Command *command = Commands::find(name);
                    if (command == nullptr) {
                        std::println(stderr, "unknown command: {}\n", name);
                        std::print(stderr, "{}", Commands::usage());
                        return 2;
                    }
                    std::print("{}", Commands::usage(*command));
                    return 0;
                }
        };

        const Help command;
        const int registered = [] {
            Commands::add(command);
            return 0;
        }();
    }
}
