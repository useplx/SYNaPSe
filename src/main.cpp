#include "syn/cli/command.hpp"
#include "syn/error.hpp"

#include <cstdio>
#include <print>
#include <string_view>

int main(int argc, char **argv) {
    using namespace Syn;

    if (argc < 2) {
        std::print("{}", Cli::Commands::usage());
        return 2;
    }

    std::string_view name = argv[1];
    if (name == "--version" || name == "-V") {
        std::println("synps {}", SYNPS_VERSION);
        return 0;
    }
    if (name == "--help" || name == "-h")
        name = "help";

    const Cli::Command *command = Cli::Commands::find(name);
    if (command == nullptr) {
        std::println(stderr, "unknown command: {}\n", name);
        std::print(stderr, "{}", Cli::Commands::usage());
        return 2;
    }

    try {
        return command->run(Cli::Commands::parse(*command, argc, argv, 2));
    } catch (const Diagnostic &diagnostic) {
        Errors::report(diagnostic);
        return 1;
    } catch (const std::exception &error) {
        std::println(stderr, "error: {}", error.what());
        return 1;
    }
}
