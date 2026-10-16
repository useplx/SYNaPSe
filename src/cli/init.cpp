#include "syn/cli/command.hpp"
#include "syn/cli/manifest.hpp"
#include "syn/error.hpp"

#include <print>

namespace Syn::Cli {
    namespace {
        class Init final : public Command {
            public:
                Init() : Command("init", "cria synps.toml no diretorio atual") {}

                int run(const Invocation &) const override {
                    const std::filesystem::path directory = std::filesystem::current_path();
                    const std::filesystem::path file = directory / "synps.toml";
                    if (std::filesystem::exists(file))
                        Errors::raise(Error::Target::Exists, "synps.toml");

                    Manifest::create(file, directory.filename().string(), "0.1.0", nullptr);
                    std::println("synps.toml criado");
                    return 0;
                }
        };

        const Init command;
        const int registered = [] {
            Commands::add(command);
            return 0;
        }();
    }
}
