#include "syn/cli/command.hpp"
#include "syn/cli/manifest.hpp"
#include "syn/error.hpp"

#include <print>

namespace Syn::Cli {
    namespace {
        class New final : public Command {
            public:
                New()
                    : Command("new", "cria um projeto",
                              {{"nome", "diretorio a criar e nome do "
                                        "package raiz"}}) {}

                int run(const Invocation &invocation) const override {
                    const std::string name(invocation.argument("nome"));
                    const std::filesystem::path root = name;
                    if (std::filesystem::exists(root))
                        Errors::raise(Error::Target::Exists, name);

                    for (const char *directory : {"src", "include", "assets"})
                        std::filesystem::create_directories(root / directory);

                    Manifest::create(root / "synps.toml", name, "0.1.0");
                    std::println("projeto `{}` criado", name);
                    return 0;
                }
        };

        const New command;
        const int registered = [] {
            Commands::add(command);
            return 0;
        }();
    }
}
