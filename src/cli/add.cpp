#include "syn/cli/command.hpp"
#include "syn/cli/manifest.hpp"
#include "syn/cli/workspace.hpp"
#include "syn/error.hpp"

#include <print>

namespace Syn::Cli {
    namespace {
        class Add final : public Command {
            public:
                Add()
                    : Command("add", "acrescenta uma dependencia",
                              {{"pkg", "nome da dependencia, como ela "
                                       "aparece em [dependencies]"}},
                              {{"path", "P",
                                "local directory containing the synps.toml "
                                "da dependencia"},
                               {"git", "URL", "repositorio git de onde clona-la"}}) {}

                int run(const Invocation &invocation) const override {
                    Manifest::Dependency dependency{
                        std::string(invocation.argument("pkg")),
                        std::string(invocation.option("path")),
                        std::string(invocation.option("git")),
                    };
                    if (dependency.path.empty() && dependency.git.empty())
                        Errors::raise(Error::Dependency::Source::Required);

                    Manifest::add(project_file(invocation), dependency);
                    std::println("dependencia `{}` acrescentada", dependency.name);
                    return 0;
                }
        };

        const Add command;
        const int registered = [] {
            Commands::add(command);
            return 0;
        }();
    }
}
