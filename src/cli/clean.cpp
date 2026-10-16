#include "syn/cli/command.hpp"
#include "syn/cli/workspace.hpp"
#include "syn/error.hpp"

#include <print>

namespace Syn::Cli {
    namespace {
        class Clean final : public Command {
            public:
                Clean() : Command("clean", "apaga os artefatos de build") {}

                int run(const Invocation &invocation) const override {
                    const std::filesystem::path root = project_file(invocation).parent_path();
                    const std::filesystem::path build = root / "build";
                    std::error_code error;
                    const std::uintmax_t removed = std::filesystem::remove_all(build, error);
                    if (error)
                        Errors::raise(Error::Build::Directory::Not::Removed, build,
                                      error.message());
                    std::println("{} arquivos removidos", removed);
                    return 0;
                }
        };

        const Clean command;
        const int registered = [] {
            Commands::add(command);
            return 0;
        }();
    }
}
