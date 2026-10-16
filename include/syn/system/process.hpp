#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace Syn::System {
    struct Command {
            std::string executable;
            std::vector<std::string> arguments;
            std::filesystem::path working_directory;
            std::map<std::string, std::string, std::less<>> environment;
    };

    namespace Process {
        struct Result {
                int exit_code = -1;
                std::string output;
                std::string error;
        };
    }

    Process::Result run(const Command &command);
    Process::Result run_checked(const Command &command, std::string_view operation = {});
}
