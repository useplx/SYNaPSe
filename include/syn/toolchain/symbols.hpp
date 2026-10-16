#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace Syn::Toolchain {
    namespace Symbols {
        std::vector<std::string> exported(const std::filesystem::path &artifact);

        void require(const std::filesystem::path &artifact,
                     const std::vector<std::string> &required, const std::string &origin);
    }
}
