#pragma once

#include <filesystem>
#include <vector>

namespace Syn::Toolchain::Sdk {
    std::filesystem::path myelin();
    std::filesystem::path compiler();
    std::vector<std::filesystem::path> ir();
    std::vector<std::filesystem::path> headers();
}
