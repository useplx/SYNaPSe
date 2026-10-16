#include "syn/toolchain/sdk.hpp"

#include <algorithm>
#include <string_view>

namespace Syn::Toolchain::Sdk {
    namespace {
        std::filesystem::path installed() {
            std::error_code error;
            const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
            if (error)
                return {};
            const auto sdk = (executable.parent_path() / SYNPS_INSTALLED_SDK).lexically_normal();
            return std::filesystem::is_regular_file(sdk / "myelin/src/sdk/buffer.cpp", error)
                       ? sdk
                       : std::filesystem::path{};
        }
    }

    std::filesystem::path myelin() {
        const auto sdk = installed();
        return sdk.empty() ? std::filesystem::path(SYNPS_DEFAULT_MLN_ROOT) : sdk / "myelin";
    }

    std::filesystem::path compiler() {
        const auto sdk = installed();
        return sdk.empty() ? std::filesystem::path(SYNPS_DEFAULT_ROOT) / "include"
                           : sdk / "synapse/include";
    }

    std::vector<std::filesystem::path> ir() {
        const auto sdk = installed();
        if (!sdk.empty())
            return {sdk / "compiler/include"};
        std::vector<std::filesystem::path> paths;
        std::string_view remaining = SYNPS_IR_INCLUDES;
        while (!remaining.empty()) {
            const auto separator = remaining.find('|');
            paths.emplace_back(remaining.substr(0, separator));
            if (separator == std::string_view::npos)
                break;
            remaining.remove_prefix(separator + 1);
        }
        return paths;
    }

    std::vector<std::filesystem::path> headers() {
        std::vector<std::filesystem::path> paths;
        for (const auto &include : ir()) {
            if (!std::filesystem::is_directory(include))
                continue;
            for (const auto &entry : std::filesystem::recursive_directory_iterator(include)) {
                const auto extension = entry.path().extension();
                if (entry.is_regular_file() && (extension == ".h" || extension == ".hpp" ||
                                                extension == ".inc" || extension == ".def"))
                    paths.push_back(entry.path());
            }
        }
        std::ranges::sort(paths);
        return paths;
    }
}
