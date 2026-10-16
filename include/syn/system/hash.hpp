#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <string_view>

namespace Syn::System {
    std::string sha256(std::span<const std::byte> bytes);
    std::string sha256(std::string_view text);
    std::string hash_file(const std::filesystem::path &file);
    std::string hash_tree(const std::filesystem::path &root);
}
