#pragma once

#include "mln/sdk/operation/compatibility.hpp"
#include "syn/codec_compatibility.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Syn::Compiler {
    struct Manifest;
}

namespace Syn::Compiler::Compatibility {
    enum class Subject { Operation, Codec };

    struct Entry {
            Subject subject = Subject::Operation;
            std::string backend;
            std::vector<std::pair<std::string, std::string>> properties;
            std::string name;
            std::string path;
            bool implemented = false;
            std::size_t uses = 0;
            std::size_t covered = 0;
    };

    class Matrix {
        public:
            explicit Matrix(const Manifest &manifest) : manifest_(manifest) {}

            bool operation(const Mln::Sdk::Operation::Compatibility::Context &context,
                           std::string_view path);
            bool codec(const Codec::Compatibility::Context &context);

            [[nodiscard]] const std::vector<Entry> &entries() const noexcept { return entries_; }
            [[nodiscard]] const Entry *
            find(Subject subject, std::string_view backend, std::string_view name,
                 std::span<const std::pair<std::string, std::string>> properties = {}) const;

        private:
            const Manifest &manifest_;
            std::vector<Entry> entries_;
            std::unordered_map<std::string, std::size_t> index_;

            bool record(Subject subject, std::string_view backend, std::string name,
                        std::string_view path, bool implemented,
                        std::span<const std::pair<std::string, std::string>> properties);
    };
}
