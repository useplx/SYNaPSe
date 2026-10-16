#include "syn/system/hash.hpp"
#include "syn/error.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <format>
#include <fstream>
#include <iterator>
#include <vector>

namespace Syn::System {
    namespace {
        class Sha256 {
            public:
                void update(std::span<const std::byte> input) {
                    total_ += input.size();
                    while (!input.empty()) {
                        const std::size_t count = std::min(input.size(), block_.size() - buffered_);
                        std::copy_n(input.begin(), count, block_.begin() + buffered_);
                        buffered_ += count;
                        input = input.subspan(count);
                        if (buffered_ == block_.size()) {
                            transform(block_);
                            buffered_ = 0;
                        }
                    }
                }

                std::array<std::byte, 32> finish() {
                    const std::uint64_t bits = total_ * 8;
                    block_[buffered_++] = std::byte{0x80};
                    if (buffered_ > 56) {
                        std::fill(block_.begin() + buffered_, block_.end(), std::byte{});
                        transform(block_);
                        buffered_ = 0;
                    }
                    std::fill(block_.begin() + buffered_, block_.begin() + 56, std::byte{});
                    for (std::size_t index = 0; index < 8; ++index)
                        block_[63 - index] = std::byte((bits >> (index * 8)) & 0xff);
                    transform(block_);

                    std::array<std::byte, 32> result{};
                    for (std::size_t word = 0; word < state_.size(); ++word)
                        for (std::size_t byte = 0; byte < 4; ++byte)
                            result[word * 4 + byte] =
                                std::byte((state_[word] >> ((3 - byte) * 8)) & 0xff);
                    return result;
                }

            private:
                inline static constexpr std::array<std::uint32_t, 64> constants{{
                    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
                    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
                    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
                    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
                    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
                    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
                    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
                    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
                    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
                    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
                    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
                }};
                std::array<std::uint32_t, 8> state_{{
                    0x6a09e667,
                    0xbb67ae85,
                    0x3c6ef372,
                    0xa54ff53a,
                    0x510e527f,
                    0x9b05688c,
                    0x1f83d9ab,
                    0x5be0cd19,
                }};
                std::array<std::byte, 64> block_{};
                std::size_t buffered_ = 0;
                std::uint64_t total_ = 0;

                void transform(const std::array<std::byte, 64> &block) {
                    std::array<std::uint32_t, 64> words{};
                    for (std::size_t index = 0; index < 16; ++index)
                        for (std::size_t byte = 0; byte < 4; ++byte)
                            words[index] |= std::uint32_t(std::to_integer<unsigned char>(
                                                block[index * 4 + byte]))
                                            << ((3 - byte) * 8);
                    for (std::size_t index = 16; index < words.size(); ++index) {
                        const std::uint32_t s0 = std::rotr(words[index - 15], 7) ^
                                                 std::rotr(words[index - 15], 18) ^
                                                 (words[index - 15] >> 3);
                        const std::uint32_t s1 = std::rotr(words[index - 2], 17) ^
                                                 std::rotr(words[index - 2], 19) ^
                                                 (words[index - 2] >> 10);
                        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
                    }
                    auto [a, b, c, d, e, f, g, h] = state_;
                    for (std::size_t index = 0; index < words.size(); ++index) {
                        const std::uint32_t upper =
                            std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
                        const std::uint32_t choose = (e & f) ^ (~e & g);
                        const std::uint32_t first =
                            h + upper + choose + constants[index] + words[index];
                        const std::uint32_t lower =
                            std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
                        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
                        const std::uint32_t second = lower + majority;
                        h = g;
                        g = f;
                        f = e;
                        e = d + first;
                        d = c;
                        c = b;
                        b = a;
                        a = first + second;
                    }
                    state_[0] += a;
                    state_[1] += b;
                    state_[2] += c;
                    state_[3] += d;
                    state_[4] += e;
                    state_[5] += f;
                    state_[6] += g;
                    state_[7] += h;
                }
        };

        std::string hexadecimal(std::span<const std::byte> bytes) {
            std::string output;
            output.reserve(bytes.size() * 2);
            for (const std::byte byte : bytes)
                std::format_to(std::back_inserter(output), "{:02x}",
                               std::to_integer<unsigned int>(byte));
            return output;
        }

        bool ignored(const std::filesystem::path &relative) {
            for (const auto &build_package : relative)
                if (build_package == ".git" || build_package == "build" ||
                    build_package == ".synps")
                    return true;
            return relative == "synps.lock";
        }
    }

    std::string sha256(std::span<const std::byte> bytes) {
        Sha256 digest;
        digest.update(bytes);
        const auto result = digest.finish();
        return hexadecimal(result);
    }

    std::string sha256(std::string_view text) {
        return sha256(std::as_bytes(std::span(text.data(), text.size())));
    }

    std::string hash_file(const std::filesystem::path &file) {
        std::ifstream input(file, std::ios::binary);
        if (!input)
            Errors::at(Error::File::Unreadable, file.string(), file.string(),
                       "could not open for reading");
        Sha256 digest;
        std::array<char, 65536> buffer{};
        while (input) {
            input.read(buffer.data(), buffer.size());
            digest.update(
                std::as_bytes(std::span(buffer.data(), static_cast<std::size_t>(input.gcount()))));
        }
        return hexadecimal(digest.finish());
    }

    std::string hash_tree(const std::filesystem::path &root) {
        std::vector<std::filesystem::path> files;
        for (std::filesystem::recursive_directory_iterator iterator(root), end; iterator != end;
             ++iterator) {
            const auto relative = iterator->path().lexically_relative(root);
            if (ignored(relative)) {
                if (iterator->is_directory())
                    iterator.disable_recursion_pending();
                continue;
            }
            if (iterator->is_regular_file())
                files.push_back(relative);
        }
        std::sort(files.begin(), files.end());
        Sha256 digest;
        for (const auto &relative : files) {
            const std::string name = relative.generic_string();
            digest.update(std::as_bytes(std::span(name.data(), name.size())));
            const std::byte separator{};
            digest.update(std::span(&separator, 1));
            std::ifstream input(root / relative, std::ios::binary);
            std::array<char, 65536> buffer{};
            while (input) {
                input.read(buffer.data(), buffer.size());
                digest.update(std::as_bytes(
                    std::span(buffer.data(), static_cast<std::size_t>(input.gcount()))));
            }
        }
        return hexadecimal(digest.finish());
    }
}
