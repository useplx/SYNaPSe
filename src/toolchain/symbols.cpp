#include "syn/toolchain/symbols.hpp"

#include "syn/error.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>

namespace Syn::Toolchain {
    namespace {
        std::vector<uint8_t> read(const std::filesystem::path &path) {
            std::ifstream file(path, std::ios::binary);
            if (!file)
                return {};
            return std::vector<uint8_t>(std::istreambuf_iterator<char>(file),
                                        std::istreambuf_iterator<char>());
        }

        template <typename T> T at(const std::vector<uint8_t> &bytes, std::uint64_t offset) {
            T value{};
            if (offset > bytes.size() || sizeof(T) > bytes.size() - offset)
                return value;
            std::memcpy(&value, bytes.data() + offset, sizeof(T));
            return value;
        }

        std::string cstring(const std::vector<uint8_t> &bytes, std::uint64_t offset) {
            if (offset >= bytes.size())
                return {};
            const auto *begin = reinterpret_cast<const char *>(bytes.data() + offset);
            const std::size_t limit = bytes.size() - offset;
            return std::string(begin, ::strnlen(begin, limit));
        }

        template <typename Header, typename Section, typename Symbol>
        std::vector<std::string> elf_symbols(const std::vector<uint8_t> &bytes) {
            std::vector<std::string> symbols;
            const Header header = at<Header>(bytes, 0);
            if (header.e_shentsize != sizeof(Section))
                return symbols;

            for (std::size_t index = 0; index < header.e_shnum; ++index) {
                const Section section =
                    at<Section>(bytes, header.e_shoff + index * sizeof(Section));
                constexpr uint32_t SHT_DYNSYM = 11;
                if (section.sh_type != SHT_DYNSYM)
                    continue;
                if (section.sh_entsize != sizeof(Symbol))
                    continue;

                const Section strings =
                    at<Section>(bytes, header.e_shoff + section.sh_link * sizeof(Section));
                const std::size_t count =
                    section.sh_entsize == 0 ? 0 : section.sh_size / section.sh_entsize;
                for (std::size_t item = 0; item < count; ++item) {
                    const Symbol symbol =
                        at<Symbol>(bytes, section.sh_offset + item * sizeof(Symbol));
                    if (symbol.st_shndx == 0)
                        continue;
                    const uint8_t binding = symbol.st_info >> 4;
                    if (binding != 1 && binding != 2)
                        continue;
                    std::string name = cstring(bytes, strings.sh_offset + symbol.st_name);
                    if (!name.empty())
                        symbols.push_back(std::move(name));
                }
            }
            return symbols;
        }

        namespace Elf64 {
            struct Header {
                    uint8_t e_ident[16];
                    uint16_t e_type, e_machine;
                    uint32_t e_version;
                    uint64_t e_entry, e_phoff, e_shoff;
                    uint32_t e_flags;
                    uint16_t e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx;
            };
            struct Section {
                    uint32_t sh_name, sh_type;
                    uint64_t sh_flags, sh_addr, sh_offset, sh_size;
                    uint32_t sh_link, sh_info;
                    uint64_t sh_addralign, sh_entsize;
            };
            struct Symbol {
                    uint32_t st_name;
                    uint8_t st_info, st_other;
                    uint16_t st_shndx;
                    uint64_t st_value, st_size;
            };
        }

        namespace Elf32 {
            struct Header {
                    uint8_t e_ident[16];
                    uint16_t e_type, e_machine;
                    uint32_t e_version, e_entry, e_phoff, e_shoff, e_flags;
                    uint16_t e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx;
            };
            struct Section {
                    uint32_t sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link,
                        sh_info, sh_addralign, sh_entsize;
            };
            struct Symbol {
                    uint32_t st_name, st_value, st_size;
                    uint8_t st_info, st_other;
                    uint16_t st_shndx;
            };
        }

        uint64_t leb128(const std::vector<uint8_t> &bytes, std::size_t &offset) {
            uint64_t value = 0;
            uint32_t shift = 0;
            while (offset < bytes.size()) {
                const uint8_t byte = bytes[offset++];
                value |= static_cast<uint64_t>(byte & 0x7f) << shift;
                if ((byte & 0x80) == 0)
                    break;
                shift += 7;
                if (shift > 63)
                    break;
            }
            return value;
        }

        std::vector<std::string> wasm_symbols(const std::vector<uint8_t> &bytes) {
            std::vector<std::string> symbols;
            std::size_t offset = 8;
            while (offset < bytes.size()) {
                const uint8_t id = bytes[offset++];
                const uint64_t size = leb128(bytes, offset);
                if (size > bytes.size() - offset)
                    break;
                const std::size_t end = offset + size;
                if (id != 7) {
                    offset = end;
                    continue;
                }
                const uint64_t count = leb128(bytes, offset);
                for (uint64_t item = 0; item < count && offset < end; ++item) {
                    const uint64_t length = leb128(bytes, offset);
                    if (length > end - offset)
                        break;
                    symbols.emplace_back(reinterpret_cast<const char *>(bytes.data() + offset),
                                         length);
                    offset += length;
                    ++offset;
                    leb128(bytes, offset);
                }
                offset = end;
            }
            return symbols;
        }
    }

    namespace Symbols {
        std::vector<std::string> exported(const std::filesystem::path &artifact) {
            const std::vector<uint8_t> bytes = read(artifact);
            if (bytes.size() < 8)
                return {};

            if (std::memcmp(bytes.data(),
                            "\x7f"
                            "ELF",
                            4) == 0)
                return bytes[4] == 2
                           ? elf_symbols<Elf64::Header, Elf64::Section, Elf64::Symbol>(bytes)
                           : elf_symbols<Elf32::Header, Elf32::Section, Elf32::Symbol>(bytes);

            if (std::memcmp(bytes.data(), "\0asm", 4) == 0)
                return wasm_symbols(bytes);

            return {};
        }

        void require(const std::filesystem::path &artifact,
                     const std::vector<std::string> &required, const std::string &origin) {
            if (required.empty())
                return;
            const std::vector<std::string> present = exported(artifact);
            if (present.empty())
                return;

            std::vector<std::string> missing;
            for (const std::string &symbol : required)
                if (std::find(present.begin(), present.end(), symbol) == present.end())
                    missing.push_back(symbol);
            if (missing.empty())
                return;

            std::string list;
            for (const std::string &symbol : missing)
                list += (list.empty() ? "" : ", ") + symbol;
            Errors::at(Error::Exports::Missing, origin, artifact.filename(), list);
        }
    }
}
