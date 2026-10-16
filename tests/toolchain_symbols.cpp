#include "test.hpp"

#include "syn/error.hpp"
#include "syn/toolchain/symbols.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace {
    const std::vector<uint8_t> WASM_MODULE = {
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x04, 0x01, 0x60, 0x00, 0x00,

        0x03, 0x02, 0x01, 0x00, 0x07, 0x11, 0x02, 0x07, 'f',  'o',  'r',  'w',  'a',  'r',
        'd',  0x00, 0x00, 0x05, 'o',  'u',  't',  'r',  'a',  0x00, 0x00, 0x0a, 0x04, 0x01,
        0x02, 0x00, 0x0b,
    };

    std::filesystem::path write_temporary(const std::string &name,
                                          const std::vector<uint8_t> &bytes) {
        const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        return path;
    }
}

SYNPS_TEST(reads_wasm_export_section) {
    const std::filesystem::path module = write_temporary("synapse_symbols_test.wasm", WASM_MODULE);
    const std::vector<std::string> symbols = Syn::Toolchain::Symbols::exported(module);

    Syn::Testing::require(symbols.size() == 2, "expected two exports");
    Syn::Testing::require(symbols[0] == "forward", "expected `forward`");
    Syn::Testing::require(symbols[1] == "outra", "expected `outra`");

    Syn::Toolchain::Symbols::require(module, {"forward", "outra"}, "teste");
    std::filesystem::remove(module);
}

SYNPS_TEST(rejects_artifact_missing_a_required_export) {
    const std::filesystem::path module =
        write_temporary("synapse_symbols_missing.wasm", WASM_MODULE);
    bool rejected = false;
    try {
        Syn::Toolchain::Symbols::require(module, {"forward", "generate"}, "teste");
    } catch (const Syn::Diagnostic &) {
        rejected = true;
    }
    std::filesystem::remove(module);
    Syn::Testing::require(rejected, "expected the missing export to be rejected");
}

SYNPS_TEST(ignores_formats_it_cannot_read) {
    const std::filesystem::path archive =
        write_temporary("synapse_symbols_test.a", {'!', '<', 'a', 'r', 'c', 'h', '>', '\n'});
    Syn::Toolchain::Symbols::require(archive, {"mln_native_module"}, "teste");
    std::filesystem::remove(archive);
}

SYNPS_TEST(reads_wasm_export_with_overflowing_name_length) {
    const std::vector<uint8_t> module = {
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x07, 0x0b, 0x01,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01,
    };
    const auto path = std::filesystem::temp_directory_path() / "synapse-wasm-overflow.wasm";
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(module.data()),
                   static_cast<std::streamsize>(module.size()));
    }
    const std::vector<std::string> symbols = Syn::Toolchain::Symbols::exported(path);
    Syn::Testing::require(symbols.empty(), "malformed export length produced a symbol");
    std::filesystem::remove(path);
}

SYNPS_TEST(reads_elf_with_overflowing_section_offset) {
    std::vector<uint8_t> elf(64, 0);
    elf[0] = 0x7f;
    elf[1] = 'E';
    elf[2] = 'L';
    elf[3] = 'F';
    elf[4] = 2;
    const uint64_t shoff = ~uint64_t{0} - 63;
    std::memcpy(elf.data() + 40, &shoff, sizeof(shoff));
    const uint16_t shentsize = 64;
    const uint16_t shnum = 1;
    std::memcpy(elf.data() + 58, &shentsize, sizeof(shentsize));
    std::memcpy(elf.data() + 60, &shnum, sizeof(shnum));

    const auto path = std::filesystem::temp_directory_path() / "synapse-elf-overflow.so";
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(elf.data()),
                   static_cast<std::streamsize>(elf.size()));
    }
    const std::vector<std::string> symbols = Syn::Toolchain::Symbols::exported(path);
    Syn::Testing::require(symbols.empty(), "malformed section offset produced a symbol");
    std::filesystem::remove(path);
}
