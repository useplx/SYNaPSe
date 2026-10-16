#include "mln/package.hpp"

#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 4)
        return 2;

    const auto package = Mln::Package::File::open(argv[1]);
    std::ifstream input(argv[2], std::ios::binary);
    if (!input)
        throw std::runtime_error("failed to open replacement VMFB");

    std::vector<std::uint8_t> vmfb(std::istreambuf_iterator<char>(input), {});
    Mln::Package::Builder builder;

    for (std::uint32_t value = static_cast<std::uint32_t>(Mln::Package::Section::Type::Devices);
         value <= static_cast<std::uint32_t>(Mln::Package::Section::Type::Hooks); ++value) {
        const auto type = static_cast<Mln::Package::Section::Type>(value);
        for (const Mln::Package::Section &section : package->sections(type)) {
            if (type == Mln::Package::Section::Type::Vmfb && section.name == "model") {
                if (vmfb.size() > section.bytes.size())
                    throw std::runtime_error("replacement VMFB exceeds the original section");
                vmfb.resize(section.bytes.size());
                builder.add(type, std::string(section.name), vmfb);
            } else {
                builder.add(type, std::string(section.name), section.bytes);
            }
        }
    }

    builder.write(argv[3]);
    return 0;
}
