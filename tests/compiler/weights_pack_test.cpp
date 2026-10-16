#include "mln/weight.hpp"
#include "syn/weights_pack.hpp"

#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    std::vector<uint8_t> make_safetensors(const std::string &name, const float (&values)[4]) {
        std::string header =
            "{\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}";
        while (header.size() % 8 != 0)
            header.push_back(' ');
        std::vector<uint8_t> bytes;
        const uint64_t header_size = header.size();
        for (unsigned i = 0; i < 8; ++i)
            bytes.push_back(static_cast<uint8_t>(header_size >> (i * 8)));
        bytes.insert(bytes.end(), header.begin(), header.end());
        const std::size_t offset = bytes.size();
        bytes.resize(offset + sizeof(values));
        std::memcpy(bytes.data() + offset, values, sizeof(values));
        return bytes;
    }
}

int main() {
    using namespace Mln::Sdk;
    namespace Format = Mln::Weight;

    std::string header =
        R"({"used":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"unused":{"dtype":"F32","shape":[4],"data_offsets":[16,32]}})";
    while (header.size() % 8 != 0)
        header.push_back(' ');

    std::vector<uint8_t> source;
    const uint64_t header_size = header.size();
    for (unsigned i = 0; i < 8; ++i)
        source.push_back(static_cast<uint8_t>(header_size >> (i * 8)));
    source.insert(source.end(), header.begin(), header.end());
    const float values[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::size_t data_offset = source.size();
    source.resize(data_offset + sizeof(values));
    std::memcpy(source.data() + data_offset, values, sizeof(values));

    const auto base = std::filesystem::temp_directory_path() /
                      ("myelin-weights-pack-test-" + std::to_string(getpid()));
    const std::string source_path = base.string() + ".safetensors";
    const std::string archive_path = base.string() + ".irpa";
    const std::string shards_path = base.string() + "-shards";
    Weight::write_binary_file(source_path, source);

    try {
        const std::vector<Weight::Descriptor> descriptors = {
            {"used", "F32", "F32", {4}, 16},
            {"unused", "F32", "F32", {4}, 16},
        };
        const std::vector<Weight::Descriptor> requirements = {
            {"used", "F32", {}, {4}, 0},
        };
        const auto no_codec = [](std::string_view) -> Mln::Sdk::Codec & {
            throw std::runtime_error("codec should not be called");
        };
        Mln::Sdk::Codec::Scheme scheme;
        scheme.residency = Mln::Weight::Residency::Volatile;
        const auto volatile_scheme =
            [&scheme](std::string_view) -> const Mln::Sdk::Codec::Scheme & { return scheme; };

        Syn::Compiler::Weight::Conversion::Result converted =
            Syn::Compiler::Weight::Conversion::convert(source_path, descriptors, requirements,
                                                       no_codec, volatile_scheme);
        if (converted.descriptors.size() != 1 || converted.descriptors.front().name != "used" ||
            converted.descriptors.front().residency != Mln::Weight::Residency::Volatile)
            throw std::runtime_error("dead weight was preserved in IRPA");
        if (converted.statistics.source_bytes != 16 ||
            converted.statistics.source_parameters != 2 ||
            converted.statistics.output_parameters != 1 || converted.statistics.output_bytes == 0 ||
            converted.statistics.passthrough_parameters != 1 ||
            converted.statistics.converted_parameters != 0)
            throw std::runtime_error("metricas da conversao mapeada incorretas");

        Weight::write_binary_file(archive_path, converted.data);
        const auto archived = Syn::Compiler::Weight::read(archive_path);
        if (archived.size() != 1 || archived.front().name != "used" ||
            archived.front().residency != Mln::Weight::Residency::Volatile)
            throw std::runtime_error("IRPA index preserved a dead weight");

        const auto resident = [](std::string_view) { return Mln::Weight::Residency::Retained; };
        const auto loading_override = Syn::Compiler::Weight::Conversion::convert(
            source_path, descriptors, requirements, no_codec, volatile_scheme, resident);
        if (loading_override.descriptors.front().residency != Mln::Weight::Residency::Retained)
            throw std::runtime_error("loading policy did not override legacy scheme residency");

        std::filesystem::create_directory(shards_path);
        const float used_values[4] = {1, 2, 3, 4};
        const float unused_values[4] = {5, 6, 7, 8};
        const auto used_shard = make_safetensors("used", used_values);
        const auto unused_shard = make_safetensors("unused", unused_values);
        Weight::write_binary_file(shards_path + "/model-00001-of-00002.safetensors", used_shard);
        Weight::write_binary_file(shards_path + "/model-00002-of-00002.safetensors", unused_shard);

        const auto sharded_descriptors = Syn::Compiler::Weight::read(shards_path);
        if (sharded_descriptors.size() != 2)
            throw std::runtime_error("catalog did not aggregate the shards");
        Syn::Compiler::Weight::Conversion::Result sharded =
            Syn::Compiler::Weight::Conversion::convert(shards_path, sharded_descriptors,
                                                       requirements, no_codec, volatile_scheme);
        if (sharded.descriptors.size() != 1 || sharded.descriptors.front().name != "used" ||
            sharded.statistics.source_parameters != 2 ||
            sharded.statistics.output_parameters != 1 || sharded.statistics.source_bytes != 16)
            throw std::runtime_error("incorrect sharded-checkpoint conversion");
    } catch (...) {
        std::filesystem::remove(source_path);
        std::filesystem::remove(archive_path);
        std::filesystem::remove_all(shards_path);
        throw;
    }

    std::filesystem::remove(source_path);
    std::filesystem::remove(archive_path);
    std::filesystem::remove_all(shards_path);
    return 0;
}
