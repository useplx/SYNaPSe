#pragma once

#include "mln/sdk/codec.hpp"
#include "mln/weight.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace Syn::Compiler::Weight {
    namespace Quant {
        using Lookup = std::function<Mln::Sdk::Codec &(std::string_view)>;
    }

    namespace Scheme {
        using Lookup = std::function<const Mln::Sdk::Codec::Scheme &(std::string_view)>;
    }

    namespace Residency {
        using Lookup = std::function<Mln::Weight::Residency(std::string_view)>;
    }

    std::vector<Mln::Weight::Descriptor> read(std::string_view path);

    namespace Conversion {
        struct Statistics {
                uint64_t source_bytes = 0;
                uint64_t output_bytes = 0;

                uint64_t source_parameters = 0;
                uint64_t output_parameters = 0;

                uint64_t converted_parameters = 0;
                uint64_t passthrough_parameters = 0;
        };

        struct Result {
                struct Resource {
                        std::string name;
                        uint64_t offset = 0;
                        uint64_t length = 0;
                };

                std::vector<Mln::Weight::Descriptor> descriptors;

                std::vector<Resource> resources;

                Statistics statistics;

                std::vector<uint8_t> data;
        };

        Result empty();

        Result inspect(std::span<const uint8_t> archive);

        Result convert(std::string_view path,
                       const std::vector<Mln::Weight::Descriptor> &descriptors,
                       const std::vector<Mln::Weight::Descriptor> &requirements,
                       const Quant::Lookup &quants, const Scheme::Lookup &schemes,
                       const Residency::Lookup &residency = {});
    }
}
