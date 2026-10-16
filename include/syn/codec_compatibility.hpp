#pragma once

#include "mln/sdk/codec.hpp"
#include "mln/sdk/operation.hpp"
#include "mln/sdk/operations.hpp"

#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace Syn::Compiler::Codec {
    class Compatibility {
        public:
            struct Context {
                    Mln::Sdk::Codec &codec;
                    const Mln::Sdk::Codec::Scheme &scheme;
                    const Mln::Sdk::Weight &weight;
                    Mln::DType dtype;
                    std::string_view parameter_path;
                    std::string_view backend;
                    std::span<const std::pair<std::string, std::string>> properties;
                    std::span<const Mln::Sdk::Operation *const> consumers;
            };

            virtual ~Compatibility() = default;

            virtual int priority() const { return 0; }

            virtual bool applies(const Context &context) const = 0;

            virtual Mln::Sdk::Tensor adapt(Mln::Sdk::Operations &, const Context &,
                                           Mln::Sdk::Tensor decoded) const {
                return decoded;
            }
    };
}
