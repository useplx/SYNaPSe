#include "syn/operations.hpp"
#include "mln/error.hpp"
#include "mln/sdk/functional/const.hpp"
#include "syn/dynamic_shape.hpp"
#include "syn/errors/graph.hpp"

#include <mlir/IR/BuiltinTypes.h>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace Syn::Compiler {
    Mln::Sdk::Tensor Operations::decode_weight(const Mln::Sdk::Functional::Const &constant) {
        const Police police = police_for(constant.weight.name);
        const auto &scheme = select_scheme(constant.weight);
        auto &codec = dynamic_cast<Mln::Sdk::Codec &>(scheme.format.get());

        expansions.set_policy_path(constant.weight.name);
        Mln::Sdk::Tensor expanded =
            codec.decode(expansions, constant.weight, scheme, *police.dtype);

        const Codec::Compatibility::Context context{
            .codec = codec,
            .scheme = scheme,
            .weight = constant.weight,
            .dtype = *police.dtype,
            .parameter_path = constant.weight.name,
            .backend = police.backend ? std::string_view(police.backend->device) : "none",
            .properties = police.backend ? std::span<const std::pair<std::string, std::string>>(
                                               police.backend->flags)
                                         : std::span<const std::pair<std::string, std::string>>{},
            .consumers = consumer_chain(),
        };
        matrix_.codec(context);
        for (Codec::Compatibility *compatibility : manifest_.codec_compatibilities) {
            if (compatibility && compatibility->applies(context))
                expanded = compatibility->adapt(expansions, context, std::move(expanded));
        }

        require_weights(constant.weight, codec, expansions);
        return expanded;
    }

    const Mln::Sdk::Codec::Scheme &Operations::select_scheme(const Mln::Sdk::Weight &weight) const {
        const auto found = std::find_if(
            available_weights_.begin(), available_weights_.end(),
            [&](const Mln::Weight::Descriptor &entry) { return entry.name == weight.name; });

        if (found != available_weights_.end() && found->shape != weight.shape) {
            raise(Errors::Graph::Weight::Shape::Mismatch, weight.name);
        }

        const std::string storage = found == available_weights_.end()
                                        ? std::string{}
                                        : (found->storage.empty() ? found->quant : found->storage);

        const bool source_float =
            storage.empty() || storage == "F32" || storage == "F16" || storage == "BF16";

        const auto match = scheme_for(manifest_, weight.name, Mln::Sdk::Codec::Scope::Constant,
                                      Mln::Sdk::Codec::Lifetime::Immutable);

        if (!match) {
            raise(Errors::Graph::Weight::Policy::Missing, weight.name);
        }

        const auto *scheme = match.scheme;
        auto *codec = dynamic_cast<Mln::Sdk::Codec *>(&scheme->format.get());

        if (!codec || !codec->supports(weight, *scheme)) {
            raise(Errors::Graph::Weight::Format::Unsupported, weight.name);
        }

        if (!source_float && codec->storage_name() != storage) {
            raise(Errors::Graph::Weight::Storage::Mismatch, weight.name, storage,
                  codec->storage_name());
        }

        return *scheme;
    }

    void Operations::require_weights(const Mln::Sdk::Weight &weight, const Mln::Sdk::Codec &codec,
                                     const Mln::Sdk::Operations &decoded) {
        const std::string quant = codec.storage_name();
        std::vector<const Mln::Sdk::Functional::Const *> components;
        const std::string prefix = weight.name + ".";
        for (const auto &operation : decoded) {
            if (operation->kind != Mln::Sdk::Functional::Kind::Const)
                continue;
            const auto &constant = static_cast<const Mln::Sdk::Functional::Const &>(*operation);
            if (constant.quantize || constant.weight.value.has_value())
                continue;
            if (constant.weight.name == weight.name || constant.weight.name.starts_with(prefix)) {
                if (std::any_of(components.begin(), components.end(), [&](const auto *existing) {
                        return existing->weight.name == constant.weight.name;
                    }))
                    continue;
                components.push_back(&constant);
            }
        }

        if (components.empty())
            raise(Errors::Graph::Codec::Storage::Missing, codec.name, weight.name);

        const bool composite =
            components.size() != 1 || components.front()->weight.name != weight.name;
        for (const auto *component : components) {
            const std::string &name = component->weight.name;
            auto found = std::find_if(
                weights_.begin(), weights_.end(),
                [&](const Mln::Weight::Descriptor &entry) { return entry.name == name; });

            const std::string storage = composite ? weight.name : std::string{};
            if (found != weights_.end()) {
                if (found->quant != quant || found->shape != component->weight.shape ||
                    found->storage != storage)
                    raise(Errors::Graph::Codec::Requirements::Conflict, name);
                continue;
            }

            weights_.push_back(Mln::Weight::Descriptor{
                .name = name,
                .quant = quant,
                .storage = storage,
                .shape = component->weight.shape,
                .byte_length = 0,
            });
        }
    }

    mlir::Type dtype_to_mlir_type(Mln::DType dtype, mlir::MLIRContext &ctx) {
        switch (dtype) {
        case Mln::DType::Float32:
            return mlir::Float32Type::get(&ctx);
        case Mln::DType::Float16:
            return mlir::Float16Type::get(&ctx);
        case Mln::DType::BFloat16:
            return mlir::BFloat16Type::get(&ctx);
        case Mln::DType::Int32:
            return mlir::IntegerType::get(&ctx, 32);
        case Mln::DType::Int16:
            return mlir::IntegerType::get(&ctx, 16);
        case Mln::DType::Int8:
            return mlir::IntegerType::get(&ctx, 8);
        case Mln::DType::Int4:
            return mlir::IntegerType::get(&ctx, 4);
        case Mln::DType::UInt64:
            return mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Unsigned);
        case Mln::DType::UInt32:
            return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Unsigned);
        case Mln::DType::UInt8:
            return mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Unsigned);
        case Mln::DType::UInt4:
            return mlir::IntegerType::get(&ctx, 4, mlir::IntegerType::Unsigned);
        }

        return mlir::Float32Type::get(&ctx);
    }

    mlir::Type tensor_type(const Mln::Sdk::Tensor &t, Mln::DType dtype, mlir::MLIRContext &ctx) {
        if (t.dtype)
            dtype = *t.dtype;

        return mlir::RankedTensorType::get(mlir_shape(t.shape), dtype_to_mlir_type(dtype, ctx));
    }

    int64_t dtype_byte_width(Mln::DType dtype) {
        switch (dtype) {
        case Mln::DType::Float32:
            return 4;
        case Mln::DType::Float16:
            return 2;
        case Mln::DType::BFloat16:
            return 2;
        case Mln::DType::Int32:
            return 4;
        case Mln::DType::Int16:
            return 2;
        case Mln::DType::Int8:
            return 1;
        case Mln::DType::Int4:
            return 1;
        case Mln::DType::UInt64:
            return 8;
        case Mln::DType::UInt32:
            return 4;
        case Mln::DType::UInt8:
            return 1;
        case Mln::DType::UInt4:
            return 1;
        }

        return 4;
    }

}
