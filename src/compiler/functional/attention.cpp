#include "syn/functional/attention.hpp"

#include "syn/operations.hpp"

#include "mln/sdk/functional.hpp"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

#include <numeric>
#include <optional>
#include <vector>

namespace Syn::Compiler::Functional {
    namespace {
        mlir::Value emit_portable(const Mln::Sdk::Functional::Attention &attention, Operations &ctx,
                                  mlir::OpBuilder &builder, mlir::Location loc,
                                  const Police &police) {
            namespace F = Mln::Sdk::Functional;

            Mln::Sdk::Operations &operations = ctx.expansion_container();
            operations.set_policy_path(attention.policy_path);

            std::vector<std::int64_t> permutation(attention.key.shape.size());
            std::iota(permutation.begin(), permutation.end(), 0);
            std::swap(permutation[permutation.size() - 2], permutation.back());

            Mln::Sdk::Tensor key =
                operations.insert<F::Transpose>(attention.key, std::move(permutation));
            Mln::Sdk::Tensor scores = operations.insert<F::MatMul>(attention.query, std::move(key));

            const Mln::DType dtype = attention.query.dtype.value_or(*police.dtype);
            Mln::Sdk::Tensor scale = operations.insert<F::Const>(
                Mln::Sdk::Weight{"", {}, attention.scale}, false, dtype);
            scale = operations.insert<F::Broadcast>(std::move(scale), scores.shape,
                                                    std::vector<std::int64_t>{}, scores);
            scores = operations.insert<F::Multiply>(std::move(scores), std::move(scale));

            if (attention.mask) {
                std::vector<std::int64_t> dimensions(attention.mask->shape.size());
                if (attention.mask->shape.size() == scores.shape.size()) {
                    std::iota(dimensions.begin(), dimensions.end(), 0);
                } else {
                    const std::int64_t offset = static_cast<std::int64_t>(
                        scores.shape.size() - attention.mask->shape.size());
                    std::iota(dimensions.begin(), dimensions.end(), offset);
                }
                Mln::Sdk::Tensor mask = operations.insert<F::Broadcast>(
                    *attention.mask, scores.shape, std::move(dimensions), scores);
                scores = operations.insert<F::Add>(std::move(scores), std::move(mask));
            }

            scores = operations.insert<F::Convert>(std::move(scores), Mln::DType::Float32);

            const std::int64_t reduction_axis = static_cast<std::int64_t>(scores.shape.size() - 1);
            std::vector<std::int64_t> retained_dimensions(scores.shape.size() - 1);
            std::iota(retained_dimensions.begin(), retained_dimensions.end(), 0);

            Mln::Sdk::Tensor maximum =
                operations.insert<F::ReduceMax>(scores, std::vector{reduction_axis});
            maximum = operations.insert<F::Broadcast>(std::move(maximum), scores.shape,
                                                      retained_dimensions, scores);
            Mln::Sdk::Tensor exponent = operations.insert<F::Exp>(
                operations.insert<F::Subtract>(scores, std::move(maximum)));
            Mln::Sdk::Tensor sum =
                operations.insert<F::ReduceSum>(exponent, std::vector{reduction_axis});
            sum = operations.insert<F::Broadcast>(std::move(sum), scores.shape,
                                                  std::move(retained_dimensions), exponent);
            Mln::Sdk::Tensor probabilities =
                operations.insert<F::Divide>(std::move(exponent), std::move(sum));
            probabilities = operations.insert<F::Convert>(std::move(probabilities), dtype);
            Mln::Sdk::Tensor output =
                operations.insert<F::MatMul>(std::move(probabilities), attention.value);
            return ctx.resolve(output, builder, loc);
        }
    }

    mlir::Value Attention::emit_typed(const Mln::Sdk::Functional::Attention &attention,
                                      Operations &ctx, mlir::OpBuilder &builder,
                                      mlir::Location loc) const {
        const Police police = ctx.police_for(attention.policy_path);
        if (ctx.expand(attention))
            return emit_portable(attention, ctx, builder, loc, police);

        mlir::Value query = ctx.resolve(attention.query, builder, loc);
        mlir::Value key = ctx.resolve(attention.key, builder, loc);
        mlir::Value value = ctx.resolve(attention.value, builder, loc);
        std::optional<mlir::Value> mask;
        if (attention.mask)
            mask = ctx.resolve(*attention.mask, builder, loc);

        auto query_type = mlir::cast<mlir::RankedTensorType>(query.getType());
        auto value_type = mlir::cast<mlir::RankedTensorType>(value.getType());
        std::vector<std::int64_t> output_shape(query_type.getShape());
        output_shape.back() = value_type.getDimSize(value_type.getRank() - 1);
        auto output_type = mlir::RankedTensorType::get(output_shape, value_type.getElementType());

        mlir::SmallVector<mlir::Value> dynamic_sizes;
        for (std::int64_t axis = 0; axis < output_type.getRank(); ++axis) {
            if (!output_type.isDynamicDim(axis))
                continue;
            const bool value_channel = axis == output_type.getRank() - 1;
            dynamic_sizes.push_back(
                mlir::tensor::DimOp::create(builder, loc, value_channel ? value : query,
                                            value_channel ? value_type.getRank() - 1 : axis)
                    .getResult());
        }
        mlir::Value output =
            mlir::tensor::EmptyOp::create(builder, loc, output_shape, value_type.getElementType(),
                                          dynamic_sizes)
                .getResult();
        mlir::Value scale = mlir::arith::ConstantOp::create(
            builder, loc, mlir::FloatAttr::get(query_type.getElementType(), attention.scale));

        const unsigned batch_dimensions = static_cast<unsigned>(query_type.getRank() - 2);
        const unsigned query_dimension = batch_dimensions;
        const unsigned key_dimension = batch_dimensions + 1;
        const unsigned contraction_dimension = batch_dimensions + 2;
        const unsigned value_dimension = batch_dimensions + 3;
        const unsigned loop_count = batch_dimensions + 4;
        auto d = [&](unsigned axis) { return builder.getAffineDimExpr(axis); };
        mlir::SmallVector<mlir::AffineExpr> batch;
        for (unsigned axis = 0; axis < batch_dimensions; ++axis)
            batch.push_back(d(axis));
        const auto mapped = [&](std::initializer_list<unsigned> dimensions) {
            mlir::SmallVector<mlir::AffineExpr> result = batch;
            for (unsigned dimension : dimensions)
                result.push_back(d(dimension));
            return result;
        };
        std::vector<mlir::AffineMap> maps{
            mlir::AffineMap::get(loop_count, 0, mapped({query_dimension, contraction_dimension}),
                                 builder.getContext()),
            mlir::AffineMap::get(loop_count, 0, mapped({key_dimension, contraction_dimension}),
                                 builder.getContext()),
            mlir::AffineMap::get(loop_count, 0, mapped({key_dimension, value_dimension}),
                                 builder.getContext()),
            mlir::AffineMap::get(loop_count, 0, {}, builder.getContext()),
        };
        if (mask) {
            auto mask_type = mlir::cast<mlir::RankedTensorType>(mask->getType());
            maps.push_back(
                mask_type.getRank() == 2
                    ? mlir::AffineMap::get(loop_count, 0, {d(query_dimension), d(key_dimension)},
                                           builder.getContext())
                    : mlir::AffineMap::get(loop_count, 0, mapped({query_dimension, key_dimension}),
                                           builder.getContext()));
        }
        maps.push_back(mlir::AffineMap::get(
            loop_count, 0, mapped({query_dimension, value_dimension}), builder.getContext()));

        auto operation = mlir::iree_compiler::IREE::LinalgExt::AttentionOp::create(
            builder, loc, mlir::TypeRange{output_type}, query, key, value, scale, output,
            builder.getAffineMapArrayAttr(maps), mask);
        mlir::Block *body = builder.createBlock(&operation.getRegion());
        body->addArgument(builder.getF32Type(), loc);
        builder.setInsertionPointToEnd(body);
        mlir::iree_compiler::IREE::LinalgExt::YieldOp::create(builder, loc, body->getArgument(0));
        builder.setInsertionPointAfter(operation);
        return operation.getResult(0);
    }
}
