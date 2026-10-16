#include "syn/functional/reducemax.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mln/sdk/functional/reducemax.hpp"
#include "stablehlo/dialect/StablehloOps.h"
#include "syn/operations.hpp"
#include <limits>
#include <vector>

namespace Syn::Compiler::Functional {
    namespace {
        mlir::Value emit_reduce(mlir::OpBuilder &builder, mlir::Location location,
                                mlir::Value input, mlir::Type element,
                                const std::vector<int64_t> &axes,
                                const std::vector<int64_t> &result_shape) {
            mlir::TypedAttr fill;
            if (mlir::isa<mlir::FloatType>(element)) {
                fill = mlir::FloatAttr::get(element, -std::numeric_limits<float>::infinity());
            } else {
                const auto integer = mlir::cast<mlir::IntegerType>(element);
                fill = mlir::IntegerAttr::get(
                    element, integer.isUnsigned() ? 0 : std::numeric_limits<int64_t>::min());
            }
            const auto scalar_type = mlir::RankedTensorType::get({}, element);
            mlir::Value initial =
                mlir::stablehlo::ConstantOp::create(builder, location,
                                                    mlir::DenseElementsAttr::get(scalar_type, fill))
                    .getResult();
            auto reduction = mlir::stablehlo::ReduceOp::create(
                builder, location,
                mlir::TypeRange{mlir::RankedTensorType::get(result_shape, element)},
                mlir::ValueRange{input}, mlir::ValueRange{initial},
                mlir::DenseI64ArrayAttr::get(builder.getContext(), axes));
            mlir::OpBuilder::InsertionGuard guard(builder);
            mlir::Block *block = builder.createBlock(
                &reduction.getBody(), {}, {scalar_type, scalar_type}, {location, location});
            builder.setInsertionPointToStart(block);
            auto maximum = mlir::stablehlo::MaxOp::create(builder, location, block->getArgument(0),
                                                          block->getArgument(1));
            mlir::stablehlo::ReturnOp::create(builder, location, maximum.getResult());
            return reduction.getResult(0);
        }
    }

    mlir::Value ReduceMax::emit_typed(const Mln::Sdk::Functional::ReduceMax &reducemax,
                                      Operations &ctx, mlir::OpBuilder &builder,
                                      mlir::Location loc) const {
        mlir::Value x = ctx.resolve(reducemax.x, builder, loc);
        const auto &axes = reducemax.axes;

        auto x_type = mlir::cast<mlir::RankedTensorType>(x.getType());
        auto elem = x_type.getElementType();
        auto shape = x_type.getShape();

        std::vector<int64_t> result_shape;
        for (std::size_t i = 0; i < shape.size(); i++) {
            bool reduced = false;
            for (int64_t axis : axes)
                reduced = reduced || axis == static_cast<int64_t>(i);
            if (!reduced)
                result_shape.push_back(shape[i]);
        }
        return emit_reduce(builder, loc, x, elem, axes, result_shape);
    }
}
