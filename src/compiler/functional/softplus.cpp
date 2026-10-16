#include "syn/functional/unary.hpp"

#include "syn/operations.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace Syn::Compiler::Functional {
    mlir::Value Softplus::emit_typed(const Mln::Sdk::Functional::Softplus &node, Operations &ctx,
                                     mlir::OpBuilder &builder, mlir::Location loc) const {
        mlir::Value input = ctx.resolve(node.x, builder, loc);
        auto type = mlir::cast<mlir::RankedTensorType>(input.getType());

        std::vector<mlir::OpFoldResult> sizes;
        sizes.reserve(static_cast<std::size_t>(type.getRank()));
        for (int64_t dimension = 0; dimension < type.getRank(); ++dimension) {
            if (type.isDynamicDim(dimension)) {
                sizes.push_back(
                    mlir::tensor::DimOp::create(builder, loc, input, dimension).getResult());
            } else {
                sizes.push_back(builder.getIndexAttr(type.getDimSize(dimension)));
            }
        }

        mlir::Value zero = mlir::arith::ConstantOp::create(
            builder, loc, type.getElementType(), builder.getZeroAttr(type.getElementType()));
        mlir::Value empty =
            mlir::tensor::EmptyOp::create(builder, loc, sizes, type.getElementType()).getResult();
        mlir::Value zeros = mlir::linalg::FillOp::create(builder, loc, zero, empty).getResult(0);

        mlir::Value positive =
            mlir::stablehlo::MaxOp::create(builder, loc, input, zeros).getResult();
        mlir::Value magnitude = mlir::stablehlo::AbsOp::create(builder, loc, input).getResult();
        mlir::Value negative = mlir::stablehlo::NegOp::create(builder, loc, magnitude).getResult();
        mlir::Value exponential =
            mlir::stablehlo::ExpOp::create(builder, loc, negative).getResult();
        mlir::Value correction =
            mlir::stablehlo::Log1pOp::create(builder, loc, exponential).getResult();
        return mlir::stablehlo::AddOp::create(builder, loc, positive, correction).getResult();
    }
}
