#include "syn/functional/upsample.hpp"
#include "mlir/IR/BuiltinTypes.h"
#include "mln/sdk/functional/upsample.hpp"
#include "stablehlo/dialect/StablehloOps.h"
#include "syn/operations.hpp"

namespace Syn::Compiler::Functional {
    mlir::Value Upsample::emit_typed(const Mln::Sdk::Functional::Upsample &upsample,
                                     Operations &ctx, mlir::OpBuilder &builder,
                                     mlir::Location loc) const {
        mlir::Value x = ctx.resolve(upsample.x, builder, loc);
        const auto &scales = upsample.scale;

        auto x_type = mlir::cast<mlir::RankedTensorType>(x.getType());
        auto shape = x_type.getShape();
        auto elem = x_type.getElementType();

        int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
        int64_t sh = scales[0], sw = scales[1];

        auto s1_type = mlir::RankedTensorType::get({N, C, H, 1, W, 1}, elem);
        auto s1 = mlir::stablehlo::ReshapeOp::create(builder, loc, s1_type, x).getResult();

        auto s2_type = mlir::RankedTensorType::get({N, C, H, sh, W, sw}, elem);
        auto s2 = mlir::stablehlo::BroadcastInDimOp::create(
                      builder, loc, s2_type, s1, llvm::ArrayRef<int64_t>{0, 1, 2, 3, 4, 5})
                      .getResult();

        auto s3_type = mlir::RankedTensorType::get({N, C, H * sh, W * sw}, elem);
        return mlir::stablehlo::ReshapeOp::create(builder, loc, s3_type, s2).getResult();
    }
}
