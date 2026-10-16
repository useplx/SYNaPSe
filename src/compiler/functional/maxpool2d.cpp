#include "syn/functional/maxpool2d.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mln/sdk/functional/maxpool2d.hpp"
#include "stablehlo/dialect/StablehloOps.h"
#include "syn/operations.hpp"
#include <limits>

namespace Syn::Compiler::Functional {
    mlir::Value MaxPool2D::emit_typed(const Mln::Sdk::Functional::MaxPool2D &maxpool2d,
                                      Operations &ctx, mlir::OpBuilder &builder,
                                      mlir::Location loc) const {
        mlir::Value x = ctx.resolve(maxpool2d.x, builder, loc);
        const auto &kernel = maxpool2d.kernel;
        const auto &stride = maxpool2d.stride;
        const auto &padding = maxpool2d.padding;

        auto x_type = mlir::cast<mlir::RankedTensorType>(x.getType());
        auto shape = x_type.getShape();
        auto elem = x_type.getElementType();
        auto mlir_context = builder.getContext();

        int64_t N = shape[0], Ck = shape[1], H = shape[2], W = shape[3];
        int64_t kernel_height = kernel[0], kernel_width = kernel[1];
        int64_t stride_height = stride[0], stride_width = stride[1];
        int64_t padding_top = padding[0], padding_bottom = padding[1], padding_left = padding[2],
                padding_right = padding[3];

        int64_t output_height =
            (H + padding_top + padding_bottom - kernel_height) / stride_height + 1;
        int64_t output_width = (W + padding_left + padding_right - kernel_width) / stride_width + 1;
        mlir::TypedAttr init_fill;
        if (mlir::isa<mlir::FloatType>(elem)) {
            init_fill = mlir::FloatAttr::get(elem, -std::numeric_limits<float>::infinity());
        } else {
            const auto integer = mlir::cast<mlir::IntegerType>(elem);
            init_fill = mlir::IntegerAttr::get(
                elem, integer.isUnsigned() ? 0 : std::numeric_limits<int64_t>::min());
        }
        auto initial_scalar = mlir::RankedTensorType::get({}, elem);
        auto initial_value =
            mlir::stablehlo::ConstantOp::create(
                builder, loc, mlir::DenseElementsAttr::get(initial_scalar, init_fill))
                .getResult();

        mlir::Value padded = mlir::stablehlo::PadOp::create(
                                 builder, loc, x, initial_value,
                                 llvm::ArrayRef<int64_t>{0, 0, padding_top, padding_left},
                                 llvm::ArrayRef<int64_t>{0, 0, padding_bottom, padding_right},
                                 llvm::ArrayRef<int64_t>{0, 0, 0, 0})
                                 .getResult();

        mlir::Value pooled;
        for (int64_t kh = 0; kh < kernel_height; ++kh) {
            for (int64_t kw = 0; kw < kernel_width; ++kw) {
                auto window =
                    mlir::stablehlo::SliceOp::create(
                        builder, loc, padded,
                        mlir::DenseI64ArrayAttr::get(mlir_context, {0, 0, kh, kw}),
                        mlir::DenseI64ArrayAttr::get(
                            mlir_context, {N, Ck, kh + (output_height - 1) * stride_height + 1,
                                           kw + (output_width - 1) * stride_width + 1}),
                        mlir::DenseI64ArrayAttr::get(mlir_context,
                                                     {1, 1, stride_height, stride_width}))
                        .getResult();
                pooled =
                    pooled
                        ? mlir::stablehlo::MaxOp::create(builder, loc, pooled, window).getResult()
                        : window;
            }
        }
        return pooled;
    }
}
