#include "syn/functional/conv2d.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mln/sdk/functional/conv2d.hpp"
#include "stablehlo/dialect/StablehloOps.h"
#include "syn/operations.hpp"

namespace Syn::Compiler::Functional {
    mlir::Value Conv2D::emit_typed(const Mln::Sdk::Functional::Conv2D &conv2d, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const {
        mlir::Value x = ctx.resolve(conv2d.x, builder, loc);
        mlir::Value kw = ctx.resolve(conv2d.weight, builder, loc);
        mlir::Value b = ctx.resolve(conv2d.bias, builder, loc);

        const auto &stride = conv2d.stride;
        const auto &padding = conv2d.padding;
        const auto &dilation = conv2d.dilation;
        int64_t groups = conv2d.groups;

        auto x_type = mlir::cast<mlir::RankedTensorType>(x.getType());
        auto weight_type = mlir::cast<mlir::RankedTensorType>(kw.getType());
        auto elem = x_type.getElementType();
        auto mlir_context = builder.getContext();

        auto xs = x_type.getShape();
        auto weight_shape = weight_type.getShape();

        int64_t batch = xs[0];
        int64_t input_height = xs[2];
        int64_t input_width = xs[3];
        int64_t output_channels = weight_shape[0];
        int64_t kernel_height = weight_shape[2];
        int64_t kernel_width = weight_shape[3];
        int64_t stride_height = stride[0], stride_width = stride[1];
        int64_t dilation_height = dilation[0];
        int64_t dilation_width = dilation[1];
        int64_t padding_top = padding[0], padding_bottom = padding[1], padding_left = padding[2],
                padding_right = padding[3];

        int64_t output_height = (input_height + padding_top + padding_bottom -
                                 dilation_height * (kernel_height - 1) - 1) /
                                    stride_height +
                                1;
        int64_t output_width =
            (input_width + padding_left + padding_right - dilation_width * (kernel_width - 1) - 1) /
                stride_width +
            1;
        auto result_type = mlir::RankedTensorType::get(
            {batch, output_channels, output_height, output_width}, elem);

        auto dimension_numbers = mlir::stablehlo::ConvDimensionNumbersAttr::get(
            mlir_context, 0, 1, {2, 3}, 1, 0, {2, 3}, 0, 1, {2, 3});

        auto window_strides =
            mlir::DenseI64ArrayAttr::get(mlir_context, {stride_height, stride_width});
        auto rhs_dilation =
            mlir::DenseI64ArrayAttr::get(mlir_context, {dilation_height, dilation_width});

        auto padding_attribute = mlir::DenseIntElementsAttr::get(
            mlir::RankedTensorType::get({2, 2}, builder.getI64Type()),
            llvm::ArrayRef<int64_t>{padding_top, padding_bottom, padding_left, padding_right});

        auto conv = mlir::stablehlo::ConvolutionOp::create(
                        builder, loc, result_type, x, kw, window_strides, padding_attribute,
                        mlir::DenseI64ArrayAttr{}, rhs_dilation, mlir::DenseBoolArrayAttr{},
                        dimension_numbers, static_cast<uint64_t>(groups), 1UL, nullptr)
                        .getResult();

        auto b_type = mlir::cast<mlir::RankedTensorType>(b.getType());
        if (b_type.getNumElements() > 0) {
            auto broadcast_bias = mlir::stablehlo::BroadcastInDimOp::create(
                                      builder, loc, result_type, b, llvm::ArrayRef<int64_t>{1})
                                      .getResult();
            return mlir::stablehlo::AddOp::create(builder, loc,
                                                  llvm::ArrayRef<mlir::Value>{conv, broadcast_bias})
                .getResult();
        }
        return conv;
    }
}
