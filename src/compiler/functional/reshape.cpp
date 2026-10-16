#include "syn/functional/reshape.hpp"
#include "mln/sdk/functional/reshape.hpp"
#include "syn/dynamic_shape.hpp"
#include "syn/operations.hpp"

#include "mlir/IR/BuiltinTypes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace Syn::Compiler::Functional {
    mlir::Value Reshape::emit_typed(const Mln::Sdk::Functional::Reshape &reshape, Operations &ctx,
                                    mlir::OpBuilder &builder, mlir::Location loc) const {
        mlir::Value operand = ctx.resolve(reshape.x, builder, loc);
        auto element_type = mlir::cast<mlir::ShapedType>(operand.getType()).getElementType();
        auto result_type = mlir::RankedTensorType::get(mlir_shape(reshape.shape), element_type);
        if (result_type.hasStaticShape())
            return mlir::stablehlo::ReshapeOp::create(builder, loc, result_type, operand)
                .getResult();

        auto operand_type = mlir::cast<mlir::RankedTensorType>(operand.getType());

        const auto expand = [&](std::vector<mlir::ReassociationIndices> reassociation) {
            return mlir::tensor::ExpandShapeOp::create(builder, loc, result_type, operand,
                                                       reassociation)
                .getResult();
        };
        const auto collapse = [&](std::vector<mlir::ReassociationIndices> reassociation) {
            return mlir::tensor::CollapseShapeOp::create(builder, loc, result_type, operand,
                                                         reassociation)
                .getResult();
        };

        if (operand_type.getRank() == 2 && result_type.getRank() == 4 &&
            operand_type.isDynamicDim(0) && result_type.isDynamicDim(0))
            return expand({{0}, {1, 2, 3}});

        if (operand_type.getRank() == 2 && result_type.getRank() == 3 &&
            operand_type.isDynamicDim(0) && result_type.isDynamicDim(0))
            return expand({{0, 1}, {2}});

        if (operand_type.getRank() == 4 && result_type.getRank() == 3 &&
            operand_type.isDynamicDim(0) && result_type.isDynamicDim(0))
            return collapse({{0}, {1, 2}, {3}});

        if (operand_type.getRank() == 3 && result_type.getRank() == 2 &&
            operand_type.isDynamicDim(0) && result_type.isDynamicDim(0))
            return collapse({{0}, {1, 2}});

        if (operand_type.getRank() == result_type.getRank() + 1 &&
            operand_type.getDimSize(0) != mlir::ShapedType::kDynamic &&
            operand_type.getDimSize(1) != mlir::ShapedType::kDynamic &&
            result_type.getDimSize(0) == operand_type.getDimSize(0) * operand_type.getDimSize(1)) {
            std::vector<mlir::ReassociationIndices> reassociation;
            reassociation.push_back({0, 1});
            for (int64_t i = 2; i < operand_type.getRank(); ++i)
                reassociation.push_back({i});
            return mlir::tensor::CollapseShapeOp::create(builder, loc, result_type, operand,
                                                         reassociation)
                .getResult();
        }

        std::vector<int64_t> dynamic_dimensions;
        for (int64_t i = 0; i < operand_type.getRank(); ++i)
            if (operand_type.isDynamicDim(i))
                dynamic_dimensions.push_back(i);
        mlir::Value output_shape =
            runtime_shape(builder, loc, reshape.shape, operand, dynamic_dimensions);
        return mlir::stablehlo::DynamicReshapeOp::create(builder, loc, result_type, operand,
                                                         output_shape)
            .getResult();
    }
}
