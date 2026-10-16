#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mln/error.hpp"
#include "stablehlo/dialect/StablehloOps.h"
#include "syn/errors/graph.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace Syn::Compiler {
    inline std::vector<int64_t> mlir_shape(const std::vector<int64_t> &shape) {
        std::vector<int64_t> result = shape;
        for (int64_t &dimension : result)
            if (dimension == -1)
                dimension = mlir::ShapedType::kDynamic;
        return result;
    }

    inline mlir::Value shape_scalar(mlir::OpBuilder &builder, mlir::Location location,
                                    int64_t value) {
        auto type = mlir::RankedTensorType::get({}, builder.getI32Type());
        return mlir::stablehlo::ConstantOp::create(
                   builder, location,
                   mlir::DenseElementsAttr::get(type, builder.getI32IntegerAttr(value)))
            .getResult();
    }

    inline mlir::Value dimension_size(mlir::OpBuilder &builder, mlir::Location location,
                                      mlir::Value source, int64_t dimension) {
        return mlir::stablehlo::GetDimensionSizeOp::create(builder, location, source, dimension)
            .getResult();
    }

    inline mlir::Value runtime_shape(mlir::OpBuilder &builder, mlir::Location location,
                                     const std::vector<int64_t> &shape, mlir::Value dynamic_source,
                                     const std::vector<int64_t> &source_dimensions,
                                     const std::vector<int64_t> &divisors = {}) {
        std::vector<mlir::Value> dimensions;
        dimensions.reserve(shape.size());
        std::size_t dynamic_index = 0;

        for (int64_t value : shape) {
            mlir::Value dimension;
            if (value == -1 || value == mlir::ShapedType::kDynamic) {
                if (!dynamic_source || dynamic_index >= source_dimensions.size())
                    raise(Errors::Graph::Shape::Dynamic::Source::Missing);
                dimension = dimension_size(builder, location, dynamic_source,
                                           source_dimensions[dynamic_index++]);
                if (!divisors.empty()) {
                    const int64_t divisor = divisors.at(dynamic_index - 1);
                    if (divisor != 1) {
                        dimension =
                            mlir::stablehlo::DivOp::create(builder, location, dimension,
                                                           shape_scalar(builder, location, divisor))
                                .getResult();
                    }
                }
            } else {
                dimension = shape_scalar(builder, location, value);
            }
            dimensions.push_back(mlir::stablehlo::ReshapeOp::create(
                                     builder, location,
                                     mlir::RankedTensorType::get({1}, builder.getI32Type()),
                                     dimension)
                                     .getResult());
        }

        return mlir::stablehlo::ConcatenateOp::create(
                   builder, location,
                   mlir::RankedTensorType::get({static_cast<int64_t>(shape.size())},
                                               builder.getI32Type()),
                   dimensions, 0)
            .getResult();
    }
}
