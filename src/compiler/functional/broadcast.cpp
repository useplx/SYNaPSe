#include "syn/functional/broadcast.hpp"
#include "mln/error.hpp"
#include "mln/sdk/functional/broadcast.hpp"
#include "syn/dynamic_shape.hpp"
#include "syn/errors/graph.hpp"
#include "syn/operations.hpp"

#include "mlir/IR/BuiltinTypes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <algorithm>
#include <stdexcept>

namespace Syn::Compiler::Functional {
    mlir::Value Broadcast::emit_typed(const Mln::Sdk::Functional::Broadcast &broadcast,
                                      Operations &ctx, mlir::OpBuilder &builder,
                                      mlir::Location loc) const {
        mlir::Value operand = ctx.resolve(broadcast.x, builder, loc);
        auto element_type = mlir::cast<mlir::ShapedType>(operand.getType()).getElementType();
        auto result_type = mlir::RankedTensorType::get(mlir_shape(broadcast.shape), element_type);
        if (result_type.hasStaticShape())
            return mlir::stablehlo::BroadcastInDimOp::create(
                       builder, loc, result_type, operand, llvm::ArrayRef<int64_t>(broadcast.dims))
                .getResult();

        mlir::Value source = operand;
        if (broadcast.shape_source.def) {
            source = ctx.resolve(broadcast.shape_source, builder, loc);
        }

        auto source_type = mlir::cast<mlir::RankedTensorType>(source.getType());
        std::vector<int64_t> source_dynamic_dimensions;
        for (int64_t i = 0; i < source_type.getRank(); ++i)
            if (source_type.isDynamicDim(i))
                source_dynamic_dimensions.push_back(i);

        std::vector<mlir::OpFoldResult> mixed_sizes;
        mixed_sizes.reserve(broadcast.shape.size());
        std::size_t dynamic_index = 0;
        for (int64_t dimension : broadcast.shape) {
            if (dimension == Mln::Sdk::Tensor::Dynamic) {
                if (dynamic_index >= source_dynamic_dimensions.size()) {
                    if (source_dynamic_dimensions.size() == 1) {
                        dynamic_index = 0;
                    } else {
                        raise(Errors::Graph::Broadcast::Shape::Source::Incompatible,
                              broadcast.policy_path,
                              std::ranges::count(broadcast.shape, Mln::Sdk::Tensor::Dynamic),
                              source_dynamic_dimensions.size());
                    }
                }
                mixed_sizes.push_back(
                    mlir::tensor::DimOp::create(builder, loc, source,
                                                source_dynamic_dimensions[dynamic_index++])
                        .getResult());
            } else {
                mixed_sizes.push_back(builder.getIndexAttr(dimension));
            }
        }

        mlir::Value empty =
            mlir::tensor::EmptyOp::create(builder, loc, mixed_sizes, element_type).getResult();
        std::vector<int64_t> added_dimensions;
        for (int64_t target = 0; target < static_cast<int64_t>(broadcast.shape.size()); ++target)
            if (std::find(broadcast.dims.begin(), broadcast.dims.end(), target) ==
                broadcast.dims.end())
                added_dimensions.push_back(target);
        return mlir::linalg::BroadcastOp::create(builder, loc, operand, empty, added_dimensions)
            .getResult()
            .front();
    }
}
