#include "syn/functional/concatenate.hpp"
#include "mln/sdk/functional/concatenate.hpp"
#include "syn/operations.hpp"

#include "mlir/IR/BuiltinTypes.h"

#include "stablehlo/dialect/StablehloOps.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace Syn::Compiler::Functional {
    mlir::Value Concatenate::emit_typed(const Mln::Sdk::Functional::Concatenate &concatenate,
                                        Operations &ctx, mlir::OpBuilder &builder,
                                        mlir::Location loc) const {
        const int64_t dim = concatenate.dim;

        std::vector<const Mln::Sdk::Tensor *> leaves;
        std::function<void(const Mln::Sdk::Tensor &)> collect =
            [&](const Mln::Sdk::Tensor &tensor) {
                if (tensor.def && tensor.def->kind == Mln::Sdk::Functional::Kind::Concatenate) {
                    const auto &child =
                        static_cast<const Mln::Sdk::Functional::Concatenate &>(*tensor.def);
                    if (child.dim == dim) {
                        collect(child.a);
                        collect(child.b);
                        return;
                    }
                }
                leaves.push_back(&tensor);
            };
        collect(concatenate.a);
        collect(concatenate.b);

        std::function<mlir::Value(std::size_t, std::size_t)> emit_balanced =
            [&](std::size_t begin, std::size_t end) -> mlir::Value {
            if (end - begin == 1)
                return ctx.resolve(*leaves[begin], builder, loc);
            const std::size_t middle = begin + (end - begin) / 2;
            mlir::Value a = emit_balanced(begin, middle);
            mlir::Value b = emit_balanced(middle, end);
            auto a_type = mlir::cast<mlir::RankedTensorType>(a.getType());
            auto b_type = mlir::cast<mlir::RankedTensorType>(b.getType());
            std::vector<int64_t> result_shape(a_type.getShape().begin(), a_type.getShape().end());
            result_shape[dim] += b_type.getShape()[dim];
            auto result_type = mlir::RankedTensorType::get(result_shape, a_type.getElementType());
            return mlir::stablehlo::ConcatenateOp::create(builder, loc, result_type,
                                                          mlir::ValueRange{a, b},
                                                          static_cast<uint64_t>(dim))
                .getResult();
        };
        return emit_balanced(0, leaves.size());
    }
}
