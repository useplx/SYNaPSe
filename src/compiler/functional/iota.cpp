#include "syn/functional/iota.hpp"
#include "mln/error.hpp"
#include "mln/sdk/functional/iota.hpp"
#include "syn/dynamic_shape.hpp"
#include "syn/errors/graph.hpp"
#include "syn/operations.hpp"

#include "mlir/IR/BuiltinTypes.h"

#include "stablehlo/dialect/StablehloOps.h"

#include <stdexcept>

namespace Syn::Compiler::Functional {
    mlir::Value Iota::emit_typed(const Mln::Sdk::Functional::Iota &iota, Operations &ctx,
                                 mlir::OpBuilder &builder, mlir::Location loc) const {
        (void)ctx;

        for (int64_t dimension : iota.shape)
            if (dimension == Mln::Sdk::Tensor::Dynamic)
                raise(Errors::Graph::Iota::Shape::Dynamic);
        if (iota.dimension < 0 || iota.dimension >= static_cast<int64_t>(iota.shape.size()))
            raise(Errors::Graph::Iota::Dimension::Invalid);

        auto element_type = dtype_to_mlir_type(iota.dtype, *builder.getContext());
        auto result_type = mlir::RankedTensorType::get(mlir_shape(iota.shape), element_type);
        return mlir::stablehlo::IotaOp::create(builder, loc, result_type,
                                               static_cast<uint64_t>(iota.dimension))
            .getResult();
    }
}
