#include "syn/functional/const.hpp"
#include "mln/sdk/functional/const.hpp"
#include "syn/operations.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

namespace Syn::Compiler::Functional {
    namespace {
        mlir::Value materialize(const Mln::Sdk::Weight &weight, Mln::DType dtype,
                                mlir::OpBuilder &builder, mlir::Location location) {
            mlir::Type element_type = dtype_to_mlir_type(dtype, *builder.getContext());
            auto type = mlir::RankedTensorType::get(weight.shape, element_type);

            if (weight.value) {
                mlir::Attribute value;
                if (mlir::isa<mlir::FloatType>(element_type)) {
                    value = builder.getFloatAttr(element_type, *weight.value);
                } else {
                    value = builder.getIntegerAttr(element_type,
                                                   static_cast<std::int64_t>(*weight.value));
                }
                return mlir::stablehlo::ConstantOp::create(
                           builder, location, mlir::DenseElementsAttr::get(type, value))
                    .getResult();
            }

            auto module =
                builder.getInsertionBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>();

            if (auto existing =
                    module.lookupSymbol<mlir::iree_compiler::IREE::Util::GlobalOp>(weight.name)) {
                return existing.createLoadOp(location, builder)->getResult(0);
            }

            mlir::iree_compiler::IREE::Util::GlobalOp global;
            {
                mlir::OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPointToStart(module.getBody());
                auto parameter = mlir::iree_compiler::IREE::Flow::NamedParameterAttr::get(
                    builder.getContext(), type, builder.getStringAttr(""),
                    builder.getStringAttr(weight.name), builder.getDictionaryAttr({}));
                global = mlir::iree_compiler::IREE::Util::GlobalOp::create(
                    builder, location, weight.name, false, type, parameter);
                global.setPrivate();
            }
            return global.createLoadOp(location, builder)->getResult(0);
        }
    }

    mlir::Value Const::emit_typed(const Mln::Sdk::Functional::Const &c, Operations &ctx,
                                  mlir::OpBuilder &builder, mlir::Location loc) const {

        if (c.quantize) {
            const Police police = ctx.police_for(c.weight.name);
            if (c.weight.value.has_value())
                return materialize(c.weight, *police.dtype, builder, loc);

            return ctx.resolve(ctx.decode_weight(c), builder, loc);
        }

        const Mln::DType dtype = c.dtype ? *c.dtype : *ctx.police_for(c.weight.name).dtype;
        return materialize(c.weight, dtype, builder, loc);
    }
}
