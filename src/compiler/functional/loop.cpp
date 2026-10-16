#include "syn/functional/loop.hpp"

#include "syn/errors/compiler.hpp"
#include "syn/operations.hpp"

#include "stablehlo/dialect/StablehloOps.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

#include <limits>

namespace Syn::Compiler::Functional {
    namespace {
        mlir::Value scalar_i32(mlir::OpBuilder &builder, mlir::Location loc, int32_t value) {
            const auto type = mlir::RankedTensorType::get({}, builder.getI32Type());
            const auto attribute = mlir::DenseIntElementsAttr::get(type, value);
            return mlir::stablehlo::ConstantOp::create(builder, loc, attribute).getResult();
        }
    }

    mlir::Value Loop::emit_typed(const Mln::Sdk::Functional::Loop &loop, Operations &ctx,
                                 mlir::OpBuilder &builder, mlir::Location loc) const {
        constexpr int64_t max_iterations = std::numeric_limits<int32_t>::max();
        if (loop.iterations <= 0 || loop.iterations > max_iterations)
            Mln::raise(Errors::Compiler::Graph::Loop::Iterations::Invalid, max_iterations,
                       loop.iterations);
        if (loop.result.shape != loop.initial.shape || loop.result.dtype != loop.initial.dtype)
            Mln::raise(Errors::Compiler::Graph::Loop::State::Mismatch);

        mlir::Value initial = ctx.resolve(loop.initial, builder, loc);
        mlir::Value index = scalar_i32(builder, loc, 0);
        if (loop.iterations == 1) {
            ctx.seed(loop.index_argument.def.get(), index);
            ctx.seed(loop.value_argument.def.get(), initial);
            return ctx.resolve(loop.result, builder, loc);
        }

        const std::array<mlir::Type, 2> result_types{index.getType(), initial.getType()};
        const std::array<mlir::Value, 2> operands{index, initial};
        auto operation = mlir::stablehlo::WhileOp::create(builder, loc, result_types, operands);

        mlir::OpBuilder::InsertionGuard guard(builder);
        mlir::Block &condition = operation.getCond().emplaceBlock();
        condition.addArguments(result_types, std::array<mlir::Location, 2>{loc, loc});
        builder.setInsertionPointToStart(&condition);
        mlir::Value limit = scalar_i32(builder, loc, static_cast<int32_t>(loop.iterations));
        mlir::Value predicate =
            mlir::stablehlo::CompareOp::create(builder, loc, condition.getArgument(0), limit,
                                               mlir::stablehlo::ComparisonDirection::LT,
                                               mlir::stablehlo::ComparisonType::SIGNED)
                .getResult();
        mlir::stablehlo::ReturnOp::create(builder, loc, predicate);

        mlir::Block &body = operation.getBody().emplaceBlock();
        body.addArguments(result_types, std::array<mlir::Location, 2>{loc, loc});
        builder.setInsertionPointToStart(&body);
        ctx.seed(loop.index_argument.def.get(), body.getArgument(0));
        ctx.seed(loop.value_argument.def.get(), body.getArgument(1));
        mlir::Value next = ctx.resolve(loop.result, builder, loc);
        mlir::Value one = scalar_i32(builder, loc, 1);
        mlir::Value next_index =
            mlir::stablehlo::AddOp::create(builder, loc, body.getArgument(0), one).getResult();
        mlir::stablehlo::ReturnOp::create(builder, loc, mlir::ValueRange{next_index, next});

        return operation.getResult(1);
    }
}
