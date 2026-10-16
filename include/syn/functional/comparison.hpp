#pragma once

#include "mln/sdk/functional/greater.hpp"
#include "mln/sdk/functional/greaterequal.hpp"
#include "mln/sdk/functional/less.hpp"
#include "mln/sdk/functional/lessequal.hpp"
#include "mln/sdk/functional/notequal.hpp"
#include "syn/operation.hpp"
#include "syn/operations.hpp"

#include "stablehlo/dialect/StablehloOps.h"

namespace Syn::Compiler::Functional {
    template <typename Node, mlir::stablehlo::ComparisonDirection Direction>
    class Compare : public Operation::For<Node> {
        public:
            mlir::Value emit_typed(const Node &node, Operations &ctx, mlir::OpBuilder &builder,
                                   mlir::Location loc) const override {
                mlir::Value lhs = ctx.resolve(node.lhs, builder, loc);
                mlir::Value rhs = ctx.resolve(node.rhs, builder, loc);
                return mlir::stablehlo::CompareOp::create(builder, loc, lhs, rhs, Direction)
                    .getResult();
            }
    };

    class NotEqual
        : public Compare<Mln::Sdk::Functional::NotEqual, mlir::stablehlo::ComparisonDirection::NE> {
    };
    class Greater
        : public Compare<Mln::Sdk::Functional::Greater, mlir::stablehlo::ComparisonDirection::GT> {
    };
    class GreaterEqual : public Compare<Mln::Sdk::Functional::GreaterEqual,
                                        mlir::stablehlo::ComparisonDirection::GE> {};
    class Less
        : public Compare<Mln::Sdk::Functional::Less, mlir::stablehlo::ComparisonDirection::LT> {};
    class LessEqual : public Compare<Mln::Sdk::Functional::LessEqual,
                                     mlir::stablehlo::ComparisonDirection::LE> {};
}
