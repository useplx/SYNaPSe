#pragma once

#include "mln/sdk/functional/abs.hpp"
#include "mln/sdk/functional/bitwisenot.hpp"
#include "mln/sdk/functional/cbrt.hpp"
#include "mln/sdk/functional/ceil.hpp"
#include "mln/sdk/functional/cos.hpp"
#include "mln/sdk/functional/countleadingzeros.hpp"
#include "mln/sdk/functional/exp.hpp"
#include "mln/sdk/functional/expm1.hpp"
#include "mln/sdk/functional/floor.hpp"
#include "mln/sdk/functional/isfinite.hpp"
#include "mln/sdk/functional/log.hpp"
#include "mln/sdk/functional/log1p.hpp"
#include "mln/sdk/functional/logistic.hpp"
#include "mln/sdk/functional/negate.hpp"
#include "mln/sdk/functional/populationcount.hpp"
#include "mln/sdk/functional/round.hpp"
#include "mln/sdk/functional/roundnearesteven.hpp"
#include "mln/sdk/functional/rsqrt.hpp"
#include "mln/sdk/functional/sign.hpp"
#include "mln/sdk/functional/sin.hpp"
#include "mln/sdk/functional/softplus.hpp"
#include "mln/sdk/functional/sqrt.hpp"
#include "mln/sdk/functional/tan.hpp"
#include "mln/sdk/functional/tanh.hpp"
#include "syn/operation.hpp"
#include "syn/operations.hpp"

#include "stablehlo/dialect/StablehloOps.h"

namespace Syn::Compiler::Functional {
    template <typename Node, typename OpTy> class Unary : public Operation::For<Node> {
        public:
            mlir::Value emit_typed(const Node &node, Operations &ctx, mlir::OpBuilder &builder,
                                   mlir::Location loc) const override {
                mlir::Value x = ctx.resolve(node.x, builder, loc);
                return OpTy::create(builder, loc, x).getResult();
            }
    };

    class Logistic : public Unary<Mln::Sdk::Functional::Logistic, mlir::stablehlo::LogisticOp> {};
    class Exp : public Unary<Mln::Sdk::Functional::Exp, mlir::stablehlo::ExpOp> {};
    class Log : public Unary<Mln::Sdk::Functional::Log, mlir::stablehlo::LogOp> {};

    class Softplus : public Operation::For<Mln::Sdk::Functional::Softplus> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Softplus &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };

    class Sqrt : public Unary<Mln::Sdk::Functional::Sqrt, mlir::stablehlo::SqrtOp> {};
    class Rsqrt : public Unary<Mln::Sdk::Functional::Rsqrt, mlir::stablehlo::RsqrtOp> {};
    class Sin : public Unary<Mln::Sdk::Functional::Sin, mlir::stablehlo::SineOp> {};
    class Cos : public Unary<Mln::Sdk::Functional::Cos, mlir::stablehlo::CosineOp> {};
    class Tanh : public Unary<Mln::Sdk::Functional::Tanh, mlir::stablehlo::TanhOp> {};
    class Abs : public Unary<Mln::Sdk::Functional::Abs, mlir::stablehlo::AbsOp> {};
    class Negate : public Unary<Mln::Sdk::Functional::Negate, mlir::stablehlo::NegOp> {};
    class Floor : public Unary<Mln::Sdk::Functional::Floor, mlir::stablehlo::FloorOp> {};
    class Ceil : public Unary<Mln::Sdk::Functional::Ceil, mlir::stablehlo::CeilOp> {};
    class Cbrt : public Unary<Mln::Sdk::Functional::Cbrt, mlir::stablehlo::CbrtOp> {};
    class Expm1 : public Unary<Mln::Sdk::Functional::Expm1, mlir::stablehlo::Expm1Op> {};
    class Log1p : public Unary<Mln::Sdk::Functional::Log1p, mlir::stablehlo::Log1pOp> {};
    class Round : public Unary<Mln::Sdk::Functional::Round, mlir::stablehlo::RoundOp> {};
    class RoundNearestEven : public Unary<Mln::Sdk::Functional::RoundNearestEven,
                                          mlir::stablehlo::RoundNearestEvenOp> {};
    class Sign : public Unary<Mln::Sdk::Functional::Sign, mlir::stablehlo::SignOp> {};
    class Tan : public Unary<Mln::Sdk::Functional::Tan, mlir::stablehlo::TanOp> {};
    class CountLeadingZeros
        : public Unary<Mln::Sdk::Functional::CountLeadingZeros, mlir::stablehlo::ClzOp> {};
    class PopulationCount
        : public Unary<Mln::Sdk::Functional::PopulationCount, mlir::stablehlo::PopulationCountOp> {
    };
    class BitwiseNot : public Unary<Mln::Sdk::Functional::BitwiseNot, mlir::stablehlo::NotOp> {};
    class IsFinite : public Operation::For<Mln::Sdk::Functional::IsFinite> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::IsFinite &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override {
                mlir::Value input = ctx.resolve(node.x, builder, loc);
                return mlir::stablehlo::IsFiniteOp::create(builder, loc, input).getResult();
            }
    };
}
