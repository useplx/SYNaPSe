#pragma once

#include "mln/sdk/functional/add.hpp"
#include "mln/sdk/functional/atan2.hpp"
#include "mln/sdk/functional/bitwiseand.hpp"
#include "mln/sdk/functional/bitwiseor.hpp"
#include "mln/sdk/functional/bitwisexor.hpp"
#include "mln/sdk/functional/divide.hpp"
#include "mln/sdk/functional/maximum.hpp"
#include "mln/sdk/functional/minimum.hpp"
#include "mln/sdk/functional/multiply.hpp"
#include "mln/sdk/functional/power.hpp"
#include "mln/sdk/functional/remainder.hpp"
#include "mln/sdk/functional/shiftleftlogical.hpp"
#include "mln/sdk/functional/shiftrightarithmetic.hpp"
#include "mln/sdk/functional/shiftrightlogical.hpp"
#include "mln/sdk/functional/subtract.hpp"
#include "syn/operation.hpp"
#include "syn/operations.hpp"

#include "stablehlo/dialect/StablehloOps.h"

#include <vector>

namespace Syn::Compiler::Functional {
    template <typename Node, typename OpTy> class Binary : public Operation::For<Node> {
        public:
            mlir::Value emit_typed(const Node &binary, Operations &ctx, mlir::OpBuilder &builder,
                                   mlir::Location loc) const override {
                mlir::Value lhs = ctx.resolve(binary.lhs, builder, loc);
                mlir::Value rhs = ctx.resolve(binary.rhs, builder, loc);
                return OpTy::create(builder, loc, std::vector<mlir::Value>{lhs, rhs}).getResult();
            }
    };

    class Add : public Binary<Mln::Sdk::Functional::Add, mlir::stablehlo::AddOp> {};
    class Subtract : public Binary<Mln::Sdk::Functional::Subtract, mlir::stablehlo::SubtractOp> {};
    class Multiply : public Binary<Mln::Sdk::Functional::Multiply, mlir::stablehlo::MulOp> {};
    class Divide : public Binary<Mln::Sdk::Functional::Divide, mlir::stablehlo::DivOp> {};
    class BitwiseAnd : public Binary<Mln::Sdk::Functional::BitwiseAnd, mlir::stablehlo::AndOp> {};
    class ShiftRightLogical : public Binary<Mln::Sdk::Functional::ShiftRightLogical,
                                            mlir::stablehlo::ShiftRightLogicalOp> {};
    class ShiftLeftLogical
        : public Binary<Mln::Sdk::Functional::ShiftLeftLogical, mlir::stablehlo::ShiftLeftOp> {};
    class Maximum : public Binary<Mln::Sdk::Functional::Maximum, mlir::stablehlo::MaxOp> {};
    class Minimum : public Binary<Mln::Sdk::Functional::Minimum, mlir::stablehlo::MinOp> {};
    class Power : public Binary<Mln::Sdk::Functional::Power, mlir::stablehlo::PowOp> {};
    class Remainder : public Binary<Mln::Sdk::Functional::Remainder, mlir::stablehlo::RemOp> {};
    class BitwiseOr : public Binary<Mln::Sdk::Functional::BitwiseOr, mlir::stablehlo::OrOp> {};
    class BitwiseXor : public Binary<Mln::Sdk::Functional::BitwiseXor, mlir::stablehlo::XorOp> {};
    class Atan2 : public Binary<Mln::Sdk::Functional::Atan2, mlir::stablehlo::Atan2Op> {};
    class ShiftRightArithmetic : public Binary<Mln::Sdk::Functional::ShiftRightArithmetic,
                                               mlir::stablehlo::ShiftRightArithmeticOp> {};
}
