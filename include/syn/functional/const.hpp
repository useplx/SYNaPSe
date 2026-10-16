#pragma once

#include "mln/sdk/functional/const.hpp"
#include "mln/sdk/weight.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class Const : public Operation::For<Mln::Sdk::Functional::Const> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Const &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
