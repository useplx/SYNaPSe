#pragma once

#include "mln/sdk/functional/bitcastconvert.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class BitcastConvert : public Operation::For<Mln::Sdk::Functional::BitcastConvert> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::BitcastConvert &node,
                                   Operations &ctx, mlir::OpBuilder &builder,
                                   mlir::Location loc) const override;
    };
}
