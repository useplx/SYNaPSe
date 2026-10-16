#pragma once

#include "mln/sdk/functional/select.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class Select : public Operation::For<Mln::Sdk::Functional::Select> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Select &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
