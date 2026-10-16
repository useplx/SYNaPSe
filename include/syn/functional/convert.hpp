#pragma once

#include "mln/sdk/functional/convert.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {

    class Convert : public Operation::For<Mln::Sdk::Functional::Convert> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Convert &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
