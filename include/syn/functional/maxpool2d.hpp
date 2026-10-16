#pragma once

#include "mln/sdk/functional/maxpool2d.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {

    class MaxPool2D : public Operation::For<Mln::Sdk::Functional::MaxPool2D> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::MaxPool2D &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
