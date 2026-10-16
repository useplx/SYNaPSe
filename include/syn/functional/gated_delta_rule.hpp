#pragma once

#include "mln/sdk/functional/gated_delta_rule.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class GatedDeltaRule : public Operation::For<Mln::Sdk::Functional::GatedDeltaRule> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::GatedDeltaRule &node,
                                   Operations &ctx, mlir::OpBuilder &builder,
                                   mlir::Location loc) const override;
    };
}
