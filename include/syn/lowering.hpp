#pragma once

#include "mln/sdk/lowering/compatibility.hpp"

#include <span>
#include <string>
#include <string_view>

namespace Syn::Compiler::Lowering {
    [[nodiscard]] std::string
    apply(Mln::Ir::Module module, const Mln::Sdk::Backend &backend,
          std::span<Mln::Sdk::Lowering::Compatibility *const> compatibilities,
          std::string_view source = Mln::Ir::Graph::Format);
}
