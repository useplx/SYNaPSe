#pragma once

#include "mln/error.hpp"

namespace Syn::Compiler::Errors::Json {
    namespace Integer {
        inline constexpr Mln::Error Expected{Mln::Range::Compiler + 100, "JSON: expected integer"};
    }
    namespace Field {
        inline constexpr Mln::Error Missing{Mln::Range::Compiler + 101, "JSON: missing field '{}'"};
    }
    inline constexpr Mln::Error Malformed{Mln::Range::Compiler + 102, "JSON: {} at byte {}"};
}
