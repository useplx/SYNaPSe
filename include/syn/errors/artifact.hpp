#pragma once

#include "mln/error.hpp"

namespace Syn::Compiler::Errors::Artifact {
    namespace Name {
        inline constexpr Mln::Error Empty{Mln::Range::Compiler + 0, "artifact {} has no name"};
        inline constexpr Mln::Error Invalid{Mln::Range::Compiler + 1,
                                            "artifact {} has an invalid name: '{}'"};
    }

    namespace File {
        inline constexpr Mln::Error Missing{Mln::Range::Compiler + 10,
                                            "artifact {} '{}' source file does not exist: {}"};
        namespace Path {
            inline constexpr Mln::Error Missing{Mln::Range::Compiler + 11,
                                                "artifact {} '{}' has no source file"};
        }
    }

    namespace Entrypoint {
        inline constexpr Mln::Error Missing{Mln::Range::Compiler + 20,
                                            "artifact requires at least one executable entrypoint"};
        inline constexpr Mln::Error Duplicated{Mln::Range::Compiler + 21,
                                               "artifact entrypoint '{}' is duplicated"};
        namespace Module {
            inline constexpr Mln::Error Missing{
                Mln::Range::Compiler + 22,
                "artifact entrypoint '{}' references missing Wasm module '{}'"};
        }
    }

    namespace Module {
        inline constexpr Mln::Error Duplicated{Mln::Range::Compiler + 30,
                                               "artifact module '{}' is duplicated"};
    }

    namespace Hook {
        namespace Path {
            inline constexpr Mln::Error Empty{Mln::Range::Compiler + 32,
                                              "artifact hook has an empty target path"};
        }
        namespace Module {
            inline constexpr Mln::Error Missing{
                Mln::Range::Compiler + 33,
                "artifact hook for '{}' references missing Wasm module '{}'"};
        }
    }

    namespace Asset {
        inline constexpr Mln::Error Duplicated{Mln::Range::Compiler + 31,
                                               "artifact asset '{}' is duplicated"};
    }
}
