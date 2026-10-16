#pragma once

#include "mln/error.hpp"

namespace Syn::Compiler::Errors::Manifest {
    namespace Format {
        namespace Name {
            inline constexpr Mln::Error Missing{Mln::Range::Compiler + 400,
                                                "format registered by '{}' has no name"};
        }
        namespace Name {
            inline constexpr Mln::Error Conflict{
                Mln::Range::Compiler + 401, "format name '{}' is used by different format classes"};
        }
        inline constexpr Mln::Error Uninitialized{Mln::Range::Compiler + 402,
                                                  "format registry contains an empty format"};
        namespace Immutable {
            inline constexpr Mln::Error Unsupported{
                Mln::Range::Compiler + 403, "format '{}' does not support immutable constants"};
        }
        namespace Constant {
            inline constexpr Mln::Error Missing{Mln::Range::Compiler + 404,
                                                "no constant format named '{}' is registered"};
        }
    }
    namespace Scheme {
        namespace Format {
            inline constexpr Mln::Error Missing{Mln::Range::Compiler + 410,
                                                "scheme at '{}' has no format"};
        }
        namespace Lifetime {
            inline constexpr Mln::Error Unsupported{
                Mln::Range::Compiler + 411,
                "persisted codecs only support Immutable lifetime at '{}'"};
        }
        namespace Immutable::Update {
            inline constexpr Mln::Error Invalid{Mln::Range::Compiler + 412,
                                                "Immutable lifetime requires Update::None at '{}'"};
        }
        inline constexpr Mln::Error Duplicated{Mln::Range::Compiler + 413,
                                               "duplicate (Scope, Lifetime) scheme at '{}'"};
        namespace Wildcard {
            inline constexpr Mln::Error Ambiguous{Mln::Range::Compiler + 414,
                                                  "ambiguous wildcard schemes at '{}'"};
        }
        namespace Root {
            namespace Constant {
                inline constexpr Mln::Error Missing{
                    Mln::Range::Compiler + 415,
                    "root policy must cover Constant/Immutable weights"};
            }
            namespace Constant::Format {
                inline constexpr Mln::Error Invalid{
                    Mln::Range::Compiler + 416,
                    "root Constant/Immutable scheme requires a constant format"};
            }
        }
    }
    namespace Policy {
        namespace Root {
            inline constexpr Mln::Error Missing{Mln::Range::Compiler + 420,
                                                "manifest requires a root policy"};
            namespace DType {
                inline constexpr Mln::Error Missing{Mln::Range::Compiler + 421,
                                                    "root policy must define a dtype"};
            }
        }
        namespace Path {
            inline constexpr Mln::Error Empty{Mln::Range::Compiler + 422,
                                              "policy path cannot be empty"};
        }
    }
    namespace Entrypoint {
        namespace Name {
            inline constexpr Mln::Error Missing{Mln::Range::Compiler + 430,
                                                "manifest entrypoint has no name"};
        }
        inline constexpr Mln::Error Duplicated{Mln::Range::Compiler + 431,
                                               "manifest entrypoint '{}' is duplicated"};
        namespace Specialization {
            inline constexpr Mln::Error Missing{Mln::Range::Compiler + 432,
                                                "entrypoint '{}' has no specializations"};
            namespace Inputs {
                inline constexpr Mln::Error Missing{
                    Mln::Range::Compiler + 433,
                    "entrypoint '{}' has a specialization without inputs"};
            }
            namespace Dimension {
                inline constexpr Mln::Error Invalid{
                    Mln::Range::Compiler + 434, "entrypoint '{}' has an invalid input dimension"};
            }
        }
    }
    namespace Hook {
        namespace Path {
            inline constexpr Mln::Error Missing{Mln::Range::Compiler + 440,
                                                "graph hook has no path"};
            inline constexpr Mln::Error Invalid{Mln::Range::Compiler + 441,
                                                "graph hook path '{}' is invalid"};
            namespace Wildcard {
                inline constexpr Mln::Error Invalid{
                    Mln::Range::Compiler + 442,
                    "graph hook path '{}' uses '*' outside a complete segment"};
            }
        }
        namespace Forward {
            inline constexpr Mln::Error Missing{Mln::Range::Compiler + 443,
                                                "graph hook at '{}' has no forward function"};
        }
    }
    namespace Backend {
        namespace Device {
            inline constexpr Mln::Error Missing{Mln::Range::Compiler + 450,
                                                "backend device is empty at '{}'"};
        }
        namespace Ordinal {
            inline constexpr Mln::Error Negative{Mln::Range::Compiler + 451,
                                                 "backend ordinal is negative at '{}'"};
        }
    }
}
