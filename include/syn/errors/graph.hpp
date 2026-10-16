#pragma once

#include "mln/error.hpp"

namespace Syn::Compiler::Errors::Graph {
    namespace Operation::Emitter {
        inline constexpr Mln::Error Missing{
            Mln::Range::Compiler + 199,
            "no compiler emitter is registered for operation {} produced at '{}'"};
    }

    namespace Codec {
        namespace Compatibility {
            inline constexpr Mln::Error Missing{
                Mln::Range::Compiler + 200,
                "compiler manifest contains a null codec compatibility"};
        }
        namespace Compatibility::Output {
            inline constexpr Mln::Error Missing{
                Mln::Range::Compiler + 201,
                "codec compatibility returned an undefined tensor for weight '{}'"};
        }
        namespace Storage {
            inline constexpr Mln::Error Missing{
                Mln::Range::Compiler + 202,
                "codec '{}' did not declare storage while decoding '{}'"};
        }
        namespace Requirements {
            inline constexpr Mln::Error Conflict{Mln::Range::Compiler + 203,
                                                 "codec requirements conflict for weight '{}'"};
        }
    }

    namespace Weight {
        namespace Shape {
            inline constexpr Mln::Error Mismatch{Mln::Range::Compiler + 210,
                                                 "source shape does not match graph weight '{}'"};
        }
        namespace Policy {
            inline constexpr Mln::Error Missing{
                Mln::Range::Compiler + 211,
                "no Constant/Immutable policy covers graph weight '{}'"};
        }
        namespace Format {
            inline constexpr Mln::Error Unsupported{
                Mln::Range::Compiler + 212,
                "the Constant/Immutable policy format does not support weight '{}'"};
        }
        namespace Storage {
            inline constexpr Mln::Error Mismatch{
                Mln::Range::Compiler + 213,
                "weight '{}' is stored as '{}', but its policy selected '{}'"};
        }
    }

    namespace Map {
        namespace DType {
            inline constexpr Mln::Error Unknown{Mln::Range::Compiler + 220,
                                                "map body contains an unknown dtype"};
        }
        namespace Scalar::Conversion {
            inline constexpr Mln::Error Unsupported{
                Mln::Range::Compiler + 221, "map body requests an unsupported scalar conversion"};
        }
        namespace Scalar::Constant {
            inline constexpr Mln::Error Incomplete{
                Mln::Range::Compiler + 222,
                "map scalar constant requires an explicit value and dtype"};
        }
        namespace Integer::Division {
            inline constexpr Mln::Error Ambiguous{
                Mln::Range::Compiler + 223, "map integer division requires explicit semantics"};
        }
        namespace Operation {
            inline constexpr Mln::Error Unsupported{
                Mln::Range::Compiler + 224,
                "map body contains an operation that is not scalar elementwise"};
        }
        namespace Gather {
            namespace Indices::Rank {
                inline constexpr Mln::Error Invalid{Mln::Range::Compiler + 225,
                                                    "map gather pushdown requires rank-1 indices"};
            }
            inline constexpr Mln::Error Unsupported{Mln::Range::Compiler + 226,
                                                    "map gather configuration is unsupported"};
        }
        namespace Axis {
            inline constexpr Mln::Error Invalid{Mln::Range::Compiler + 227,
                                                "map input axis is outside the output rank"};
        }
    }

    namespace Iota {
        namespace Shape {
            inline constexpr Mln::Error Dynamic{Mln::Range::Compiler + 230,
                                                "iota requires a static shape"};
        }
        namespace Dimension {
            inline constexpr Mln::Error Invalid{Mln::Range::Compiler + 231,
                                                "iota dimension is outside the shape rank"};
        }
    }

    namespace Slice {
        namespace Limit::Source {
            inline constexpr Mln::Error Missing{Mln::Range::Compiler + 240,
                                                "dynamic slice requires a limit source"};
        }
    }

    namespace Broadcast {
        namespace Shape::Source {
            inline constexpr Mln::Error Incompatible{
                Mln::Range::Compiler + 250, "dynamic broadcast at '{}' requires {} dimensions, but "
                                            "its shape source provides {}"};
        }
    }

    namespace Shape::Dynamic::Source {
        inline constexpr Mln::Error Missing{Mln::Range::Compiler + 251,
                                            "dynamic dimension has no runtime shape source"};
    }

    namespace Deployment::Vulkan::Ordinal {
        inline constexpr Mln::Error Unsupported{
            Mln::Range::Compiler + 260,
            "Vulkan accepts ordinal 0 only; multi-GPU selection requires a runtime GPU-<UUID> "
            "device path"};
    }
    namespace Deployment::Backend {
        inline constexpr Mln::Error Missing{
            Mln::Range::Compiler + 261,
            "operation backend is absent from the deployment device set"};
    }
}
