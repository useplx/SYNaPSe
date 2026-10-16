#pragma once

#include "mln/error.hpp"

namespace Syn::Compiler::Errors::Api {
    namespace Field {
        namespace Value {
            inline constexpr Mln::Error Invalid{Mln::Range::Compiler + 600,
                                                "field '{}' has invalid value '{}'"};
        }
    }
    namespace Export {
        namespace Reference {
            inline constexpr Mln::Error Invalid{Mln::Range::Compiler + 601,
                                                "export reference '{}' must use package:name"};
        }
        namespace Invocation {
            inline constexpr Mln::Error Failed{Mln::Range::Compiler + 602,
                                               "package export '{}' failed: {}"};
        }
    }
    namespace Codec {
        inline constexpr Mln::Error Unknown{Mln::Range::Compiler + 610,
                                            "codec export '{}' was not resolved"};
        namespace Format {
            inline constexpr Mln::Error Missing{Mln::Range::Compiler + 611,
                                                "codec export '{}' returned no format"};
        }
        namespace Scheme::Field {
            namespace Owned::By {
                inline constexpr Mln::Error Codec{
                    Mln::Range::Compiler + 612,
                    "scheme field '{}' belongs to the codec implementation"};
            }
        }
        namespace Compatibility {
            inline constexpr Mln::Error Missing{
                Mln::Range::Compiler + 613,
                "codec compatibility export '{}' returned no compatibility"};
        }
    }
    namespace Operation::Compatibility {
        inline constexpr Mln::Error Missing{
            Mln::Range::Compiler + 614,
            "operation compatibility export '{}' returned no compatibility"};
    }
    namespace Lowering::Compatibility {
        inline constexpr Mln::Error Missing{
            Mln::Range::Compiler + 615,
            "lowering compatibility export '{}' returned no compatibility"};
    }
    namespace Backend::Compatibility {
        inline constexpr Mln::Error Missing{
            Mln::Range::Compiler + 616,
            "backend compatibility export '{}' returned no compatibility"};
    }
    namespace Policy::Volatilized {
        inline constexpr Mln::Error Removed{
            Mln::Range::Compiler + 620,
            "policy '{}' uses removed field 'volatilized'; use scheme volatile for host weight "
            "residency or optimization_barrier for graph optimization"};
    }
    namespace Hook {
        inline constexpr Mln::Error Unknown{Mln::Range::Compiler + 630,
                                            "hook export '{}' was not resolved"};
    }
    namespace Entrypoint::Wasm::Module {
        inline constexpr Mln::Error Missing{Mln::Range::Compiler + 640,
                                            "entrypoint '{}' must select a Wasm module"};
    }
    namespace Plan {
        namespace Version {
            inline constexpr Mln::Error Unsupported{Mln::Range::Compiler + 650,
                                                    "plan version is unsupported"};
        }
        namespace Mode {
            inline constexpr Mln::Error Unsupported{Mln::Range::Compiler + 651,
                                                    "plan mode '{}' is unsupported"};
        }
    }
    namespace Graph {
        inline constexpr Mln::Error Unknown{Mln::Range::Compiler + 660,
                                            "graph export '{}' was not resolved"};
        namespace Namespace {
            inline constexpr Mln::Error Missing{Mln::Range::Compiler + 661,
                                                "graph export '{}' declared no namespace"};
        }
        namespace Module {
            inline constexpr Mln::Error Missing{Mln::Range::Compiler + 662,
                                                "graph export '{}' returned no module"};
        }
    }
    namespace Weight {
        namespace Duplicated::Across {
            inline constexpr Mln::Error Roots{Mln::Range::Compiler + 670,
                                              "weight '{}' occurs in more than one source root"};
        }
    }
    namespace Plugin::Exception {
        inline constexpr Mln::Error Escaped{Mln::Range::Compiler + 671,
                                            "package plugin raised an unexpected exception: {}"};
    }
}
