#pragma once

#include "mln/error.hpp"

namespace Syn::Compiler::Errors::Package::Registry {
    namespace Library {
        namespace Load {
            inline constexpr Mln::Error Failed{Mln::Range::Compiler + 300,
                                               "failed to load package library '{}': {}"};
        }
        namespace Registration::Symbol {
            inline constexpr Mln::Error Missing{
                Mln::Range::Compiler + 301,
                "package library '{}' does not export mln_package_register: {}"};
        }
    }
    namespace Registration {
        inline constexpr Mln::Error Missing{Mln::Range::Compiler + 302,
                                            "package library '{}' returned a null registration"};
        namespace Abi {
            inline constexpr Mln::Error Mismatch{
                Mln::Range::Compiler + 303,
                "package library '{}' uses ABI {}, but MyeLiN requires ABI {}"};
        }
        inline constexpr Mln::Error Truncated{Mln::Range::Compiler + 304,
                                              "package registration is truncated in '{}'"};
        namespace Identity {
            inline constexpr Mln::Error Missing{Mln::Range::Compiler + 305,
                                                "package in '{}' has no name or version"};
        }
        namespace Exports {
            inline constexpr Mln::Error Missing{
                Mln::Range::Compiler + 306,
                "package registration in '{}' declares exports without an export array"};
        }
    }
    namespace Export {
        namespace Kind {
            inline constexpr Mln::Error Unknown{Mln::Range::Compiler + 310,
                                                "package export '{}' has unknown kind {}"};
        }
        inline constexpr Mln::Error Invalid{Mln::Range::Compiler + 311,
                                            "package export {} in '{}' is incomplete"};
        inline constexpr Mln::Error Duplicated{Mln::Range::Compiler + 312,
                                               "package export '{}' is duplicated"};
    }
}
