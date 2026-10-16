#pragma once

#include "mln/error.hpp"

namespace Syn::Compiler::Errors::Compiler {
    namespace Iree::Flags::Set {
        inline constexpr Mln::Error Failed{Mln::Range::Compiler + 503,
                                           "IREE failed to set compiler flags: {}"};
    }
    namespace Iree::Source::Wrap {
        inline constexpr Mln::Error Failed{Mln::Range::Compiler + 504,
                                           "IREE failed to wrap generated MLIR: {}"};
    }
    namespace Iree::Output::Open {
        inline constexpr Mln::Error Failed{Mln::Range::Compiler + 505,
                                           "IREE failed to open a VM bytecode output: {}"};
    }
    namespace Iree::Output::Emit {
        inline constexpr Mln::Error Failed{Mln::Range::Compiler + 506,
                                           "IREE failed to emit VM bytecode: {}"};
    }
    namespace Iree::Output::Map {
        inline constexpr Mln::Error Failed{Mln::Range::Compiler + 507,
                                           "IREE failed to map VM bytecode output: {}"};
    }
    namespace Iree::Source {
        inline constexpr Mln::Error Rejected{
            Mln::Range::Compiler + 501,
            "IREE rejected the generated MLIR; diagnostics: {}; source: {}"};
    }
    namespace Iree::Pipeline {
        inline constexpr Mln::Error Failed{
            Mln::Range::Compiler + 502,
            "IREE compilation pipeline failed; diagnostics: {}; source: {}"};
    }
    namespace Hook::Output {
        inline constexpr Mln::Error Missing{
            Mln::Range::Compiler + 510,
            "hook '{}' returned an undefined tensor while handling '{}'"};
    }
    namespace Hook::Path {
        inline constexpr Mln::Error Unmatched{Mln::Range::Compiler + 511,
                                              "hook path '{}' matched no graph operation"};
    }
    namespace Entrypoint::Not {
        inline constexpr Mln::Error Compiled{Mln::Range::Compiler + 520,
                                             "entrypoint '{}' was not compiled by any module"};
    }
    namespace Entrypoint::Path {
        inline constexpr Mln::Error Invalid{Mln::Range::Compiler + 521,
                                            "deployment contains an invalid entrypoint path"};
    }
    namespace Entrypoint::Backend {
        inline constexpr Mln::Error Missing{Mln::Range::Compiler + 522,
                                            "entrypoint backend was not materialized"};
    }
    namespace Entrypoint::Specialization::Name {
        inline constexpr Mln::Error Missing{Mln::Range::Compiler + 523,
                                            "entrypoint specialization has an empty name"};
    }
    namespace Entrypoint::Specialization::Arity {
        inline constexpr Mln::Error Mismatch{
            Mln::Range::Compiler + 524,
            "entrypoint '{}' specialization declares {} inputs, but forward requires {}"};
    }
    namespace Entrypoint::Result {
        inline constexpr Mln::Error Missing{Mln::Range::Compiler + 525,
                                            "entrypoint must return at least one tensor"};
    }
    namespace Entrypoint::Module {
        inline constexpr Mln::Error Unmatched{Mln::Range::Compiler + 526,
                                              "no manifest entrypoint belongs to module '{}'"};
    }
    namespace Entrypoint::Export {
        inline constexpr Mln::Error Unmatched{
            Mln::Range::Compiler + 527,
            "no manifest entrypoint belongs to export '{}' in module '{}'"};
    }
    namespace Module {
        inline constexpr Mln::Error Invalid{Mln::Range::Compiler + 530,
                                            "generated MLIR module failed verification"};
    }
    namespace Abi::DType {
        inline constexpr Mln::Error Unsupported{Mln::Range::Compiler + 531,
                                                "dynamic dtype cannot cross the byte ABI"};
    }
    namespace Abi::Argument::DType {
        inline constexpr Mln::Error Unsupported{Mln::Range::Compiler + 532,
                                                "argument dtype cannot cross the byte ABI"};
    }
    namespace Abi::Argument::Dynamic::Dimensions {
        inline constexpr Mln::Error Unsupported{
            Mln::Range::Compiler + 533,
            "the byte ABI supports at most one dynamic dimension per argument"};
    }
    namespace Abi::Result::Dynamic::Shape {
        inline constexpr Mln::Error Unsupported{Mln::Range::Compiler + 534,
                                                "dynamic result cannot cross the byte ABI"};
    }
    namespace Abi::Result::DType {
        inline constexpr Mln::Error Unsupported{Mln::Range::Compiler + 535,
                                                "result dtype cannot cross the byte ABI"};
    }
    namespace Abi::Arity {
        inline constexpr Mln::Error Unsupported{Mln::Range::Compiler + 536,
                                                "input arity {} has no compile dispatch"};
    }
    namespace Package::Graph::Output {
        inline constexpr Mln::Error Missing{Mln::Range::Compiler + 537,
                                            "package graph module '{}' produced no output"};
    }
    namespace Backend {
        inline constexpr Mln::Error Missing{Mln::Range::Compiler + 540,
                                            "manifest does not declare a backend"};
    }
    namespace Backend::Compatibility {
        inline constexpr Mln::Error Duplicated{
            Mln::Range::Compiler + 541,
            "backend compatibility '{}' was registered more than once with different objects"};
    }
    namespace Lowering::Chain {
        inline constexpr Mln::Error Ambiguous{
            Mln::Range::Compiler + 542,
            "multiple lowering compatibilities with priority {} accept '{}' for backend '{}'"};
        inline constexpr Mln::Error Cyclic{
            Mln::Range::Compiler + 543,
            "lowering compatibility chain revisited representation '{}' for backend '{}'"};
    }
    namespace Lowering {
        inline constexpr Mln::Error Failed{
            Mln::Range::Compiler + 544,
            "lowering compatibility failed to transform '{}' into '{}' for backend '{}'"};
    }
    namespace Mlir::Dump::Write {
        inline constexpr Mln::Error Failed{Mln::Range::Compiler + 550,
                                           "failed to write generated MLIR to '{}'"};
    }
    namespace Devices::File::Create {
        inline constexpr Mln::Error Failed{Mln::Range::Compiler + 560,
                                           "failed to create device manifest '{}'"};
    }
    namespace Weight::Scheme {
        inline constexpr Mln::Error Missing{Mln::Range::Compiler + 570,
                                            "no Constant/Immutable scheme covers weight '{}'"};
    }
    namespace Graph::Loop {
        namespace Iterations {
            inline constexpr Mln::Error Invalid{
                Mln::Range::Compiler + 580,
                "Loop iteration count must be between 1 and {} (received {})"};
        }
        namespace State {
            inline constexpr Mln::Error Mismatch{
                Mln::Range::Compiler + 581, "Loop body must preserve the state shape and dtype"};
        }
    }
}
