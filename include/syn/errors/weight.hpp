#pragma once

#include "mln/error.hpp"

namespace Syn::Compiler::Errors::Weight {
    struct Iree {
            struct Safetensors {
                    struct Header {
                            struct Parse {
                                    static inline constexpr Mln::Error Failed{
                                        Mln::Range::Weights + 100,
                                        "IREE failed to parse the safetensors "
                                        "header: {}"};
                            };
                    };
            };
            struct Archive {
                    struct File {
                            struct Open {
                                    static inline constexpr Mln::Error Failed{
                                        Mln::Range::Weights + 101,
                                        "IREE failed to open a parameter archive: "
                                        "{}"};
                            };
                    };
                    struct Index {
                            struct Parse {
                                    static inline constexpr Mln::Error Failed{
                                        Mln::Range::Weights + 105,
                                        "IREE failed to parse a parameter archive "
                                        "index: {}"};
                            };
                    };
            };
            struct Source {
                    struct Index {
                            struct Create {
                                    static inline constexpr Mln::Error Failed{
                                        Mln::Range::Weights + 102,
                                        "IREE failed to create a source parameter "
                                        "index: {}"};
                            };
                    };
            };
            struct Target {
                    struct Index {
                            struct Create {
                                    static inline constexpr Mln::Error Failed{
                                        Mln::Range::Weights + 103,
                                        "IREE failed to create a target parameter "
                                        "index: {}"};
                            };
                    };
            };
            struct Annotated {
                    struct Index {
                            struct Create {
                                    static inline constexpr Mln::Error Failed{
                                        Mln::Range::Weights + 104,
                                        "IREE failed to create an annotated "
                                        "parameter index: {}"};
                            };
                            struct Reserve {
                                    static inline constexpr Mln::Error Failed{
                                        Mln::Range::Weights + 108,
                                        "IREE failed to reserve the annotated "
                                        "parameter index: {}"};
                            };
                    };
                    struct Entry {
                            struct Add {
                                    static inline constexpr Mln::Error Failed{
                                        Mln::Range::Weights + 113,
                                        "IREE failed to add an annotated parameter "
                                        "entry: {}"};
                            };
                    };
            };
            struct Index {
                    struct Entry {
                            struct Read {
                                    static inline constexpr Mln::Error Failed{
                                        Mln::Range::Weights + 106,
                                        "IREE failed to read a parameter index "
                                        "entry: {}"};
                            };
                    };
            };
            struct Empty {
                    struct Archive {
                            struct Build {
                                    static inline constexpr Mln::Error Failed{
                                        Mln::Range::Weights + 107,
                                        "IREE failed to build an empty parameter "
                                        "archive: {}"};
                            };
                    };
            };
            struct Transformed {
                    struct Buffer {
                            struct Wrap {
                                    static inline constexpr Mln::Error Failed{
                                        Mln::Range::Weights + 109,
                                        "IREE failed to wrap the transformed "
                                        "weight buffer: {}"};
                            };
                    };
            };
            struct Converted {
                    struct Archive {
                            struct Build {
                                    static inline constexpr Mln::Error Failed{
                                        Mln::Range::Weights + 114,
                                        "IREE failed to build the converted "
                                        "parameter archive: {}"};
                            };
                    };
            };
    };
    struct File {
            struct Open {
                    static inline constexpr Mln::Error Failed{
                        Mln::Range::Weights + 110, "failed to open weight file '{}': {}"};
            };
            struct Size {
                    static inline constexpr Mln::Error Invalid{
                        Mln::Range::Weights + 111, "weight file '{}' has invalid size: {}"};
            };
            struct Mapping {
                    static inline constexpr Mln::Error Failed{Mln::Range::Weights + 112,
                                                              "failed to map weight file '{}': {}"};
            };
    };
    struct Safetensors {
            struct Header {
                    static inline constexpr Mln::Error Missing{
                        Mln::Range::Weights + 120, "safetensors file is smaller than its header"};
                    static inline constexpr Mln::Error Truncated{Mln::Range::Weights + 121,
                                                                 "safetensors header is truncated"};
            };
    };
    struct Checkpoint {
            struct Inspect {
                    static inline constexpr Mln::Error Failed{
                        Mln::Range::Weights + 130, "failed to inspect checkpoint '{}': {}"};
            };
            struct Type {
                    static inline constexpr Mln::Error Invalid{
                        Mln::Range::Weights + 131,
                        "checkpoint '{}' is neither a file nor a directory"};
            };
            struct Shards {
                    static inline constexpr Mln::Error Missing{
                        Mln::Range::Weights + 132,
                        "checkpoint '{}' contains no .safetensors shards"};
            };
    };
    struct Gguf {
            struct String {
                    static inline constexpr Mln::Error Truncated{Mln::Range::Weights + 140,
                                                                 "GGUF string is truncated"};
            };
            struct File {
                    static inline constexpr Mln::Error Truncated{Mln::Range::Weights + 141,
                                                                 "GGUF file is truncated"};
            };
            struct Metadata {
                    struct Array {
                            struct Too {
                                    static inline constexpr Mln::Error Large{
                                        Mln::Range::Weights + 142,
                                        "GGUF metadata array is too large"};
                            };
                    };
                    struct Type {
                            static inline constexpr Mln::Error Unknown{
                                Mln::Range::Weights + 143, "GGUF metadata type {} is unknown"};
                    };
            };
            struct Storage {
                    static inline constexpr Mln::Error Unsupported{
                        Mln::Range::Weights + 144, "GGUF storage type {} is unsupported"};
            };
            struct Magic {
                    static inline constexpr Mln::Error Invalid{Mln::Range::Weights + 145,
                                                               "GGUF magic is invalid"};
            };
            struct Version {
                    static inline constexpr Mln::Error Unsupported{
                        Mln::Range::Weights + 146, "GGUF version {} is unsupported"};
            };
            struct Entry {
                    struct Count {
                            static inline constexpr Mln::Error Invalid{
                                Mln::Range::Weights + 147, "GGUF entry count is invalid"};
                    };
            };
            struct Rank {
                    static inline constexpr Mln::Error Invalid{Mln::Range::Weights + 148,
                                                               "GGUF tensor rank is invalid"};
            };
            struct Shape {
                    static inline constexpr Mln::Error Invalid{
                        Mln::Range::Weights + 149, "GGUF tensor '{}' has an invalid shape"};
            };
            struct Block {
                    struct Alignment {
                            static inline constexpr Mln::Error Invalid{
                                Mln::Range::Weights + 150,
                                "GGUF tensor '{}' is not aligned to its storage "
                                "block"};
                    };
            };
            struct Tensor {
                    struct Too {
                            static inline constexpr Mln::Error Large{Mln::Range::Weights + 151,
                                                                     "GGUF tensor size overflows"};
                    };
            };
            struct Declared {
                    struct Storage {
                            static inline constexpr Mln::Error Missing{
                                Mln::Range::Weights + 152,
                                "GGUF declares quantization '{}', but contains no "
                                "'{}' storage"};
                    };
            };
    };
    struct Irpa {
            struct Metadata {
                    static inline constexpr Mln::Error Missing{
                        Mln::Range::Weights + 160, "IRPA entry '{}' has no MyeLiN metadata"};
                    static inline constexpr Mln::Error Mismatch{
                        Mln::Range::Weights + 161, "IRPA metadata does not match entry '{}'"};
            };
            struct Descriptor {
                    static inline constexpr Mln::Error Missing{
                        Mln::Range::Weights + 162, "IRPA weight '{}' has no descriptor"};
                    struct Size {
                            static inline constexpr Mln::Error Mismatch{
                                Mln::Range::Weights + 163, "IRPA weight '{}' size differs from its "
                                                           "descriptor"};
                    };
            };
            struct Required {
                    struct Weight {
                            static inline constexpr Mln::Error Missing{
                                Mln::Range::Weights + 164,
                                "IRPA plan references missing weight '{}'"};
                    };
            };
    };
    struct Descriptor {
            struct Shape {
                    static inline constexpr Mln::Error Invalid{
                        Mln::Range::Weights + 170, "weight descriptor '{}' has an invalid shape"};
            };
    };
    struct Metadata {
            struct Typed {
                    static inline constexpr Mln::Error Unsupported{
                        Mln::Range::Weights + 171,
                        "typed weight metadata is unsupported for '{}'; use "
                        "Safetensors, GGUF, or IRPA"};
            };
    };
    struct Duplicate {
            static inline constexpr Mln::Error Sources{
                Mln::Range::Weights + 180, "weight '{}' occurs in multiple source files"};
            static inline constexpr Mln::Error Catalog{
                Mln::Range::Weights + 181, "weight '{}' occurs more than once in the catalog"};
            static inline constexpr Mln::Error Shards{Mln::Range::Weights + 182,
                                                      "weight '{}' occurs in multiple shards"};
    };
    struct Storage {
            struct Outside {
                    static inline constexpr Mln::Error File{
                        Mln::Range::Weights + 190, "weight '{}' points outside its source file"};
            };
            struct Splat {
                    static inline constexpr Mln::Error Empty{
                        Mln::Range::Weights + 191, "weight '{}' has an empty splat pattern"};
            };
            struct Kind {
                    static inline constexpr Mln::Error Unknown{
                        Mln::Range::Weights + 192, "weight '{}' uses an unknown IREE storage kind"};
            };
    };
    struct Plan {
            struct Shape {
                    static inline constexpr Mln::Error Mismatch{
                        Mln::Range::Weights + 200, "plan shape differs for weight '{}'"};
            };
            struct Component {
                    struct Codec {
                            static inline constexpr Mln::Error Mismatch{
                                Mln::Range::Weights + 201,
                                "components of weight '{}' use different codecs"};
                    };
            };
            struct Scheme {
                    static inline constexpr Mln::Error Mismatch{
                        Mln::Range::Weights + 202,
                        "target scheme does not match the plan for weight '{}'"};
            };
    };
    struct Codec {
            struct Element {
                    struct Count {
                            static inline constexpr Mln::Error Mismatch{
                                Mln::Range::Weights + 210,
                                "codec produced the wrong element count for weight "
                                "'{}'"};
                    };
            };
            struct Component {
                    static inline constexpr Mln::Error Missing{
                        Mln::Range::Weights + 212, "codec did not produce required component '{}'"};
                    struct Count {
                            static inline constexpr Mln::Error Mismatch{
                                Mln::Range::Weights + 211,
                                "codec produced the wrong component count for "
                                "weight '{}'"};
                    };
            };
    };
}
