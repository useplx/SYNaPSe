#pragma once

#include "mln/weight.hpp"
#include "syn/compatibility.hpp"

namespace Syn::Compiler {
    namespace Format = Mln::Weight;
}

#include "mln/weight.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <mlir/IR/BuiltinOps.h>

namespace Syn::Compiler {
    struct Metric {
            enum class Quality {
                Exact,
                Estimated,
                Unknown,
            };

            struct Ratio {
                    std::optional<double> value;
                    Quality quality = Quality::Unknown;
            };

            std::optional<uint64_t> value;
            Quality quality = Quality::Unknown;

            static Metric exact(uint64_t value) { return Metric{value, Quality::Exact}; }

            static Metric estimated(uint64_t value) { return Metric{value, Quality::Estimated}; }

            static Metric unknown() { return {}; }
    };

    namespace Value {
        struct Lifetime {
                std::string function;
                std::string producer;
                std::size_t producer_index = 0;
                std::size_t first_use = 0;
                std::size_t last_use = 0;
                std::optional<uint64_t> bytes;
                bool persistent = false;
                bool external = false;
                bool view_candidate = false;
        };
    }

    namespace Function::Memory {
        struct Estimate {
                std::string function;
                std::string backend_affinity;
                Metric temporary_value_bytes;
                Metric estimated_peak_live_bytes;
        };
    }

    namespace Compilation {
        struct Legalization {
                std::string backend;
                std::string path;
                int operation = 0;
                std::string outcome;
        };

        struct Report {
                struct Storage {
                        Metric parameter_bytes;
                        Metric executable_bytes;
                        Metric metadata_bytes;
                        Metric total_artifact_bytes;
                } storage;

                struct Memory {
                        Metric persistent_bytes;
                        Metric temporary_value_bytes;
                        Metric estimated_peak_live_bytes;
                } memory;

                struct Graph {
                        Metric modules;
                        Metric operations_before_iree;
                        Metric operations_after_iree;
                        Metric constants;
                        Metric parameters;
                } graph;

                struct Quantization {
                        Metric bytes_before;
                        Metric bytes_after;
                        Metric::Ratio compression_ratio;
                } quantization;

                struct Packaging {
                        Metric source_bytes;
                        Metric output_bytes;
                        Metric source_parameters;
                        Metric output_parameters;
                        Metric passthrough_parameters;
                        Metric converted_parameters;
                } packaging;

                struct Optimization {
                        Metric eliminated_values;
                        Metric eliminated_operations;
                        Metric eliminated_parameters;
                        Metric eliminated_parameter_bytes;
                        Metric view_candidates;
                        Metric inplace_candidates;
                        Metric reused_storage_candidates;
                } optimization;

                struct Partitioning {
                        Metric partitions;
                        Metric estimated_transfers;
                        std::vector<std::string> target_backends;
                } partitioning;

                std::vector<Value::Lifetime> lifetimes;
                std::vector<Function::Memory::Estimate> functions;
                std::vector<Legalization> legalizations;
                std::vector<Compatibility::Entry> compatibilities;

                std::string text() const;
        };
    }

    Compilation::Report
    analyze_compilation(mlir::ModuleOp module,
                        const std::vector<Mln::Weight::Descriptor> &required_weights,
                        const std::vector<Mln::Weight::Descriptor> &available_weights);
}
