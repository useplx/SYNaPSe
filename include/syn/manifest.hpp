#pragma once

#include "mln/sdk/operation/compatibility.hpp"
#include "syn/hook.hpp"
#include "syn/lowering.hpp"
#include "syn/police.hpp"

#include <functional>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Syn::Compiler {
    namespace Codec {
        class Compatibility;
    }

    struct Shape {
            static constexpr int64_t Dynamic = -1;

            std::vector<int64_t> dimensions;

            std::optional<Mln::DType> dtype;

            Shape() = default;

            Shape(std::initializer_list<int64_t> values) : dimensions(values) {}

            Shape(std::initializer_list<int64_t> values, Mln::DType element_dtype)
                : dimensions(values), dtype(element_dtype) {}
    };

    struct Specialization {
            std::vector<Shape> inputs;
    };

    struct Entrypoint {
            std::string name;

            std::string implementation;

            std::vector<Specialization> specializations;
    };

    struct Manifest {
            enum class Enforcement { Warn, Strict };

            enum class SchedulingFavor { MinMemory, MaxConcurrency, Debug };

            struct Validation {
                    Enforcement compatibility = Enforcement::Warn;
                    Enforcement numeric = Enforcement::Warn;
                    bool allow_fallback = true;
            } validation;

            SchedulingFavor scheduling_favor = SchedulingFavor::MinMemory;

            std::string tuning_spec;

            std::vector<std::pair<std::string, Police>> policies;

            std::vector<Mln::Sdk::Codec::Format> formats;

            std::vector<Codec::Compatibility *> codec_compatibilities;

            std::vector<Mln::Sdk::Operation::Compatibility *> operation_compatibilities;

            std::vector<Mln::Sdk::Lowering::Compatibility *> lowering_compatibilities;

            std::vector<Mln::Sdk::Backend::Compatibility *> backend_compatibilities;

            struct Link {
                    Mln::Sdk::Backend source;
                    Mln::Sdk::Backend target;
                    bool unified_memory = false;
                    bool transparent_access = false;
            };

            std::vector<Link> topology;

            std::vector<Entrypoint> entrypoints;

            std::vector<Hook> hooks;
    };

    void validate_manifest(const Manifest &manifest);

    Police police_for(const Manifest &manifest, const std::string &path);

    const Mln::Sdk::Codec::Scheme *scheme_for(const Police &police, Mln::Sdk::Codec::Scope scope,
                                              Mln::Sdk::Codec::Lifetime lifetime);

    namespace Scheme {
        struct Match {
                const Mln::Sdk::Codec::Scheme *scheme = nullptr;

                std::string path;

                explicit operator bool() const { return scheme != nullptr; }
        };
    }

    Scheme::Match scheme_for(const Manifest &manifest, const std::string &path,
                             Mln::Sdk::Codec::Scope scope, Mln::Sdk::Codec::Lifetime lifetime);

    Mln::Sdk::Codec &parameter_codec(const Manifest &manifest, const std::string &name);
}
