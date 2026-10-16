#include "json.hpp"
#include "mln/error.hpp"
#include "syn/errors/api.hpp"
#include "syn/errors/weight.hpp"

#include "mln/package.hpp"
#include "mln/runtime/metadata.hpp"
#include "mln/sdk/driver/module.hpp"
#include "mln/sdk/native/abi.hpp"
#include "syn/artifact.hpp"
#include "syn/codec_compatibility.hpp"
#include "syn/compiler.hpp"
#include "syn/compiler_abi.h"
#include "syn/package.hpp"
#include "syn/package_registry.hpp"
#include "syn/weights_pack.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <print>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
    using namespace Mln::Sdk;
    using Mln::raise;
    using Mln::raise_at;
    using Mln::Sdk::Backend;
    using Syn::Compiler::Entrypoint;
    using Syn::Compiler::Hook;
    using Syn::Compiler::Manifest;
    using Syn::Compiler::Police;
    using Syn::Compiler::Shape;
    using Syn::Compiler::Specialization;
    using Syn::Compiler::validate_artifact_plan;
    namespace Artifact = Mln::Artifact;
    namespace Compiler = Syn::Compiler;
    namespace Errors = Syn::Compiler::Errors;
    namespace Format = Mln::Weight;
    namespace Json = Syn::Compiler::Json;
    namespace Package = Syn::Compiler::Package;
    using Value = Json::Value;

    std::string optional_string(const Value &object, std::string_view key) {
        return object.contains(key) ? object.at(key).string() : std::string{};
    }

    template <typename T>
    T choice(std::string_view value, std::initializer_list<std::pair<std::string_view, T>> choices,
             std::string_view field) {
        for (const auto &[name, result] : choices)
            if (value == name)
                return result;
        raise(Errors::Api::Field::Value::Invalid, field, value);
    }

    Mln::DType graph_dtype(std::string_view value) {
        return choice<Mln::DType>(value,
                                  {
                                      {"f32", Mln::DType::Float32},
                                      {"f16", Mln::DType::Float16},
                                      {"bf16", Mln::DType::BFloat16},
                                      {"i32", Mln::DType::Int32},
                                      {"i16", Mln::DType::Int16},
                                      {"i8", Mln::DType::Int8},
                                      {"i4", Mln::DType::Int4},
                                      {"u64", Mln::DType::UInt64},
                                      {"u32", Mln::DType::UInt32},
                                      {"u8", Mln::DType::UInt8},
                                      {"u4", Mln::DType::UInt4},
                                  },
                                  "dtype");
    }

    Backend backend_from(const Value &value) {
        Backend backend{.device = value.at("device").string(), .flags = {}};
        if (value.contains("ordinal"))
            backend.ordinal = static_cast<int>(value.at("ordinal").integer());
        if (value.contains("flags"))
            for (const auto &[key, item] : value.at("flags").object())
                backend.flags.emplace_back(key, item.string());
        return backend;
    }

    std::string package_name(std::string_view export_id) {
        const std::size_t separator = export_id.find(':');
        if (separator == std::string_view::npos)
            return std::string(export_id);
        if (separator == 0 || separator + 1 == export_id.size())
            raise(Errors::Api::Export::Reference::Invalid, export_id);
        return std::string(export_id.substr(0, separator));
    }

    std::string module_name(std::string_view value) {
        std::string result;
        result.reserve(value.size());
        for (const char character : value)
            result.push_back(std::isalnum(static_cast<unsigned char>(character)) ? character : '_');
        return result;
    }

    void invoke(const Package::Export &item, Package::Invocation::Context &context,
                const Value *package_config) {
        std::array<char, 2048> error{};
        const std::string config =
            package_config ? Json::stringify(*package_config) : std::string("{}");
        const Package::Abi::Invocation invocation{
            Package::Abi::Version,
            sizeof(Package::Abi::Invocation),
            &context,
            config.data(),
            config.size(),
            error.data(),
            error.size(),
        };
        const int32_t status = item.invoke(&invocation, item.user_data);
        if (status != 0) {
            const std::string detail =
                error[0] ? std::string(error.data(), ::strnlen(error.data(), error.size()))
                         : std::format("callback returned {}", status);
            raise(Errors::Api::Export::Invocation::Failed, item.id(), detail);
        }
    }

    const Value *config_for(const Value &root, std::string_view package) {
        const Value &configs = root.at("package_config");
        const auto found = configs.object().find(std::string(package));
        return found == configs.object().end() ? nullptr : &found->second;
    }

    Codec::Format resolve_format(Package::Registry &registry, const Value &root,
                                 const std::string &reference, Manifest &manifest,
                                 std::map<std::string, Codec::Format> &formats) {
        if (const auto found = formats.find(reference); found != formats.end())
            return found->second;
        const Package::Export *item = registry.resolve(Package::Export::Kind::Codec, reference);
        if (!item)
            raise(Errors::Api::Codec::Unknown, reference);
        Codec::Format format;
        Package::Invocation::Context context{Package::Invocation::Kind::Codec, &manifest, nullptr,
                                             &format};
        invoke(*item, context, config_for(root, package_name(reference)));
        if (!format)
            raise(Errors::Api::Codec::Format::Missing, reference);
        manifest.formats.push_back(format);
        formats.emplace(reference, format);
        return format;
    }

    Codec::Scheme scheme_from(const Value &value, Package::Registry &registry, const Value &root,
                              Manifest &manifest, std::map<std::string, Codec::Format> &formats) {
        for (const std::string_view removed :
             {"granularity", "storage_layout", "parameters", "rounding", "saturation"}) {
            if (value.contains(removed))
                raise(Errors::Api::Codec::Scheme::Field::Owned::By::Codec, removed);
        }
        Codec::Scheme scheme;
        scheme.scope = Codec::Scope::Constant;
        const std::string scope = optional_string(value, "scope");
        if (!scope.empty())
            scheme.scope = choice<Codec::Scope>(scope,
                                                {
                                                    {"any", Codec::Scope::Any},
                                                    {"constant", Codec::Scope::Constant},
                                                    {"input", Codec::Scope::Input},
                                                    {"output", Codec::Scope::Output},
                                                    {"internal", Codec::Scope::Internal},
                                                },
                                                "scheme scope");
        scheme.lifetime = Codec::Lifetime::Immutable;
        const std::string lifetime = optional_string(value, "lifetime");
        if (!lifetime.empty())
            scheme.lifetime =
                choice<Codec::Lifetime>(lifetime,
                                        {
                                            {"any", Codec::Lifetime::Any},
                                            {"immutable", Codec::Lifetime::Immutable},
                                            {"invocation", Codec::Lifetime::Invocation},
                                            {"session", Codec::Lifetime::Session},
                                        },
                                        "scheme lifetime");
        scheme.update = scheme.lifetime == Codec::Lifetime::Immutable ? Codec::Update::None
                                                                      : Codec::Update::Whole;
        const std::string update = optional_string(value, "update");
        if (!update.empty())
            scheme.update = choice<Codec::Update>(update,
                                                  {
                                                      {"none", Codec::Update::None},
                                                      {"whole", Codec::Update::Whole},
                                                      {"region", Codec::Update::Region},
                                                  },
                                                  "scheme update");
        if (value.contains("volatile") && !value.at("volatile").null())
            scheme.residency = value.at("volatile").boolean() ? Mln::Weight::Residency::Volatile
                                                              : Mln::Weight::Residency::Retained;
        scheme.format =
            resolve_format(registry, root, value.at("format").string(), manifest, formats);

        return scheme;
    }

    Manifest manifest_from(const Value &root, Package::Registry &registry) {
        Manifest manifest;
        if (root.contains("tuning_spec"))
            manifest.tuning_spec = root.at("tuning_spec").string();
        if (root.contains("scheduling") && root.at("scheduling").contains("favor")) {
            manifest.scheduling_favor = choice<Manifest::SchedulingFavor>(
                root.at("scheduling").at("favor").string(),
                {{"min-memory", Manifest::SchedulingFavor::MinMemory},
                 {"max-concurrency", Manifest::SchedulingFavor::MaxConcurrency},
                 {"debug", Manifest::SchedulingFavor::Debug}},
                "scheduling.favor");
        }
        if (root.contains("validation")) {
            const Value &validation = root.at("validation");
            if (validation.contains("compatibility"))
                manifest.validation.compatibility =
                    choice<Manifest::Enforcement>(validation.at("compatibility").string(),
                                                  {{"warn", Manifest::Enforcement::Warn},
                                                   {"strict", Manifest::Enforcement::Strict}},
                                                  "validation.compatibility");
            if (validation.contains("numeric"))
                manifest.validation.numeric =
                    choice<Manifest::Enforcement>(validation.at("numeric").string(),
                                                  {{"warn", Manifest::Enforcement::Warn},
                                                   {"strict", Manifest::Enforcement::Strict}},
                                                  "validation.numeric");
            if (validation.contains("allow_fallback"))
                manifest.validation.allow_fallback = validation.at("allow_fallback").boolean();
        }
        if (root.contains("topology") && root.at("topology").contains("links")) {
            for (const Value &value : root.at("topology").at("links").array()) {
                Manifest::Link link{
                    .source = backend_from(value.at("source")),
                    .target = backend_from(value.at("target")),
                };
                if (value.contains("unified_memory"))
                    link.unified_memory = value.at("unified_memory").boolean();
                if (value.contains("transparent_access"))
                    link.transparent_access = value.at("transparent_access").boolean();
                manifest.topology.push_back(std::move(link));
            }
        }
        std::map<std::string, Codec::Format> formats;
        for (const auto &[path, value] : root.at("policies").object()) {
            Police police;
            const std::string dtype = optional_string(value, "dtype");
            if (!dtype.empty())
                police.dtype = graph_dtype(dtype);
            if (value.contains("backend") && !value.at("backend").null()) {
                police.backend = backend_from(value.at("backend"));
            }
            if (value.contains("entrypoints") && !value.at("entrypoints").null()) {
                std::vector<std::string> entrypoints;
                for (const Value &item : value.at("entrypoints").array())
                    entrypoints.push_back(item.string());
                police.entrypoints = std::move(entrypoints);
            }
            if (value.contains("volatilized") && !value.at("volatilized").null())
                raise(Errors::Api::Policy::Volatilized::Removed, path);
            if (value.contains("optimization_barrier") && !value.at("optimization_barrier").null())
                police.optimization_barrier = value.at("optimization_barrier").boolean();
            if (value.contains("loading") && !value.at("loading").null()) {
                const Value &loading = value.at("loading");
                police.loading = choice<Syn::Compiler::LoadingMode>(
                    loading.at("mode").string(),
                    {{"resident", Syn::Compiler::LoadingMode::Resident},
                     {"streamed", Syn::Compiler::LoadingMode::Streamed}},
                    "policy loading mode");
            }
            if (value.contains("schemes"))
                for (const Value &item : value.at("schemes").array())
                    police.schemes.push_back(scheme_from(item, registry, root, manifest, formats));
            manifest.policies.emplace_back(path, std::move(police));
        }

        if (root.contains("formats"))
            for (const Value &value : root.at("formats").array())
                resolve_format(registry, root, value.string(), manifest, formats);

        for (const Value &value : root.at("hooks").array()) {
            if (!value.contains("implementation"))
                continue;
            const std::string reference = value.at("implementation").string();
            const Package::Export *item = registry.resolve(Package::Export::Kind::Hook, reference);
            if (!item)
                raise(Errors::Api::Hook::Unknown, reference);
            Hook hook;
            hook.path = value.at("path").string();
            Package::Invocation::Context context{Package::Invocation::Kind::Hook, &manifest,
                                                 nullptr, &hook};
            invoke(*item, context, config_for(root, package_name(reference)));
            manifest.hooks.push_back(std::move(hook));
        }
        return manifest;
    }

    void load_codec_compatibilities(Package::Registry &registry, const Value &root,
                                    Manifest &manifest) {
        struct Resolved {
                std::string id;
                Compiler::Codec::Compatibility *value;
        };
        std::vector<Resolved> resolved;
        for (const Package::Export &item : registry.exports()) {
            if (item.kind != Package::Export::Kind::CodecCompatibility)
                continue;
            Compiler::Codec::Compatibility *compatibility = nullptr;
            Package::Invocation::Context context{Package::Invocation::Kind::CodecCompatibility,
                                                 &manifest, nullptr, &compatibility};
            invoke(item, context, config_for(root, package_name(item.id())));
            if (!compatibility)
                raise(Errors::Api::Codec::Compatibility::Missing, item.id());
            resolved.push_back({item.id(), compatibility});
        }
        std::stable_sort(resolved.begin(), resolved.end(),
                         [](const Resolved &left, const Resolved &right) {
                             if (left.value->priority() != right.value->priority())
                                 return left.value->priority() < right.value->priority();
                             return left.id < right.id;
                         });
        for (const Resolved &item : resolved)
            manifest.codec_compatibilities.push_back(item.value);
    }

    void load_operation_compatibilities(Package::Registry &registry, const Value &root,
                                        Manifest &manifest) {
        struct Resolved {
                std::string id;
                Mln::Sdk::Operation::Compatibility *value;
        };
        std::vector<Resolved> resolved;
        for (const Package::Export &item : registry.exports()) {
            if (item.kind != Package::Export::Kind::OperationCompatibility)
                continue;
            Mln::Sdk::Operation::Compatibility *compatibility = nullptr;
            Package::Invocation::Context context{Package::Invocation::Kind::OperationCompatibility,
                                                 &manifest, nullptr, &compatibility};
            invoke(item, context, config_for(root, package_name(item.id())));
            if (!compatibility)
                raise(Errors::Api::Operation::Compatibility::Missing, item.id());
            resolved.push_back({item.id(), compatibility});
        }
        std::stable_sort(resolved.begin(), resolved.end(),
                         [](const Resolved &left, const Resolved &right) {
                             if (left.value->priority() != right.value->priority())
                                 return left.value->priority() < right.value->priority();
                             return left.id < right.id;
                         });
        for (const Resolved &item : resolved)
            manifest.operation_compatibilities.push_back(item.value);
    }

    void load_lowering_compatibilities(Package::Registry &registry, const Value &root,
                                       Manifest &manifest) {
        struct Resolved {
                std::string id;
                Mln::Sdk::Lowering::Compatibility *value;
        };
        std::vector<Resolved> resolved;
        for (const Package::Export &item : registry.exports()) {
            if (item.kind != Package::Export::Kind::LoweringCompatibility)
                continue;
            Mln::Sdk::Lowering::Compatibility *compatibility = nullptr;
            Package::Invocation::Context context{Package::Invocation::Kind::LoweringCompatibility,
                                                 &manifest, nullptr, &compatibility};
            invoke(item, context, config_for(root, package_name(item.id())));
            if (!compatibility)
                raise(Errors::Api::Lowering::Compatibility::Missing, item.id());
            resolved.push_back({item.id(), compatibility});
        }
        std::stable_sort(resolved.begin(), resolved.end(),
                         [](const Resolved &left, const Resolved &right) {
                             if (left.value->priority() != right.value->priority())
                                 return left.value->priority() < right.value->priority();
                             return left.id < right.id;
                         });
        for (const Resolved &item : resolved)
            manifest.lowering_compatibilities.push_back(item.value);
    }

    bool load_backend_compatibilities(Package::Registry &registry, const Value &root,
                                      Manifest &manifest) {
        bool loaded = false;
        for (const Package::Export &item : registry.exports()) {
            if (item.kind != Package::Export::Kind::BackendCompatibility)
                continue;
            Mln::Sdk::Backend::Compatibility *compatibility = nullptr;
            Package::Invocation::Context context{Package::Invocation::Kind::BackendCompatibility,
                                                 &manifest, nullptr, &compatibility};
            invoke(item, context, config_for(root, package_name(item.id())));
            if (!compatibility)
                raise(Errors::Api::Backend::Compatibility::Missing, item.id());
            manifest.backend_compatibilities.push_back(compatibility);
            loaded = true;
        }
        return loaded;
    }

    Mln::Artifact::File artifact(const Value &value) {
        return {value.at("name").string(), value.at("path").string()};
    }

    Mln::Artifact::Plan artifact_plan(const Value &root) {
        Mln::Artifact::Plan artifacts;
        for (const Value &value : root.at("modules").array()) {
            const Mln::Artifact::Runtime runtime =
                choice<Mln::Artifact::Runtime>(value.at("runtime").string(),
                                               {
                                                   {"wasm", Mln::Artifact::Runtime::Wasm},
                                                   {"native", Mln::Artifact::Runtime::Native},
                                                   {"driver", Mln::Artifact::Runtime::Driver},
                                               },
                                               "module runtime");
            artifacts.modules.push_back({runtime, artifact(value)});
        }
        for (const Value &value : root.at("assets").array())
            artifacts.assets.push_back(artifact(value));

        std::vector<std::string> wasm_modules;
        for (const Mln::Artifact::Module &module : artifacts.modules)
            if (module.runtime == Mln::Artifact::Runtime::Wasm)
                wasm_modules.push_back(module.artifact.name);
        for (const Value &value : root.at("entrypoints").array()) {
            const Mln::Artifact::Runtime runtime =
                choice<Mln::Artifact::Runtime>(value.at("runtime").string(),
                                               {{"graph", Mln::Artifact::Runtime::Graph},
                                                {"wasm", Mln::Artifact::Runtime::Wasm},
                                                {"native", Mln::Artifact::Runtime::Native}},
                                               "entrypoint runtime");
            std::string export_name = optional_string(value, "export");
            if (export_name.empty())
                export_name = "forward";
            std::string module = optional_string(value, "module");
            if (module.empty() && wasm_modules.size() == 1)
                module = wasm_modules.front();
            if (module.empty())
                raise(Errors::Api::Entrypoint::Wasm::Module::Missing, value.at("name").string());
            artifacts.entrypoints.push_back(
                {value.at("name").string(), std::move(module), std::move(export_name), runtime});
        }
        for (const Value &value : root.at("hooks").array()) {
            if (!value.contains("module"))
                continue;
            std::string export_name = optional_string(value, "export");
            if (export_name.empty())
                export_name = "forward";
            artifacts.hooks.push_back(
                {value.at("path").string(), value.at("module").string(), std::move(export_name)});
        }
        return artifacts;
    }

    int repackage(const Value &root, const std::string &output_path) {
        const Mln::Artifact::Plan artifacts = artifact_plan(root);
        validate_artifact_plan(artifacts);
        const auto base = Mln::Package::File::open(root.at("base_package").string());
        Mln::Package::Builder builder;

        for (const Mln::Package::Section::Type type :
             {Mln::Package::Section::Type::Devices, Mln::Package::Section::Type::Vmfb,
              Mln::Package::Section::Type::WeightIndex, Mln::Package::Section::Type::Deployment,
              Mln::Package::Section::Type::Entrypoints}) {
            for (const Mln::Package::Section &section : base->sections(type))
                builder.add(type, std::string(section.name), section.bytes);
        }
        std::shared_ptr<Mln::Package::File> parameters;
        if (root.contains("parameters_package")) {
            parameters = Mln::Package::File::open(root.at("parameters_package").string());
            for (const Mln::Package::Section &section :
                 parameters->sections(Mln::Package::Section::Type::Parameters))
                builder.add(Mln::Package::Section::Type::Parameters, std::string(section.name),
                            section.bytes);
        } else {
            for (const Mln::Package::Section &section :
                 base->sections(Mln::Package::Section::Type::Parameters))
                builder.add(Mln::Package::Section::Type::Parameters, std::string(section.name),
                            section.bytes);
        }
        for (const Mln::Artifact::Module &module : artifacts.modules) {
            const Mln::Package::Section::Type type =
                module.runtime == Mln::Artifact::Runtime::Wasm ? Mln::Package::Section::Type::Wasm
                : module.runtime == Mln::Artifact::Runtime::Driver
                    ? Mln::Package::Section::Type::Driver
                    : Mln::Package::Section::Type::Native;
            builder.add(type, module.artifact.name,
                        Mln::Weight::read_binary_file(module.artifact.path));
        }
        for (const Mln::Artifact::File &asset : artifacts.assets)
            builder.add(Mln::Package::Section::Type::Asset, asset.name,
                        Mln::Weight::read_binary_file(asset.path));

        std::string entrypoints;
        for (const Mln::Artifact::Entrypoint &entrypoint : artifacts.entrypoints) {
            entrypoints.append(entrypoint.name);
            entrypoints += '\t';
            entrypoints.append(entrypoint.runtime == Mln::Artifact::Runtime::Graph    ? "graph"
                               : entrypoint.runtime == Mln::Artifact::Runtime::Native ? "native"
                                                                                      : "wasm");
            entrypoints += '\t';
            entrypoints.append(entrypoint.module);
            entrypoints += '\t';
            entrypoints.append(entrypoint.export_name);
            entrypoints += '\n';
        }
        builder.add(Mln::Package::Section::Type::ApplicationEntrypoints, "public",
                    std::vector<uint8_t>(entrypoints.begin(), entrypoints.end()));
        std::string hooks;
        for (const Mln::Artifact::Hook &hook : artifacts.hooks) {
            hooks.append(hook.path);
            hooks += '\t';
            hooks.append(hook.module);
            hooks += '\t';
            hooks.append(hook.export_name);
            hooks += '\n';
        }
        if (!hooks.empty())
            builder.add(Mln::Package::Section::Type::Hooks, "routes",
                        std::vector<uint8_t>(hooks.begin(), hooks.end()));
        const bool native = std::ranges::any_of(artifacts.modules, [](const auto &module) {
            return module.runtime == Mln::Artifact::Runtime::Native;
        });
        const bool driver = std::ranges::any_of(artifacts.modules, [](const auto &module) {
            return module.runtime == Mln::Artifact::Runtime::Driver;
        });
        builder.add(Mln::Package::Section::Type::Requirements, "runtime",
                    Mln::encode_requirements({
                        .runtime_abi = 1,
                        .native_abi = native ? Mln::Sdk::Native::ABI_VERSION : 0,
                        .driver_abi = driver ? Mln::Sdk::Driver::AbiVersion : 0,
                        .platform = (native || driver) ? Mln::current_platform() : std::string{},
                    }));
        builder.write(output_path);
        return 0;
    }

    int merge_graphs(const Value &root, const std::string &output_path) {
        Mln::Package::Builder builder;
        std::vector<std::shared_ptr<Mln::Package::File>> packages;
        std::vector<Mln::Weight::Descriptor> requirements;
        std::map<std::string, Mln::Weight::Descriptor, std::less<>> requirement_by_name;
        std::set<std::string> module_names;
        std::vector<uint8_t> devices;
        std::string deployment;
        std::string entrypoints;

        for (const Value &item : root.at("graphs").array()) {
            const std::string module_name = item.at("name").string();
            if (!module_names.insert(module_name).second)
                throw std::runtime_error("duplicate graph shard module: " + module_name);
            auto package = Mln::Package::File::open(item.at("path").string());
            const auto device = package->required(Mln::Package::Section::Type::Devices);
            if (devices.empty())
                devices.assign(device.bytes.begin(), device.bytes.end());
            else if (!std::ranges::equal(devices, device.bytes))
                throw std::runtime_error("graph shards declare different device tables");

            const auto vmfbs = package->sections(Mln::Package::Section::Type::Vmfb);
            if (vmfbs.size() != 1)
                throw std::runtime_error("graph shard must contain exactly one VMFB");
            builder.add(Mln::Package::Section::Type::Vmfb, module_name, vmfbs.front().bytes);

            for (const auto &section : package->sections(Mln::Package::Section::Type::Deployment))
                deployment.append(reinterpret_cast<const char *>(section.bytes.data()),
                                  section.bytes.size());
            for (const auto &section : package->sections(Mln::Package::Section::Type::Entrypoints))
                entrypoints.append(reinterpret_cast<const char *>(section.bytes.data()),
                                   section.bytes.size());

            const auto index =
                package->required(Mln::Package::Section::Type::WeightIndex, "requirements");
            for (auto descriptor : Mln::Weight::decode_weight_index(
                     index.bytes, Mln::Weight::Index::Kind::Requirements)) {
                const auto existing = requirement_by_name.find(descriptor.name);
                if (existing == requirement_by_name.end()) {
                    requirement_by_name.emplace(descriptor.name, descriptor);
                    requirements.push_back(std::move(descriptor));
                    continue;
                }
                const auto &known = existing->second;
                if (known.quant != descriptor.quant || known.storage != descriptor.storage ||
                    known.shape != descriptor.shape ||
                    known.byte_length != descriptor.byte_length ||
                    known.residency != descriptor.residency)
                    throw std::runtime_error("graph shards disagree on parameter: " +
                                             descriptor.name);
            }
            packages.push_back(std::move(package));
        }
        if (packages.empty())
            throw std::runtime_error("merge-graphs requires at least one graph shard");

        builder.add(Mln::Package::Section::Type::Devices, "devices", std::move(devices));
        builder.add(Mln::Package::Section::Type::Deployment, "entrypoints",
                    std::vector<uint8_t>(deployment.begin(), deployment.end()));
        builder.add(Mln::Package::Section::Type::Entrypoints, "typed",
                    std::vector<uint8_t>(entrypoints.begin(), entrypoints.end()));
        builder.add(
            Mln::Package::Section::Type::WeightIndex, "requirements",
            Mln::Weight::encode_weight_index(Mln::Weight::Index::Kind::Requirements, requirements));
        builder.write(output_path);
        return 0;
    }

    int compile_plan(std::string_view plan, const std::string &output_path) {
        const Value root = Json::parse(plan);
        if (root.at("version").integer() != 1)
            raise(Errors::Api::Plan::Version::Unsupported);
        const std::string mode = optional_string(root, "mode");
        if (mode == "package")
            return repackage(root, output_path);
        if (mode == "merge-graphs")
            return merge_graphs(root, output_path);
        if (mode == "requirements") {
            const auto base = Mln::Package::File::open(root.at("base_package").string());
            const Mln::Package::Section requirements =
                base->required(Mln::Package::Section::Type::WeightIndex, "requirements");
            Mln::Weight::write_binary_file(
                output_path,
                std::vector<uint8_t>(requirements.bytes.begin(), requirements.bytes.end()));
            return 0;
        }
        if (!mode.empty() && mode != "compile" && mode != "core" && mode != "graph" &&
            mode != "weights")
            raise(Errors::Api::Plan::Mode::Unsupported, mode);

        Package::Registry registry;
        try {
            for (const Value &plugin : root.at("plugins").array())
                registry.load(plugin.at("path").string());
            Manifest manifest = manifest_from(root, registry);
            load_codec_compatibilities(registry, root, manifest);
            load_operation_compatibilities(registry, root, manifest);
            load_lowering_compatibilities(registry, root, manifest);

            if (load_backend_compatibilities(registry, root, manifest))
                registry.pin();

            struct Graph {
                    Package::Export item;
                    std::unique_ptr<Mln::Sdk::Module> module;
                    std::string name_space;
            };
            struct Graphs {
                    std::vector<Graph> values;

                    ~Graphs() {

                        for (auto graph = values.rbegin(); graph != values.rend(); ++graph) {
                            graph->module->operations.clear();
                            graph->module->modules.clear();
                        }
                    }
            } graphs;
            if (mode != "weights") {
                for (const Package::Export &item : registry.exports()) {
                    if (item.kind != Package::Export::Kind::Graph)
                        continue;
                    const std::string reference = item.id();
                    std::string described;
                    Package::Invocation::Context context{Package::Invocation::Kind::Describe,
                                                         &manifest, nullptr, &described};
                    invoke(item, context, config_for(root, package_name(reference)));
                    if (described.empty())
                        raise(Errors::Api::Graph::Namespace::Missing, reference);

                    std::unique_ptr<Mln::Sdk::Module> module;
                    context = {Package::Invocation::Kind::Graph, &manifest, nullptr, &module};
                    invoke(item, context, config_for(root, package_name(reference)));
                    if (!module)
                        raise(Errors::Api::Graph::Module::Missing, reference);

                    Entrypoint entrypoint;
                    entrypoint.name = module->entrypoint();
                    entrypoint.implementation = reference;
                    for (const auto &inputs : module->specializations()) {
                        Specialization specialization;
                        for (const Mln::Sdk::Tensor &input : inputs) {
                            Shape shape;
                            shape.dimensions = input.shape;
                            shape.dtype = input.dtype;
                            specialization.inputs.push_back(std::move(shape));
                        }
                        entrypoint.specializations.push_back(std::move(specialization));
                    }
                    manifest.entrypoints.push_back(std::move(entrypoint));
                    graphs.values.push_back({item, std::move(module), std::move(described)});
                }
            }

            std::vector<std::string> weight_paths;
            const Json::Array &weights = root.at("weights").array();
            std::vector<Mln::Weight::Descriptor> available_weights;
            std::set<std::string> available_names;
            for (const Value &weight : weights) {
                const std::string path = weight.at("path").string();
                auto descriptors = Syn::Compiler::Weight::read(path);
                for (auto &descriptor : descriptors) {
                    if (!available_names.insert(descriptor.name).second)
                        raise(Errors::Api::Weight::Duplicated::Across::Roots, descriptor.name);
                    available_weights.push_back(std::move(descriptor));
                }
                weight_paths.push_back(path);
            }
            if (mode == "graph" && root.contains("graph_entrypoint")) {
                const std::string selected = root.at("graph_entrypoint").string();
                std::erase_if(manifest.entrypoints, [&](const Entrypoint &entrypoint) {
                    return entrypoint.name != selected;
                });
                if (manifest.entrypoints.empty())
                    throw std::runtime_error("graph entrypoint not found: " + selected);
            }
            std::string selected_graph_implementation;
            if (mode == "graph" && root.contains("graph_implementation")) {
                selected_graph_implementation = root.at("graph_implementation").string();
                std::erase_if(manifest.entrypoints, [&](const Entrypoint &entrypoint) {
                    return entrypoint.implementation != selected_graph_implementation;
                });
                if (manifest.entrypoints.empty())
                    throw std::runtime_error("graph implementation not found: " +
                                             selected_graph_implementation);
            }
            Syn::Compiler::Compiler compiler(manifest, std::string{}, std::move(available_weights));
            if (!selected_graph_implementation.empty())
                compiler.vm_module_name(module_name(selected_graph_implementation));
            if (mode == "weights") {
                const int64_t conversion_jobs = root.at("conversion_jobs").integer();
                const int64_t conversion_memory_bytes =
                    root.at("conversion_memory_bytes").integer();
                if (conversion_jobs <= 0 || conversion_memory_bytes <= 0)
                    throw std::runtime_error("weight conversion resources must be positive");
                std::vector<uint8_t> requirements_bytes;
                requirements_bytes =
                    Mln::Weight::read_binary_file(root.at("requirements").string());
                compiler.write_parameters_cache(
                    weight_paths,
                    Mln::Weight::decode_weight_index(requirements_bytes,
                                                     Mln::Weight::Index::Kind::Requirements),
                    output_path, static_cast<std::size_t>(conversion_jobs),
                    static_cast<uint64_t>(conversion_memory_bytes));
                return 0;
            }
            for (const Graph &graph : graphs.values) {
                if (!selected_graph_implementation.empty() &&
                    graph.item.id() != selected_graph_implementation)
                    continue;
                compiler.compile_package_entrypoints(*graph.module, graph.item.id(),
                                                     graph.name_space);
            }

            std::shared_ptr<Mln::Package::File> backend_parameter_package;
            std::vector<Syn::Compiler::Weight::Conversion::Result> inspected_archives;
            if (mode == "graph" && root.contains("parameters_package")) {
                backend_parameter_package =
                    Mln::Package::File::open(root.at("parameters_package").string());
                std::vector<Mln::Sdk::Backend::Resource> resources;
                const auto parameter_sections =
                    backend_parameter_package->sections(Mln::Package::Section::Type::Parameters);
                inspected_archives.reserve(parameter_sections.size());
                if (const char *trace = std::getenv("MLN_TRACE_BACKEND"); trace && *trace)
                    std::println(stderr, "backend parameter package: {} sections",
                                 parameter_sections.size());
                for (const Mln::Package::Section &section : parameter_sections) {
                    inspected_archives.push_back(
                        Syn::Compiler::Weight::Conversion::inspect(section.bytes));
                    const auto &inspected = inspected_archives.back();
                    if (const char *trace = std::getenv("MLN_TRACE_BACKEND"); trace && *trace)
                        std::println(
                            stderr, "backend parameter section {}: {} descriptors, {} resources",
                            section.name, inspected.descriptors.size(), inspected.resources.size());
                    for (std::size_t i = 0; i < inspected.resources.size(); ++i) {
                        const auto &resource = inspected.resources[i];
                        const auto descriptor =
                            std::ranges::find_if(inspected.descriptors, [&](const auto &candidate) {
                                return candidate.name == resource.name;
                            });
                        if (descriptor == inspected.descriptors.end() ||
                            resource.offset > section.bytes.size() ||
                            resource.length > section.bytes.size() - resource.offset)
                            raise(Syn::Compiler::Errors::Weight::Storage::Outside::File,
                                  resource.name);
                        resources.push_back(Mln::Sdk::Backend::Resource{
                            .name = descriptor->name,
                            .format = descriptor->quant,
                            .shape = descriptor->shape,
                            .data = std::as_bytes(
                                section.bytes.subspan(static_cast<std::size_t>(resource.offset),
                                                      static_cast<std::size_t>(resource.length))),
                        });
                    }
                }
                if (const char *trace = std::getenv("MLN_TRACE_BACKEND"); trace && *trace)
                    std::println(stderr, "backend parameters loaded: {} resources",
                                 resources.size());
                compiler.backend_resources(std::move(resources),
                                           root.at("parameters_package").string());
            }

            if (mode == "graph") {
                if (!root.contains("parameters_package") && root.contains("requirements_output")) {
                    compiler.write_requirements(root.at("requirements_output").string());
                    return 0;
                }
                compiler.write_graph_cache(output_path,
                                           optional_string(root, "requirements_output"));
                return 0;
            }

            Mln::Artifact::Plan artifacts;
            if (mode == "core") {
            } else {
                artifacts = artifact_plan(root);
            }
            compiler.write_package(weight_paths, output_path, artifacts, mode != "core");
            return 0;
        } catch (const Mln::Diagnostic &diagnostic) {
            const Mln::Error preserved(diagnostic.error().domain(), diagnostic.error().code(),
                                       "package diagnostic");
            throw Mln::Diagnostic(preserved, std::string(diagnostic.message()),
                                  std::string(diagnostic.location()), diagnostic.severity());
        } catch (const std::exception &error) {
            raise(Errors::Api::Plugin::Exception::Escaped, error.what());
        }
    }
}

extern "C" const mln_compiler_api *mln_compiler_api_get() {
    constexpr std::size_t request_v2_size =
        offsetof(mln_compiler_request, structured_diagnostic) +
        sizeof(decltype(mln_compiler_request::structured_diagnostic));
    constexpr std::size_t actionable_request_size =
        offsetof(mln_compiler_request, actionable_diagnostic) +
        sizeof(decltype(mln_compiler_request::actionable_diagnostic));
    static const mln_compiler_api api{
        mln_compiler_api_version,
        sizeof(mln_compiler_api),
        [](const mln_compiler_request *request) -> int32_t {
            if (!request || request->abi_version != mln_compiler_api_version ||
                request->struct_size < request_v2_size || !request->plan || !request->output_path)
                return 2;
            try {
                const auto forward = [](const Mln::Diagnostic &diagnostic,
                                        void *opaque) -> Mln::DiagnosticAction {
                    const auto &active = *static_cast<const mln_compiler_request *>(opaque);
                    if (active.struct_size >= actionable_request_size &&
                        active.actionable_diagnostic) {
                        switch (active.actionable_diagnostic(
                            active.user_data,
                            diagnostic.severity() == Mln::Severity::Error
                                ? MLN_COMPILER_DIAGNOSTIC_ERROR
                                : MLN_COMPILER_DIAGNOSTIC_WARNING,
                            diagnostic.error().domain().data(), diagnostic.error().domain().size(),
                            diagnostic.error().code(), diagnostic.message().data(),
                            diagnostic.message().size(), diagnostic.location().data(),
                            diagnostic.location().size())) {
                        case MLN_COMPILER_DIAGNOSTIC_IGNORE:
                            return Mln::DiagnosticAction::Ignore;
                        case MLN_COMPILER_DIAGNOSTIC_PROMOTE_TO_ERROR:
                            return Mln::DiagnosticAction::PromoteToError;
                        case MLN_COMPILER_DIAGNOSTIC_CONTINUE:
                        default:
                            return Mln::DiagnosticAction::Continue;
                        }
                    }
                    if (!active.structured_diagnostic)
                        return Mln::DiagnosticAction::Continue;
                    active.structured_diagnostic(
                        active.user_data,
                        diagnostic.severity() == Mln::Severity::Error
                            ? MLN_COMPILER_DIAGNOSTIC_ERROR
                            : MLN_COMPILER_DIAGNOSTIC_WARNING,
                        diagnostic.error().domain().data(), diagnostic.error().domain().size(),
                        diagnostic.error().code(), diagnostic.message().data(),
                        diagnostic.message().size(), diagnostic.location().data(),
                        diagnostic.location().size());
                    return Mln::DiagnosticAction::Ignore;
                };
                Mln::ScopedDiagnosticHandler diagnostics(
                    forward, const_cast<mln_compiler_request *>(request));
                return compile_plan(std::string_view(request->plan, request->plan_length),
                                    request->output_path);
            } catch (const Mln::Diagnostic &diagnostic) {
                if (request->structured_diagnostic) {
                    request->structured_diagnostic(
                        request->user_data, MLN_COMPILER_DIAGNOSTIC_ERROR,
                        diagnostic.error().domain().data(), diagnostic.error().domain().size(),
                        diagnostic.error().code(), diagnostic.message().data(),
                        diagnostic.message().size(), diagnostic.location().data(),
                        diagnostic.location().size());
                } else if (request->diagnostic) {
                    request->diagnostic(request->user_data, diagnostic.what(),
                                        std::char_traits<char>::length(diagnostic.what()));
                }
                return 1;
            } catch (const std::exception &error) {
                if (request->diagnostic)
                    request->diagnostic(request->user_data, error.what(),
                                        std::char_traits<char>::length(error.what()));
                return 1;
            }
        },
    };
    return &api;
}
