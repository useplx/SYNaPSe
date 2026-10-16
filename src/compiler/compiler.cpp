#include "syn/compiler.hpp"
#include "mln/error.hpp"
#include "syn/errors/compiler.hpp"
#include "syn/errors/weight.hpp"

#include "mln/package.hpp"
#include "mln/runtime/metadata.hpp"
#include "mln/sdk/driver/module.hpp"
#include "mln/sdk/native/abi.hpp"
#include "syn/weights_pack.hpp"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/embedding_api.h"
#include "iree/compiler/mlir_interop.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace Syn::Compiler {
    void register_vm_compatibility_passes();
    void register_dispatch_affinity_passes();

    namespace {

        std::mutex iree_compilation_mutex;

        struct Global {
                Global() { ireeCompilerGlobalInitialize(); }
                ~Global() { ireeCompilerGlobalShutdown(); }
        };

        namespace Session {
            struct Deleter {
                    void operator()(iree_compiler_session_t *value) const {
                        ireeCompilerSessionDestroy(value);
                    }
            };
        }

        namespace Invocation {
            struct Deleter {
                    void operator()(iree_compiler_invocation_t *value) const {
                        ireeCompilerInvocationDestroy(value);
                    }
            };
        }

        namespace Source {
            struct Deleter {
                    void operator()(iree_compiler_source_t *value) const {
                        ireeCompilerSourceDestroy(value);
                    }
            };
        }

        namespace Output {
            struct Deleter {
                    void operator()(iree_compiler_output_t *value) const {
                        ireeCompilerOutputDestroy(value);
                    }
            };
        }

        void check(iree_compiler_error_t *iree_error, const Mln::Error &error) {
            if (!iree_error)
                return;
            std::string message = ireeCompilerErrorGetMessage(iree_error);
            ireeCompilerErrorDestroy(iree_error);
            raise(error, message);
        }

        void append_diagnostic(iree_compiler_diagnostic_severity_t, const char *message,
                               size_t size, void *user_data) {
            auto &diagnostics = *static_cast<std::string *>(user_data);
            if (!diagnostics.empty())
                diagnostics.push_back('\n');
            diagnostics.append(message, size);
        }

        void append_unique(std::vector<std::string> &values, std::string value) {
            if (std::find(values.begin(), values.end(), value) == values.end())
                values.push_back(std::move(value));
        }

        bool matches_hook_path(std::string_view pattern, std::string_view path) {
            while (true) {
                const std::size_t pattern_end = pattern.find('.');
                const std::size_t path_end = path.find('.');
                const std::string_view pattern_segment = pattern.substr(0, pattern_end);
                const std::string_view path_segment = path.substr(0, path_end);

                if (pattern_segment != "*" && pattern_segment != path_segment)
                    return false;

                const bool pattern_done = pattern_end == std::string_view::npos;
                const bool path_done = path_end == std::string_view::npos;
                if (pattern_done || path_done)
                    return pattern_done && path_done;

                pattern.remove_prefix(pattern_end + 1);
                path.remove_prefix(path_end + 1);
            }
        }

    }

    Mln::Sdk::Tensor Compiler::apply(std::string_view path, Mln::Sdk::Module &model,
                                     std::span<const Mln::Sdk::Tensor> inputs,
                                     Mln::Sdk::Tensor output) {
        for (std::size_t index = 0; index < manifest_.hooks.size(); ++index) {
            const Syn::Compiler::Hook &hook = manifest_.hooks[index];
            if (!matches_hook_path(hook.path, path))
                continue;

            ++hook_hits_[index];
            output = hook.forward(model, inputs, std::move(output));
            if (!output.def) {
                raise(Errors::Compiler::Hook::Output::Missing, hook.path, path);
            }
        }
        return output;
    }

    void Compiler::finalize() {
        if (finalized_)
            return;
        for (std::size_t index = 0; index < manifest_.hooks.size(); ++index) {
            if (hook_hits_[index] == 0) {
                raise(Errors::Compiler::Hook::Path::Unmatched, manifest_.hooks[index].path);
            }
        }

        for (const Entrypoint &entrypoint : manifest_.entrypoints) {
            if (!entrypoints_compiled_.contains(entrypoint.name)) {
                raise(Errors::Compiler::Entrypoint::Not::Compiled, entrypoint.name);
            }
        }

        if (mlir::failed(mlir::verify(*module))) {
            if (const char *dump_path = std::getenv("MLN_DUMP_MLIR"); dump_path && *dump_path) {
                std::error_code error;
                llvm::raw_fd_ostream dump(dump_path, error);
                if (error)
                    raise(Errors::Compiler::Mlir::Dump::Write::Failed, dump_path);
                module->print(dump);
            }
            raise(Errors::Compiler::Module::Invalid);
        }

        report_ = analyze_compilation(*module, weights_, available_weights_);
        report_.legalizations = legalizations_;
        report_.compatibilities = matrix_.entries();
        for (const Mln::Sdk::Backend &backend : deployment_backends(manifest_))
            report_.partitioning.target_backends.push_back(runtime_device_uri(backend));
        if (!entrypoint_backends_.empty()) {
            uint64_t partitions = 1;
            uint64_t transitions = 0;
            for (std::size_t i = 1; i < entrypoint_backends_.size(); ++i) {
                const Mln::Sdk::Backend &previous = entrypoint_backends_[i - 1].second;
                const Mln::Sdk::Backend &current = entrypoint_backends_[i].second;
                if (previous.device == current.device && previous.ordinal == current.ordinal)
                    continue;
                ++partitions;
                ++transitions;
            }
            report_.partitioning.partitions = Metric::exact(partitions);
            report_.partitioning.estimated_transfers = Metric::estimated(transitions);
        }

        if (!manifest_.lowering_compatibilities.empty()) {
            const std::vector<Mln::Sdk::Backend> backends = deployment_backends(manifest_);
            if (backends.empty())
                raise(Errors::Compiler::Backend::Missing);
            for (const Mln::Sdk::Backend &backend : backends)
                (void)Lowering::apply(*module, backend, manifest_.lowering_compatibilities);
            if (mlir::failed(mlir::verify(*module)))
                raise(Errors::Compiler::Module::Invalid);
        }

        module->walk([](mlir::Operation *operation) { operation->removeAttr("myelin.backend"); });

        using StreamFavor = mlir::iree_compiler::IREE::Stream::Favor;
        StreamFavor favor = StreamFavor::MinPeakMemory;
        std::string_view favor_spelling = "min-peak-memory";
        switch (manifest_.scheduling_favor) {
        case Manifest::SchedulingFavor::MinMemory:
            favor = StreamFavor::MinPeakMemory;
            favor_spelling = "min-peak-memory";
            break;
        case Manifest::SchedulingFavor::MaxConcurrency:
            favor = StreamFavor::MaxConcurrency;
            favor_spelling = "max-concurrency";
            break;
        case Manifest::SchedulingFavor::Debug:
            favor = StreamFavor::Debug;
            favor_spelling = "debug";
            break;
        }
        auto favor_attr = mlir::iree_compiler::IREE::Stream::FavorAttr::get(context.get(), favor);
        module->getOperation()->setAttr(
            "stream.partitioning",
            mlir::iree_compiler::IREE::Stream::PartitioningConfigAttr::get(favor_attr));

        if (!manifest_.topology.empty()) {
            const std::vector<Mln::Sdk::Backend> backends = deployment_backends(manifest_);
            std::vector<mlir::iree_compiler::IREE::HAL::DeviceLinkAttr> links;
            links.reserve(manifest_.topology.size());
            for (const Manifest::Link &link : manifest_.topology) {
                const auto locate = [&](const Mln::Sdk::Backend &requested) {
                    return std::ranges::find_if(backends, [&](const Mln::Sdk::Backend &backend) {
                        return backend.device == requested.device &&
                               backend.ordinal == requested.ordinal;
                    });
                };
                const auto source = locate(link.source);
                const auto target = locate(link.target);
                if (source == backends.end() || target == backends.end())
                    raise(Errors::Compiler::Backend::Missing);
                const auto reference = [&](std::size_t index) {
                    return mlir::SymbolRefAttr::get(context.get(), std::format("device_{}", index));
                };
                links.push_back(mlir::iree_compiler::IREE::HAL::DeviceLinkAttr::get(
                    context.get(), reference(source - backends.begin()),
                    reference(target - backends.begin()), link.unified_memory,
                    link.transparent_access, mlir::DictionaryAttr::get(context.get())));
            }
            module->getOperation()->setAttr(
                "stream.topology",
                mlir::iree_compiler::IREE::HAL::DeviceTopologyAttr::get(context.get(), links));
        }

        graph.clear();
        llvm::raw_string_ostream os(graph);
        module->print(os);
        os.flush();

        const std::string printed_partitioning =
            std::format("#stream.partitioning_config<favor-{}>", favor_spelling);
        const std::string parsed_partitioning =
            std::format("#stream.partitioning_config<\"{}\">", favor_spelling);
        for (std::size_t offset = graph.find(printed_partitioning); offset != std::string::npos;
             offset = graph.find(printed_partitioning, offset + parsed_partitioning.size()))
            graph.replace(offset, printed_partitioning.size(), parsed_partitioning);
        if (const char *trace = std::getenv("MLN_TRACE_LOWERING"); trace && *trace) {
            for (std::size_t index = 0; index < deployment_backends(manifest_).size(); ++index) {
                const std::string promise = std::format("#hal.device.promise<@device_{}>", index);
                std::size_t count = 0;
                for (std::size_t offset = graph.find(promise); offset != std::string::npos;
                     offset = graph.find(promise, offset + promise.size()))
                    ++count;
                std::println(stderr, "frontend affinity device_{}: {}", index, count);
            }
        }
        finalized_ = true;
    }

    Compiler::Compiled::Graph Compiler::compile_graph() {
        finalize();

        const std::scoped_lock compilation_lock(iree_compilation_mutex);

        BackendResources backend_resources(backend_resources_identity_, backend_resources_);

        const std::vector<Mln::Sdk::Backend> backends = deployment_backends(manifest_);
        if (backends.empty())
            raise(Errors::Compiler::Backend::Missing);

        std::vector<std::string> flags;

        flags.push_back("--iree-opt-const-expr-hoisting=false");
        if (!manifest_.tuning_spec.empty())
            flags.push_back("--iree-codegen-tuning-spec-path=" + manifest_.tuning_spec);
        std::map<std::string, std::string, std::less<>> global_backend_flags;

        for (std::size_t backend_index = 0; backend_index < backends.size(); ++backend_index) {
            const Mln::Sdk::Backend &backend = backends[backend_index];

            const auto target = std::ranges::find_if(backend.flags, [](const auto &flag) {
                return flag.first == "iree-hal-target-device";
            });
            std::string external_target;
            if (target == backend.flags.end()) {
                const auto compatibility = std::ranges::find_if(
                    manifest_.backend_compatibilities, [&](const auto *candidate) {
                        return candidate && candidate->name() == backend.device;
                    });
                if (compatibility != manifest_.backend_compatibilities.end()) {
                    external_target = std::format(
                        "#hal.device.target<\"{}\", [#hal.executable.target<\"{}\", "
                        "\"{}\">]> : !hal.device",
                        backend.device, (*compatibility)->name(), (*compatibility)->format());
                }
            }
            const std::string_view target_value =
                target != backend.flags.end()
                    ? std::string_view(target->second)
                    : (!external_target.empty() ? std::string_view(external_target)
                                                : std::string_view(backend.device));
            flags.push_back(
                std::format("--iree-hal-target-device=device_{}={}", backend_index, target_value));

            const bool has_local_backend = std::ranges::any_of(backend.flags, [](const auto &flag) {
                return flag.first == "iree-hal-local-target-device-backends";
            });
            if (backend.device == "local" && !has_local_backend) {
                append_unique(flags, "--iree-hal-local-target-device-backends=llvm-cpu");
            }

            for (const auto &[key, value] : backend.flags) {
                if (key == "iree-hal-target-device")
                    continue;
                const auto [configured, inserted] = global_backend_flags.emplace(key, value);
                if (!inserted && configured->second != value)
                    throw std::runtime_error(
                        std::format("backend flag '--{}' has conflicting values '{}' and '{}'; "
                                    "use a target-scoped IREE configuration",
                                    key, configured->second, value));
                append_unique(flags, "--" + key + "=" + value);
            }
        }

        std::vector<const char *> flag_data;
        flag_data.reserve(flags.size());

        for (const std::string &flag : flags)
            flag_data.push_back(flag.c_str());

        if (const char *trace = std::getenv("MLN_TRACE_BACKEND"); trace && *trace) {
            for (const std::string &flag : flags)
                std::println(stderr, "compiler flag: {}", flag);
        }

        register_backends(manifest_.backend_compatibilities);
        register_dispatch_affinity_passes();
        register_vm_compatibility_passes();
        static Global global;
        (void)global;

        std::unique_ptr<iree_compiler_session_t, Session::Deleter> session(
            ireeCompilerSessionCreate());

        check(ireeCompilerSessionSetFlags(session.get(), static_cast<int>(flag_data.size()),
                                          flag_data.data()),
              Errors::Compiler::Iree::Flags::Set::Failed);

        const bool has_external_backend =
            std::ranges::any_of(backends, [&](const Mln::Sdk::Backend &backend) {
                return std::ranges::any_of(
                    manifest_.backend_compatibilities, [&](const auto *compatibility) {
                        return compatibility && compatibility->name() == backend.device;
                    });
            });
        if (has_external_backend) {
            MlirContext compiler_context = ireeCompilerSessionBorrowContext(session.get());
            if (!mlirContextIsNull(compiler_context))
                mlirContextEnableMultithreading(compiler_context, false);
        }

        iree_compiler_source_t *source_ptr = nullptr;

        check(ireeCompilerSourceWrapBuffer(session.get(), "myelin.mlir", graph.c_str(),
                                           graph.size() + 1, true, &source_ptr),
              Errors::Compiler::Iree::Source::Wrap::Failed);

        std::unique_ptr<iree_compiler_source_t, Source::Deleter> source(source_ptr);

        std::unique_ptr<iree_compiler_invocation_t, Invocation::Deleter> invocation(
            ireeCompilerInvocationCreate(session.get()));

        std::string diagnostics;

        ireeCompilerInvocationEnableCallbackDiagnostics(invocation.get(), 0, append_diagnostic,
                                                        &diagnostics);

        if (const char *dump_phases = std::getenv("MLN_DUMP_IREE_PHASES");
            dump_phases && *dump_phases) {
            ireeCompilerInvocationSetDumpCompilationPhasesTo(invocation.get(), dump_phases);
        }

        const char *dump_path = std::getenv("MLN_DUMP_MLIR");
        const std::string_view source_location =
            dump_path && *dump_path ? std::string_view(dump_path)
                                    : std::string_view("in memory; set MLN_DUMP_MLIR to persist");
        if (dump_path && *dump_path) {
            std::ofstream dump(dump_path);
            if (!dump)
                raise(Errors::Compiler::Mlir::Dump::Write::Failed, dump_path);
            dump << graph;
        }

        if (!ireeCompilerInvocationParseSource(invocation.get(), source.get()))
            raise(Errors::Compiler::Iree::Source::Rejected, diagnostics, source_location);

        const bool trace_compilation = [] {
            const char *value = std::getenv("MLN_TRACE_COMPILATION");
            return value && *value;
        }();
        auto phase_started = std::chrono::steady_clock::now();
        const auto report_phase = [&](std::string_view name) {
            if (!trace_compilation)
                return;
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - phase_started);
            std::println(stderr, "compiler phase {}: {} ms", name, elapsed.count());
            phase_started = now;
        };

        ireeCompilerInvocationSetCompileToPhase(invocation.get(), "dispatch-creation");
        if (!ireeCompilerInvocationPipeline(invocation.get(), IREE_COMPILER_PIPELINE_STD))
            raise(Errors::Compiler::Iree::Pipeline::Failed, diagnostics, source_location);
        report_phase("input-to-dispatch-creation");

        if (!ireeCompilerInvocationRunPassPipeline(invocation.get(),
                                                   "syn-propagate-dispatch-affinity")) {
            raise(Errors::Compiler::Iree::Pipeline::Failed, diagnostics, source_location);
        }
        report_phase("dispatch-affinity");

        ireeCompilerInvocationSetCompileFromPhase(invocation.get(), "dispatch-creation");
        ireeCompilerInvocationSetCompileToPhase(invocation.get(), "hal");
        if (!ireeCompilerInvocationPipeline(invocation.get(), IREE_COMPILER_PIPELINE_STD))
            raise(Errors::Compiler::Iree::Pipeline::Failed, diagnostics, source_location);
        report_phase("flow-to-hal");

        if (!ireeCompilerInvocationRunPassPipeline(invocation.get(),
                                                   "syn-normalize-cf-switch-index")) {
            raise(Errors::Compiler::Iree::Pipeline::Failed, diagnostics, source_location);
        }
        report_phase("hal-normalization");

        if (const char *hal_dump = std::getenv("MLN_DUMP_HAL"); hal_dump && *hal_dump) {
            iree_compiler_output_t *dump_ptr = nullptr;
            check(ireeCompilerOutputOpenFile(hal_dump, &dump_ptr),
                  Errors::Compiler::Iree::Output::Open::Failed);
            std::unique_ptr<iree_compiler_output_t, Output::Deleter> dump(dump_ptr);
            check(ireeCompilerInvocationOutputIR(invocation.get(), dump.get()),
                  Errors::Compiler::Iree::Output::Emit::Failed);
            ireeCompilerOutputKeep(dump.get());
        }

        ireeCompilerInvocationSetCompileFromPhase(invocation.get(), "hal");
        ireeCompilerInvocationSetCompileToPhase(invocation.get(), "end");
        if (!ireeCompilerInvocationPipeline(invocation.get(), IREE_COMPILER_PIPELINE_STD))
            raise(Errors::Compiler::Iree::Pipeline::Failed, diagnostics, source_location);
        report_phase("hal-to-vm");

        iree_compiler_output_t *output_ptr = nullptr;

        check(ireeCompilerOutputOpenMembuffer(&output_ptr),
              Errors::Compiler::Iree::Output::Open::Failed);

        std::unique_ptr<iree_compiler_output_t, Output::Deleter> output(output_ptr);

        if (iree_compiler_error_t *iree_error =
                ireeCompilerInvocationOutputVMBytecode(invocation.get(), output.get())) {
            std::string message = ireeCompilerErrorGetMessage(iree_error);
            ireeCompilerErrorDestroy(iree_error);
            if (!diagnostics.empty())
                message += "; diagnostics: " + diagnostics;
            raise(Errors::Compiler::Iree::Output::Emit::Failed, message);
        }

        void *contents = nullptr;
        uint64_t size = 0;

        check(ireeCompilerOutputMapMemory(output.get(), &contents, &size),
              Errors::Compiler::Iree::Output::Map::Failed);

        Compiled::Graph result;

        result.vmfb.assign(static_cast<const uint8_t *>(contents),
                           static_cast<const uint8_t *>(contents) + size);

        result.devices.reserve(backends.size());

        for (const Mln::Sdk::Backend &backend : backends)
            result.devices.push_back(runtime_device_uri(backend));

        result.deployment.reserve(entrypoint_backends_.size());

        for (const auto &[path, assigned] : entrypoint_backends_) {
            const auto found = std::find_if(
                backends.begin(), backends.end(), [&](const Mln::Sdk::Backend &backend) {
                    return backend.device == assigned.device && backend.ordinal == assigned.ordinal;
                });

            if (found == backends.end()) {
                raise(Errors::Compiler::Entrypoint::Backend::Missing);
            }

            result.deployment.emplace_back(path, static_cast<uint32_t>(found - backends.begin()));
        }

        return result;
    }

    void Compiler::write_vmfb(const std::string &vmfb_path, const std::string &devices_path,
                              const std::string &weights_path) {
        const Compiled::Graph compiled = compile_graph();

        Mln::Weight::write_binary_file(vmfb_path, compiled.vmfb);

        std::ofstream devices(devices_path);

        if (!devices) {
            raise(Errors::Compiler::Devices::File::Create::Failed, devices_path);
        }

        for (const std::string &device : compiled.devices) {
            devices << device << '\n';
        }

        if (!weights_path.empty()) {
            const std::vector<uint8_t> encoded =
                Mln::Weight::encode_weight_index(Mln::Weight::Index::Kind::Requirements, weights_);

            Mln::Weight::write_binary_file(weights_path, encoded);
        }
    }

    void Compiler::write_graph_cache(const std::string &output_path,
                                     const std::string &requirements_path) {
        Compiled::Graph compiled = compile_graph();

        std::string devices;
        for (const std::string &device : compiled.devices) {
            devices.append(device);
            devices += '\n';
        }
        std::string deployment;
        for (const auto &[path, device] : compiled.deployment) {
            deployment.append(std::format("{}\t", device));
            deployment.append(path);
            deployment += '\n';
        }
        std::string typed;
        for (const std::string &entrypoint : typed_entrypoints_)
            typed.append(entrypoint);

        Mln::Package::Builder builder;
        builder.add(Mln::Package::Section::Type::Devices, "devices",
                    std::vector<uint8_t>(devices.begin(), devices.end()));
        builder.add(Mln::Package::Section::Type::Deployment, "entrypoints",
                    std::vector<uint8_t>(deployment.begin(), deployment.end()));
        if (!typed.empty())
            builder.add(Mln::Package::Section::Type::Entrypoints, "typed",
                        std::vector<uint8_t>(typed.begin(), typed.end()));
        builder.add(Mln::Package::Section::Type::Vmfb,
                    vm_module_name_.empty() ? "model" : vm_module_name_, std::move(compiled.vmfb));
        std::vector<uint8_t> requirements =
            Mln::Weight::encode_weight_index(Mln::Weight::Index::Kind::Requirements, weights_);
        if (!requirements_path.empty())
            Mln::Weight::write_binary_file(requirements_path, requirements);
        builder.add(Mln::Package::Section::Type::WeightIndex, "requirements",
                    std::move(requirements));
        builder.write(output_path);

        module = mlir::OwningOpRef<mlir::ModuleOp>{};
        context.reset();
        graph.clear();
        graph.shrink_to_fit();
    }

    void Compiler::write_requirements(const std::string &output_path) {
        finalize();
        Mln::Weight::write_binary_file(
            output_path,
            Mln::Weight::encode_weight_index(Mln::Weight::Index::Kind::Requirements, weights_));
    }

    void Compiler::write_parameters_cache(const std::vector<std::string> &weight_paths,
                                          const std::vector<Mln::Weight::Descriptor> &requirements,
                                          const std::string &output_path,
                                          std::size_t conversion_jobs,
                                          uint64_t conversion_memory_bytes) {
        const Weight::Quant::Lookup lookup = [this](std::string_view name) -> Mln::Sdk::Codec & {
            return parameter_codec(manifest_, std::string(name));
        };
        const Weight::Scheme::Lookup schemes =
            [this](std::string_view name) -> const Mln::Sdk::Codec::Scheme & {
            const auto match =
                scheme_for(manifest_, std::string(name), Mln::Sdk::Codec::Scope::Constant,
                           Mln::Sdk::Codec::Lifetime::Immutable);
            if (!match)
                raise(Errors::Compiler::Weight::Scheme::Missing, name);
            return *match.scheme;
        };
        const Weight::Residency::Lookup residency = [this](std::string_view name) {
            const Police police = police_for(manifest_, std::string(name));
            if (police.loading)
                return *police.loading == LoadingMode::Streamed ? Mln::Weight::Residency::Volatile
                                                                : Mln::Weight::Residency::Retained;
            const auto match =
                scheme_for(manifest_, std::string(name), Mln::Sdk::Codec::Scope::Constant,
                           Mln::Sdk::Codec::Lifetime::Immutable);
            return match ? match.scheme->residency : Mln::Weight::Residency::Retained;
        };

        std::vector<Syn::Compiler::Weight::Conversion::Result> archives;
        std::vector<Mln::Weight::Descriptor> all_descriptors;
        if (requirements.empty()) {
            archives.push_back(Syn::Compiler::Weight::Conversion::empty());
        } else {
            struct Work {
                    std::string path;
                    std::vector<Mln::Weight::Descriptor> descriptors;
                    std::vector<Mln::Weight::Descriptor> selected;
                    uint64_t temporary_bytes = 1;
            };
            std::vector<Work> work;
            for (const std::string &path : weight_paths) {
                const auto descriptors = Syn::Compiler::Weight::read(path);
                std::unordered_set<std::string> names;
                for (const auto &descriptor : descriptors)
                    names.insert(descriptor.name);
                std::vector<Mln::Weight::Descriptor> selected;
                for (const auto &requirement : requirements) {
                    const std::string &source =
                        requirement.storage.empty() ? requirement.name : requirement.storage;
                    if (names.contains(source))
                        selected.push_back(requirement);
                }
                if (selected.empty())
                    continue;
                std::unordered_set<std::string> selected_sources;
                selected_sources.reserve(selected.size());
                for (const auto &requirement : selected)
                    selected_sources.insert(requirement.storage.empty() ? requirement.name
                                                                        : requirement.storage);
                uint64_t largest_temporary = 1;
                for (const auto &descriptor : descriptors) {
                    if (!selected_sources.contains(descriptor.name))
                        continue;
                    uint64_t elements = 1;
                    bool overflow = false;
                    for (const int64_t dimension : descriptor.shape) {
                        if (dimension <= 0 || elements > std::numeric_limits<uint64_t>::max() /
                                                             static_cast<uint64_t>(dimension)) {
                            overflow = true;
                            break;
                        }
                        elements *= static_cast<uint64_t>(dimension);
                    }
                    const uint64_t decoded =
                        overflow || elements > std::numeric_limits<uint64_t>::max() / 8
                            ? std::numeric_limits<uint64_t>::max()
                            : elements * 8;
                    const uint64_t estimate =
                        decoded > std::numeric_limits<uint64_t>::max() - descriptor.byte_length
                            ? std::numeric_limits<uint64_t>::max()
                            : decoded + descriptor.byte_length;
                    largest_temporary = std::max(largest_temporary, estimate);
                }
                work.push_back(
                    {path, std::move(descriptors), std::move(selected), largest_temporary});
            }

            conversion_jobs = std::clamp<std::size_t>(conversion_jobs, 1, 8);
            conversion_memory_bytes = std::max<uint64_t>(conversion_memory_bytes, 1);
            for (std::size_t begin = 0; begin < work.size();) {
                std::size_t end = begin;
                uint64_t batch_bytes = 0;
                while (end < work.size() && end - begin < conversion_jobs) {
                    const uint64_t estimate = work[end].temporary_bytes;
                    if (end != begin &&
                        (estimate >
                         conversion_memory_bytes - std::min(batch_bytes, conversion_memory_bytes)))
                        break;
                    batch_bytes = estimate > std::numeric_limits<uint64_t>::max() - batch_bytes
                                      ? std::numeric_limits<uint64_t>::max()
                                      : batch_bytes + estimate;
                    ++end;
                }

                std::vector<std::future<Syn::Compiler::Weight::Conversion::Result>> running;
                running.reserve(end - begin);
                for (std::size_t index = begin; index < end; ++index) {
                    Work *item = &work[index];
                    running.push_back(std::async(std::launch::async, [&, item] {
                        return Syn::Compiler::Weight::Conversion::convert(
                            item->path, item->descriptors, item->selected, lookup, schemes,
                            residency);
                    }));
                }
                for (auto &task : running) {
                    auto converted = task.get();
                    all_descriptors.insert(all_descriptors.end(), converted.descriptors.begin(),
                                           converted.descriptors.end());
                    archives.push_back(std::move(converted));
                }
                begin = end;
            }
        }
        Mln::Weight::validate_weights(requirements, all_descriptors);

        Mln::Package::Builder builder;
        for (std::size_t index = 0; index < archives.size(); ++index)
            builder.add(Mln::Package::Section::Type::Parameters,
                        archives.size() == 1 ? "model" : std::format("model.{}", index),
                        std::move(archives[index].data));
        builder.write(output_path);
    }

    void Compiler::write_package(const std::vector<std::string> &weight_paths,
                                 const std::string &output_path,
                                 const Mln::Artifact::Plan &artifacts, bool deployable) {
        validate_artifact_plan(artifacts, deployable);

        finalize();

        struct Encoded {
                Mln::Artifact::Runtime runtime;
                std::string name;
                std::vector<uint8_t> bytes;
        };
        const std::vector<uint8_t> requirements_bytes =
            Mln::Weight::encode_weight_index(Mln::Weight::Index::Kind::Requirements, weights_);

        const auto requirements = Mln::Weight::decode_weight_index(
            requirements_bytes, Mln::Weight::Index::Kind::Requirements);

        const Weight::Quant::Lookup lookup = [this](std::string_view name) -> Mln::Sdk::Codec & {
            return parameter_codec(manifest_, std::string(name));
        };

        const Weight::Scheme::Lookup scheme_lookup =
            [this](std::string_view name) -> const Mln::Sdk::Codec::Scheme & {
            const auto match =
                scheme_for(manifest_, std::string(name), Mln::Sdk::Codec::Scope::Constant,
                           Mln::Sdk::Codec::Lifetime::Immutable);

            if (!match) {
                raise(Errors::Compiler::Weight::Scheme::Missing, name);
            }

            return *match.scheme;
        };
        const Weight::Residency::Lookup residency_lookup = [this](std::string_view name) {
            const Police police = police_for(manifest_, std::string(name));
            if (police.loading)
                return *police.loading == LoadingMode::Streamed ? Mln::Weight::Residency::Volatile
                                                                : Mln::Weight::Residency::Retained;
            const auto match =
                scheme_for(manifest_, std::string(name), Mln::Sdk::Codec::Scope::Constant,
                           Mln::Sdk::Codec::Lifetime::Immutable);
            return match ? match.scheme->residency : Mln::Weight::Residency::Retained;
        };

        std::vector<Syn::Compiler::Weight::Conversion::Result> converted_archives;
        std::vector<Mln::Weight::Descriptor> converted_descriptors;
        Syn::Compiler::Weight::Conversion::Statistics conversion_statistics;
        if (weights_.empty()) {
            converted_archives.push_back(Syn::Compiler::Weight::Conversion::empty());
        } else {
            for (const std::string &weight_path : weight_paths) {
                const auto descriptors = Syn::Compiler::Weight::read(weight_path);
                std::unordered_set<std::string> source_names;
                for (const auto &descriptor : descriptors)
                    source_names.insert(descriptor.name);

                std::vector<Mln::Weight::Descriptor> selected;
                for (const auto &requirement : requirements) {
                    const std::string &source =
                        requirement.storage.empty() ? requirement.name : requirement.storage;
                    if (source_names.contains(source))
                        selected.push_back(requirement);
                }
                if (selected.empty())
                    continue;

                Syn::Compiler::Weight::Conversion::Result converted =
                    Syn::Compiler::Weight::Conversion::convert(weight_path, descriptors, selected,
                                                               lookup, scheme_lookup,
                                                               residency_lookup);
                converted_descriptors.insert(converted_descriptors.end(),
                                             converted.descriptors.begin(),
                                             converted.descriptors.end());
                conversion_statistics.source_bytes += converted.statistics.source_bytes;
                conversion_statistics.output_bytes += converted.statistics.output_bytes;
                conversion_statistics.source_parameters += converted.statistics.source_parameters;
                conversion_statistics.output_parameters += converted.statistics.output_parameters;
                conversion_statistics.converted_parameters +=
                    converted.statistics.converted_parameters;
                conversion_statistics.passthrough_parameters +=
                    converted.statistics.passthrough_parameters;
                converted_archives.push_back(std::move(converted));
            }
        }

        Mln::Weight::validate_weights(requirements, converted_descriptors);

        backend_resources_.clear();
        backend_resources_.reserve(converted_descriptors.size());
        for (const auto &archive : converted_archives) {
            for (const auto &resource : archive.resources) {
                const auto descriptor =
                    std::ranges::find_if(converted_descriptors, [&](const auto &candidate) {
                        return candidate.name == resource.name;
                    });
                if (descriptor == converted_descriptors.end())
                    continue;
                if (resource.offset > archive.data.size() ||
                    resource.length > archive.data.size() - resource.offset) {
                    raise(Errors::Weight::Storage::Outside::File, resource.name);
                }
                backend_resources_.push_back(Mln::Sdk::Backend::Resource{
                    .name = descriptor->name,
                    .format = descriptor->quant,
                    .shape = descriptor->shape,
                    .data = std::as_bytes(
                        std::span(archive.data.data() + static_cast<std::size_t>(resource.offset),
                                  static_cast<std::size_t>(resource.length))),
                });
            }
        }

        Compiled::Graph compiled = compile_graph();

        backend_resources_.clear();
        module = mlir::OwningOpRef<mlir::ModuleOp>{};
        context.reset();
        graph.clear();
        graph.shrink_to_fit();

        std::vector<Encoded> runtime_modules;
        runtime_modules.reserve(artifacts.modules.size());
        uint64_t executable_bytes = compiled.vmfb.size();
        for (const Mln::Artifact::Module &artifact_module : artifacts.modules) {
            std::vector<uint8_t> bytes =
                Mln::Weight::read_binary_file(artifact_module.artifact.path);
            executable_bytes += bytes.size();
            runtime_modules.push_back(
                {artifact_module.runtime, artifact_module.artifact.name, std::move(bytes)});
        }

        std::string devices_text;
        for (const std::string &device : compiled.devices) {
            devices_text.append(device);
            devices_text += '\n';
        }

        std::string deployment_text;
        for (const auto &[path, device] : compiled.deployment) {
            if (path.find_first_of("\t\r\n") != std::string::npos)
                raise(Errors::Compiler::Entrypoint::Path::Invalid);
            deployment_text.append(std::format("{}\t", device));
            deployment_text.append(path);
            deployment_text += '\n';
        }

        uint64_t parameter_bytes = 0;

        for (const auto &descriptor : converted_descriptors) {
            parameter_bytes += descriptor.byte_length;
        }

        report_.storage.parameter_bytes = Metric::exact(parameter_bytes);

        report_.memory.persistent_bytes = Metric::unknown();

        report_.quantization.bytes_after = Metric::exact(parameter_bytes);

        report_.quantization.bytes_before = Metric::exact(conversion_statistics.source_bytes);

        if (report_.quantization.bytes_before.value && parameter_bytes != 0) {
            report_.quantization.compression_ratio = Metric::Ratio{
                static_cast<double>(*report_.quantization.bytes_before.value) /
                    static_cast<double>(parameter_bytes),
                Metric::Quality::Exact,
            };
        }

        report_.packaging.source_bytes = Metric::exact(conversion_statistics.source_bytes);

        report_.packaging.output_bytes = Metric::exact(conversion_statistics.output_bytes);

        report_.packaging.source_parameters =
            Metric::exact(conversion_statistics.source_parameters);

        report_.packaging.output_parameters =
            Metric::exact(conversion_statistics.output_parameters);

        report_.packaging.passthrough_parameters =
            Metric::exact(conversion_statistics.passthrough_parameters);

        report_.packaging.converted_parameters =
            Metric::exact(conversion_statistics.converted_parameters);

        report_.storage.executable_bytes = Metric::exact(executable_bytes);

        Mln::Package::Builder builder;

        builder.add(Mln::Package::Section::Type::Devices, "devices",
                    std::vector<uint8_t>(devices_text.begin(), devices_text.end()));

        builder.add(Mln::Package::Section::Type::Deployment, "entrypoints",
                    std::vector<uint8_t>(deployment_text.begin(), deployment_text.end()));

        if (!typed_entrypoints_.empty()) {
            std::string entrypoints_text;
            for (const std::string &entrypoint : typed_entrypoints_)
                entrypoints_text.append(entrypoint);
            builder.add(Mln::Package::Section::Type::Entrypoints, "typed",
                        std::vector<uint8_t>(entrypoints_text.begin(), entrypoints_text.end()));
        }

        builder.add(Mln::Package::Section::Type::Vmfb, "model", std::move(compiled.vmfb));

        for (std::size_t index = 0; index < converted_archives.size(); ++index)
            builder.add(Mln::Package::Section::Type::Parameters,
                        converted_archives.size() == 1 ? "model" : std::format("model.{}", index),
                        std::move(converted_archives[index].data));

        builder.add(Mln::Package::Section::Type::WeightIndex, "requirements", requirements_bytes);

        for (Encoded &runtime_module : runtime_modules) {
            const Mln::Package::Section::Type type =
                runtime_module.runtime == Mln::Artifact::Runtime::Wasm
                    ? Mln::Package::Section::Type::Wasm
                : runtime_module.runtime == Mln::Artifact::Runtime::Driver
                    ? Mln::Package::Section::Type::Driver
                    : Mln::Package::Section::Type::Native;
            builder.add(type, std::move(runtime_module.name), std::move(runtime_module.bytes));
        }

        for (const Mln::Artifact::File &asset : artifacts.assets) {
            builder.add(Mln::Package::Section::Type::Asset, asset.name,
                        Mln::Weight::read_binary_file(asset.path));
        }

        std::string application_entrypoints;
        for (const Mln::Artifact::Entrypoint &entrypoint : artifacts.entrypoints) {
            application_entrypoints.append(entrypoint.name);
            application_entrypoints += '\t';
            application_entrypoints.append(
                entrypoint.runtime == Mln::Artifact::Runtime::Graph    ? "graph"
                : entrypoint.runtime == Mln::Artifact::Runtime::Native ? "native"
                                                                       : "wasm");
            application_entrypoints += '\t';
            application_entrypoints.append(entrypoint.module);
            application_entrypoints += '\t';
            application_entrypoints.append(entrypoint.export_name);
            application_entrypoints += '\n';
        }
        if (!application_entrypoints.empty()) {
            builder.add(Mln::Package::Section::Type::ApplicationEntrypoints, "public",
                        std::vector<uint8_t>(application_entrypoints.begin(),
                                             application_entrypoints.end()));
        }

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

        const bool native = std::ranges::any_of(artifacts.modules, [](const auto &artifact_module) {
            return artifact_module.runtime == Mln::Artifact::Runtime::Native;
        });
        const bool driver = std::ranges::any_of(artifacts.modules, [](const auto &artifact_module) {
            return artifact_module.runtime == Mln::Artifact::Runtime::Driver;
        });
        builder.add(Mln::Package::Section::Type::Requirements, "runtime",
                    Mln::encode_requirements({
                        .runtime_abi = 1,
                        .native_abi = native ? Mln::Sdk::Native::ABI_VERSION : 0,
                        .driver_abi = driver ? Mln::Sdk::Driver::AbiVersion : 0,
                        .platform = (native || driver) ? Mln::current_platform() : std::string{},
                    }));

        builder.write(output_path);

        const uint64_t artifact_bytes = std::filesystem::file_size(output_path);

        report_.storage.total_artifact_bytes = Metric::exact(artifact_bytes);

        report_.storage.metadata_bytes =
            Metric::exact(artifact_bytes - parameter_bytes - executable_bytes);
    }
}
