#include "mln/sdk/functional.hpp"
#include "mln/sdk/module.hpp"
#include "syn/compiler.hpp"

#include <cstdlib>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
    std::string read(const std::filesystem::path &path) {
        std::ifstream stream(path);
        if (!stream)
            throw std::runtime_error("cannot read " + path.string());
        return {std::istreambuf_iterator<char>(stream), {}};
    }

    std::size_t occurrences(std::string_view text, std::string_view needle) {
        std::size_t count = 0;
        for (std::size_t offset = text.find(needle); offset != std::string_view::npos;
             offset = text.find(needle, offset + needle.size())) {
            ++count;
        }
        return count;
    }

    class External final : public Mln::Sdk::Backend::Compatibility {
        public:
            std::string_view name() const noexcept override { return "external-test"; }
            std::string_view format() const noexcept override { return "external-test-v1"; }

            std::expected<std::vector<std::byte>, std::string>
            serialize(const Mln::Sdk::Backend::Compilation &) const override {
                return std::vector<std::byte>{std::byte{'M'}, std::byte{'L'}, std::byte{'N'}};
            }
    };

    class F32 final : public Mln::Sdk::Codec {
        public:
            F32() { name = "F32"; }

            Mln::Sdk::Tensor decode(Mln::Sdk::Operations &operations,
                                    const Mln::Sdk::Weight &weight, Mln::DType dtype) override {
                return operations.insert<Mln::Sdk::Functional::Const>(weight, false, dtype);
            }
    };

    class Add final : public Mln::Sdk::Module {
        public:
            explicit Add(std::string path) { name = std::move(path); }

            Mln::Sdk::Tensor forward(Mln::Sdk::Tensor input) {
                return operations.insert<Mln::Sdk::Functional::Add>(input, input);
            }
    };

    class Chain final : public Mln::Sdk::Module {
        public:
            Chain() {
                name = "chain";
                modules.insert<Add>("cpu", "cpu");
                modules.insert<Add>("gpu", "gpu");
                modules.insert<Add>("npu", "npu");
            }

            Mln::Sdk::Tensor forward(Mln::Sdk::Tensor input) {
                input = modules.forward<Add>("cpu", std::move(input));
                input = modules.forward<Add>("gpu", std::move(input));
                return modules.forward<Add>("npu", std::move(input));
            }
    };

}

int main(int argc, char **argv) {
    if (argc != 2)
        throw std::runtime_error("expected output directory");

    const std::filesystem::path output = argv[1];
    std::filesystem::create_directories(output);
    const std::filesystem::path phases = output / "phases";
    std::filesystem::create_directories(phases);
    setenv("MLN_DUMP_IREE_PHASES", phases.c_str(), 1);

    Mln::Sdk::Codec::Format f32{F32{}};
    External external;
    Syn::Compiler::Manifest manifest{
        .policies =
            {
                {"model",
                 {.schemes = {Mln::Sdk::Codec::Scheme{
                      .scope = Mln::Sdk::Codec::Scope::Constant,
                      .lifetime = Mln::Sdk::Codec::Lifetime::Immutable,
                      .update = Mln::Sdk::Codec::Update::None,
                      .format = f32,
                  }},
                  .dtype = Mln::DType::Float32,
                  .backend =
                      Mln::Sdk::Backend{
                          .device = "local",
                          .flags = {{"iree-llvmcpu-target-cpu", "generic"}},
                      }}},
                {"cpu",
                 {.dtype = Mln::DType::Float32,
                  .backend =
                      Mln::Sdk::Backend{
                          .device = "local",
                          .flags = {{"iree-llvmcpu-target-cpu", "generic"}},
                      }}},
                {"gpu",
                 {.dtype = Mln::DType::Float32, .backend = Mln::Sdk::Backend{.device = "vulkan"}}},
                {"npu",
                 {.dtype = Mln::DType::Float32,
                  .backend = Mln::Sdk::Backend{.device = "external-test"}}},
                {"chain",
                 {.dtype = Mln::DType::Float32,
                  .backend =
                      Mln::Sdk::Backend{
                          .device = "local",
                          .flags = {{"iree-llvmcpu-target-cpu", "generic"}},
                      }}},
                {"chain.cpu",
                 {.dtype = Mln::DType::Float32,
                  .backend =
                      Mln::Sdk::Backend{
                          .device = "local",
                          .flags = {{"iree-llvmcpu-target-cpu", "generic"}},
                      }}},
                {"chain.gpu",
                 {.dtype = Mln::DType::Float32, .backend = Mln::Sdk::Backend{.device = "vulkan"}}},
                {"chain.npu",
                 {.dtype = Mln::DType::Float32,
                  .backend = Mln::Sdk::Backend{.device = "external-test"}}},
            },
        .formats = {f32},
        .backend_compatibilities = {&external},
        .topology =
            {
                {.source = {.device = "local"},
                 .target = {.device = "vulkan"},
                 .transparent_access = true},
                {.source = {.device = "vulkan"},
                 .target = {.device = "local"},
                 .transparent_access = true},
                {.source = {.device = "vulkan"},
                 .target = {.device = "external-test"},
                 .transparent_access = true},
                {.source = {.device = "external-test"},
                 .target = {.device = "vulkan"},
                 .transparent_access = true},
                {.source = {.device = "local"},
                 .target = {.device = "external-test"},
                 .transparent_access = true},
                {.source = {.device = "external-test"},
                 .target = {.device = "local"},
                 .transparent_access = true},
            },
    };
    Syn::Compiler::Compiler compiler(manifest, "multi_device");
    Add cpu("cpu");
    Add gpu("gpu");
    Add npu("npu");
    Mln::Sdk::Tensor cpu_input({4}, Mln::DType::Float32);
    Mln::Sdk::Tensor gpu_input({4}, Mln::DType::Float32);
    Mln::Sdk::Tensor npu_input({4}, Mln::DType::Float32);
    compiler.compile(cpu, cpu_input);
    compiler.compile(gpu, gpu_input);
    compiler.compile(npu, npu_input);
    Chain chain;
    Mln::Sdk::Tensor chain_input({4}, Mln::DType::Float32);
    compiler.compile(chain, chain_input);

    const std::filesystem::path vmfb = output / "multi_device.vmfb";
    const std::filesystem::path devices = output / "multi_device.devices";
    compiler.write_vmfb(vmfb.string(), devices.string());
    unsetenv("MLN_DUMP_IREE_PHASES");

    if (!std::filesystem::exists(vmfb) || std::filesystem::file_size(vmfb) == 0)
        throw std::runtime_error("multi-device VMFB was not generated");
    std::ifstream device_stream(devices);
    const std::string device_text{std::istreambuf_iterator<char>(device_stream), {}};
    if (device_text != "local-task://\nvulkan://\nexternal-test://\n")
        throw std::runtime_error("unexpected device order: " + device_text);
    if (!compiler.graph.contains("#hal.device.promise<@device_0>") ||
        !compiler.graph.contains("#hal.device.promise<@device_1>"))
        throw std::runtime_error("IREE device affinities are absent from the source module");
    if (!compiler.graph.contains("#hal.device.promise<@device_2>"))
        throw std::runtime_error("external backend affinity is absent from the source module");

    const std::string stream_ir = read(phases / "myelin.7.stream.mlir");
    for (const std::string_view affinity : {"@device_0", "@device_1", "@device_2"}) {
        const std::string dispatch =
            "stream.cmd.execute on(#hal.device.affinity<" + std::string(affinity) + ">)";
        if (!stream_ir.contains(dispatch))
            throw std::runtime_error("missing cross-device dispatch for " + std::string(affinity));
    }
    if (!stream_ir.contains("stream.topology = #hal.device.topology"))
        throw std::runtime_error("declared device topology was not propagated to IREE");
    if (occurrences(stream_ir, "stream.cmd.copy") != 0 ||
        occurrences(stream_ir, "stream.cmd.flush") != 0)
        throw std::runtime_error("IREE did not elide transparent device transfers");
}
