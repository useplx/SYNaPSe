#include "mln/sdk/functional.hpp"
#include "mln/sdk/module.hpp"
#include "syn/compiler.hpp"

#include <barrier>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
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
}

int main(int argc, char **argv) {
    if (argc != 2)
        throw std::runtime_error("expected output directory");
    const std::filesystem::path output = argv[1];
    std::filesystem::create_directories(output);
    const std::filesystem::path tuning_spec = output / "noop_tuning.mlir";
    {
        std::ofstream stream(tuning_spec);
        stream << R"mlir(module @user_spec attributes {
    transform.with_named_sequence,
    iree_codegen.tuning_spec_with_default_entrypoint
} {
    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly})
            -> (!transform.any_op) {
        transform.yield %arg : !transform.any_op
    }
    transform.named_sequence @apply_op_config(
            %op: !transform.any_op {transform.readonly}) {
        transform.yield
    }
    transform.named_sequence @__kernel_config(
            %arg0: !transform.any_op {transform.consumed})
            -> (!transform.any_op)
            attributes {iree_codegen.tuning_spec_entrypoint} {
        %res = transform.foreach_match in %arg0
            @match -> @apply_op_config
            : (!transform.any_op) -> (!transform.any_op)
        transform.yield %res : !transform.any_op
    }
})mlir";
        if (!stream)
            throw std::runtime_error("could not write tuning spec");
    }

    Mln::Sdk::Codec::Format f32{F32{}};
    const Syn::Compiler::Manifest manifest{
        .tuning_spec = tuning_spec.string(),
        .policies = {{"model",
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
                           }}}},
        .formats = {f32},
    };
    std::barrier ready(2);
    std::exception_ptr failures[2];
    std::jthread workers[2];
    for (std::size_t index = 0; index < 2; ++index) {
        workers[index] = std::jthread([&, index] {
            bool synchronized = false;
            try {
                Syn::Compiler::Compiler compiler(manifest, "parallel_" + std::to_string(index));
                Add model("model");
                Mln::Sdk::Tensor input({16}, Mln::DType::Float32);
                compiler.compile(model, input);
                ready.arrive_and_wait();
                synchronized = true;
                compiler.write_vmfb((output / (std::to_string(index) + ".vmfb")).string(),
                                    (output / (std::to_string(index) + ".devices")).string());
            } catch (...) {
                if (!synchronized)
                    ready.arrive_and_drop();
                failures[index] = std::current_exception();
            }
        });
    }
    for (std::jthread &worker : workers)
        worker.join();
    for (const std::exception_ptr &failure : failures)
        if (failure)
            std::rethrow_exception(failure);
    for (std::size_t index = 0; index < 2; ++index) {
        const std::filesystem::path vmfb = output / (std::to_string(index) + ".vmfb");
        if (!std::filesystem::exists(vmfb) || std::filesystem::file_size(vmfb) == 0)
            throw std::runtime_error("parallel compiler did not produce a VMFB");
    }
}
