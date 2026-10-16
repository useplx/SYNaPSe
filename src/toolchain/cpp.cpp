#include "syn/toolchain/abi.hpp"
#include "syn/toolchain/driver.hpp"
#include "syn/toolchain/sdk.hpp"

#include "syn/error.hpp"
#include "syn/system/process.hpp"

#include <filesystem>

namespace Syn::Toolchain {
    namespace {
        constexpr const char *SDK_CORE[] = {
            "src/sdk/buffer.cpp",     "src/sdk/modules.cpp",    "src/sdk/orchestrations.cpp",
            "src/sdk/operations.cpp", "src/sdk/parameters.cpp", "src/sdk/module.cpp",
            "src/sdk/tensor.cpp",
        };

        constexpr const char *SDK_WASM[] = {"src/sdk/session.cpp"};
        constexpr const char *SDK_HOST[] = {"src/sdk/audio/dsp.cpp"};
        constexpr const char *SDK_NN[] = {
            "src/sdk/nn/linear.cpp",
            "src/sdk/nn/conv2d.cpp",
            "src/sdk/nn/silu.cpp",
            "src/sdk/nn/gelu.cpp",
        };

        class Cpp final : public Driver {
            public:
                explicit Cpp(bool c) : c_(c) {}

                std::string_view language() const override { return c_ ? "c" : "cpp"; }

                void probe(const Config::Toolchain &toolchain) const override {
                    if (toolchain.compiler.empty())
                        Errors::raise(Error::Toolchain::Compiler::Missing, language());

                    const System::Process::Result result =
                        System::run({.executable = toolchain.compiler.string(),
                                     .arguments = {"--version"},
                                     .working_directory = {},
                                     .environment = {}});
                    if (result.exit_code != 0)
                        Errors::raise(Error::Compiler::Unresponsive, toolchain.compiler,
                                      "--version");
                }

                bool supports(Runtime runtime, Output output) const override {
                    if (runtime == Runtime::Wasm)
                        return output == Output::WasmModule;
                    return output == Output::SharedLibrary || output == Output::Executable;
                }

                void build(const Config::Toolchain &toolchain,
                           const Request &request) const override {
                    if (!supports(request.runtime, request.output))
                        Errors::raise(Error::Output::Unsupported, language(),
                                      runtime_name(request.runtime));

                    const Abi::Requirements abi =
                        Abi::requirements(request.runtime, request.output);

                    std::vector<std::string> arguments{c_ ? "-std=gnu17" : "-std=c++23"};
                    arguments.push_back(request.optimize ? "-O2" : "-O0");
                    if (!request.optimize)
                        arguments.push_back("-g");

                    if (abi.position_independent)
                        arguments.push_back("-fPIC");
                    if (abi.disable_exceptions)
                        arguments.push_back("-fno-exceptions");
                    if (abi.reactor_model)
                        arguments.push_back("-mexec-model=reactor");

                    if (request.output == Output::SharedLibrary)
                        arguments.push_back("-shared");

                    if (!toolchain.triple.empty() && request.runtime != Runtime::Wasm)
                        arguments.push_back("--target=" + toolchain.triple);

                    for (const std::string &define : abi.defines)
                        arguments.push_back("-D" + define);
                    if (request.runtime == Runtime::Iree)
                        arguments.push_back("-DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1");
                    for (const std::string &define : request.defines)
                        arguments.push_back("-D" + define);

                    for (const std::filesystem::path &include : request.includes) {
                        arguments.push_back("-I");
                        arguments.push_back(include.string());
                    }
                    arguments.push_back("-I");
                    arguments.push_back((Myelin::root() / "include").string());
                    arguments.push_back("-I");
                    arguments.push_back(Sdk::compiler().string());
                    if (request.runtime == Runtime::Iree)
                        for (const auto &include : Sdk::ir()) {
                            arguments.push_back("-isystem");
                            arguments.push_back(include.string());
                        }
                    for (const std::string &flag : toolchain.flags)
                        arguments.push_back(flag);
                    for (const std::string &flag : request.flags)
                        arguments.push_back(flag);

                    for (const std::filesystem::path &source : request.sources)
                        arguments.push_back(source.string());
                    if (abi.mln_sdk_core) {
                        for (const char *source : SDK_CORE)
                            arguments.push_back((Myelin::root() / source).string());
                        for (const char *source : SDK_NN)
                            arguments.push_back((Myelin::root() / source).string());
                        if (request.runtime == Runtime::Wasm)
                            for (const char *source : SDK_WASM)
                                arguments.push_back((Myelin::root() / source).string());
                        else
                            for (const char *source : SDK_HOST)
                                arguments.push_back((Myelin::root() / source).string());
                    }

                    for (const std::filesystem::path &directory : request.library_directories) {
                        arguments.push_back("-L" + directory.string());
                        if (request.runtime == Runtime::Native)
                            arguments.push_back("-Wl,-rpath," + directory.string());
                    }
                    for (const std::string &library : request.libraries)
                        arguments.push_back("-l" + library);

                    arguments.push_back("-o");
                    arguments.push_back(request.artifact.string());

                    std::error_code error;
                    std::filesystem::create_directories(request.artifact.parent_path(), error);

                    System::run_checked({.executable = toolchain.compiler.string(),
                                         .arguments = arguments,
                                         .working_directory = request.working_directory,
                                         .environment = {}},
                                        "compilando " + request.artifact.filename().string());
                }

            private:
                bool c_;
        };

        const Cpp cpp_driver{false};
        const Cpp c_driver{true};

        const int registered = [] {
            register_driver(cpp_driver);
            register_driver(c_driver);
            return 0;
        }();
    }
}
