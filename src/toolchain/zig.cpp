#include "syn/toolchain/abi.hpp"
#include "syn/toolchain/driver.hpp"

#include "syn/error.hpp"
#include "syn/system/process.hpp"

namespace Syn::Toolchain {
    namespace {
        class Zig final : public Driver {
            public:
                std::string_view language() const override { return "zig"; }

                void probe(const Config::Toolchain &toolchain) const override {
                    if (toolchain.compiler.empty())
                        Errors::raise(Error::Toolchain::Compiler::Missing, language());
                    if (System::run({.executable = toolchain.compiler.string(),
                                     .arguments = {"version"},
                                     .working_directory = {},
                                     .environment = {}})
                            .exit_code != 0)
                        Errors::raise(Error::Compiler::Unresponsive, toolchain.compiler, "version");
                }

                bool supports(Runtime runtime, Output output) const override {
                    if (runtime == Runtime::Wasm)
                        return output == Output::WasmModule;
                    return output == Output::SharedLibrary || output == Output::StaticLibrary ||
                           output == Output::Executable;
                }

                void build(const Config::Toolchain &toolchain,
                           const Request &request) const override {
                    if (!supports(request.runtime, request.output))
                        Errors::raise(Error::Output::Unsupported, language(),
                                      runtime_name(request.runtime));
                    if (request.sources.empty())
                        Errors::raise(Error::Sources::Missing, language(), "sources");

                    std::vector<std::string> arguments{
                        request.output == Output::Executable ? "build-exe" : "build-lib"};
                    for (const std::filesystem::path &source : request.sources)
                        arguments.push_back(source.string());

                    if (request.output == Output::SharedLibrary)
                        arguments.push_back("-dynamic");
                    if (request.optimize) {
                        arguments.push_back("-O");
                        arguments.push_back("ReleaseFast");
                    }

                    std::string triple = toolchain.triple;
                    if (request.runtime == Runtime::Wasm && triple.empty())
                        triple = "wasm32-wasi";
                    if (!triple.empty()) {
                        arguments.push_back("-target");
                        arguments.push_back(triple);
                    }

                    for (const std::filesystem::path &include : request.includes) {
                        arguments.push_back("-I");
                        arguments.push_back(include.string());
                    }
                    arguments.push_back("-I");
                    arguments.push_back((Myelin::root() / "include").string());

                    for (const std::filesystem::path &directory : request.library_directories)
                        arguments.push_back("-L" + directory.string());
                    for (const std::string &library : request.libraries)
                        arguments.push_back("-l" + library);
                    for (const std::string &flag : toolchain.flags)
                        arguments.push_back(flag);

                    arguments.push_back("-femit-bin=" + request.artifact.string());

                    std::error_code error;
                    std::filesystem::create_directories(request.artifact.parent_path(), error);
                    System::run_checked({.executable = toolchain.compiler.string(),
                                         .arguments = arguments,
                                         .working_directory = request.working_directory,
                                         .environment = {}},
                                        "compilando " + request.artifact.filename().string());
                }
        };

        const Zig zig_driver;
        const int registered = [] {
            register_driver(zig_driver);
            return 0;
        }();
    }
}
