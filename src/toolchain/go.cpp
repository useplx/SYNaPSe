#include "syn/toolchain/abi.hpp"
#include "syn/toolchain/driver.hpp"

#include "syn/error.hpp"
#include "syn/system/process.hpp"

namespace Syn::Toolchain {
    namespace {
        class Go final : public Driver {
            public:
                std::string_view language() const override { return "go"; }

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
                    return output == Output::SharedLibrary || output == Output::Executable;
                }

                void build(const Config::Toolchain &toolchain,
                           const Request &request) const override {
                    if (!supports(request.runtime, request.output))
                        Errors::raise(Error::Output::Unsupported, language(),
                                      runtime_name(request.runtime));

                    const bool tinygo =
                        toolchain.compiler.filename().string().find("tinygo") != std::string::npos;
                    if (request.runtime == Runtime::Wasm && !tinygo)
                        Errors::raise(Error::Go::Requires::Tinygo);

                    std::vector<std::string> arguments{"build"};
                    if (tinygo) {
                        arguments.push_back("-target");
                        arguments.push_back(request.runtime == Runtime::Wasm ? "wasi" : "default");
                    } else {
                        arguments.push_back("-buildmode=c-shared");
                    }
                    if (request.optimize && tinygo) {
                        arguments.push_back("-opt");
                        arguments.push_back("2");
                    }
                    for (const std::string &flag : toolchain.flags)
                        arguments.push_back(flag);

                    arguments.push_back("-o");
                    arguments.push_back(request.artifact.string());
                    for (const std::filesystem::path &source : request.sources)
                        arguments.push_back(source.string());

                    std::error_code error;
                    std::filesystem::create_directories(request.artifact.parent_path(), error);
                    System::run_checked({.executable = toolchain.compiler.string(),
                                         .arguments = arguments,
                                         .working_directory = request.working_directory,
                                         .environment = {}},
                                        "compilando " + request.artifact.filename().string());
                }
        };

        const Go go_driver;
        const int registered = [] {
            register_driver(go_driver);
            return 0;
        }();
    }
}
