#include "syn/toolchain/abi.hpp"
#include "syn/toolchain/driver.hpp"

#include "syn/error.hpp"
#include "syn/system/process.hpp"

namespace Syn::Toolchain {
    namespace {
        class Rust final : public Driver {
            public:
                std::string_view language() const override { return "rust"; }

                void probe(const Config::Toolchain &toolchain) const override {
                    if (toolchain.compiler.empty())
                        Errors::raise(Error::Toolchain::Compiler::Missing, language());
                    if (System::run({.executable = toolchain.compiler.string(),
                                     .arguments = {"--version"},
                                     .working_directory = {},
                                     .environment = {}})
                            .exit_code != 0)
                        Errors::raise(Error::Compiler::Unresponsive, toolchain.compiler,
                                      "--version");
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
                        Errors::raise(Error::Sources::Missing, language(),
                                      "Cargo.toml em `sources`");

                    const std::filesystem::path manifest = request.sources.front();
                    std::vector<std::string> arguments{"build", "--manifest-path",
                                                       manifest.string(), "--locked"};
                    if (request.optimize)
                        arguments.push_back("--release");

                    std::string triple = toolchain.triple;
                    if (request.runtime == Runtime::Wasm && triple.empty())
                        triple = "wasm32-wasip1";
                    if (!triple.empty()) {
                        arguments.push_back("--target");
                        arguments.push_back(triple);
                    }
                    for (const std::string &flag : toolchain.flags)
                        arguments.push_back(flag);

                    System::run_checked({.executable = toolchain.compiler.string(),
                                         .arguments = arguments,
                                         .working_directory = request.working_directory,
                                         .environment = {}},
                                        "compilando crate " +
                                            manifest.parent_path().filename().string());

                    std::filesystem::path produced = manifest.parent_path() / "target";
                    if (!triple.empty())
                        produced /= triple;
                    produced /= request.optimize ? "release" : "debug";
                    produced /= request.artifact.filename();

                    std::error_code error;
                    std::filesystem::create_directories(request.artifact.parent_path(), error);
                    std::filesystem::copy_file(produced, request.artifact,
                                               std::filesystem::copy_options::overwrite_existing,
                                               error);
                    if (error)
                        Errors::raise(Error::Artifact::Not::Produced, produced);
                }
        };

        const Rust rust_driver;
        const int registered = [] {
            register_driver(rust_driver);
            return 0;
        }();
    }
}
