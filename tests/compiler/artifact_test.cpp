#include "mln/error.hpp"
#include "syn/artifact.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {
    template <typename Function> void rejects(Function &&function, const std::string &description) {
        try {
            function();
        } catch (const Mln::Diagnostic &) {
            return;
        }
        throw std::runtime_error("Artifact::Plan accepted " + description);
    }
}

int main(int argc, char **argv) {
    if (argc != 2)
        throw std::invalid_argument("artifact_test requer diretorio temporario");

    const std::filesystem::path root(argv[1]);
    std::filesystem::create_directories(root);
    const auto wasm = root / "orchestration.wasm";
    const auto tools_wasm = root / "tools.wasm";
    const auto native = root / "provider.so";
    const auto driver = root / "qnn.so";
    const auto asset = root / "tokenizer.json";
    for (const auto &path : {wasm, tools_wasm, native, driver, asset})
        std::ofstream(path, std::ios::binary).put('\0');

    using namespace Syn::Compiler;
    const Mln::Artifact::Plan valid{
        .modules =
            {
                {Mln::Artifact::Runtime::Wasm, {"generation", wasm.string()}},
                {Mln::Artifact::Runtime::Wasm, {"tools", tools_wasm.string()}},
                {Mln::Artifact::Runtime::Native, {"tokenizer", native.string()}},
                {Mln::Artifact::Runtime::Driver, {"qnn", driver.string()}},
            },
        .assets = {{"tokenizer.json", asset.string()}},
        .entrypoints =
            {
                {"chat", "generation", "forward"},
                {"embed", "tools", "forward"},
            },
    };
    validate_artifact_plan(valid);

    const Mln::Artifact::Plan native_only{
        .modules = {{Mln::Artifact::Runtime::Native, {"entrypoint", native.string()}}},
        .entrypoints = {{"run", "entrypoint", "forward", Mln::Artifact::Runtime::Native}},
    };
    validate_artifact_plan(native_only);

    const Mln::Artifact::Plan graph_only{
        .entrypoints = {{"run", "model", "model.forward", Mln::Artifact::Runtime::Graph}},
    };
    validate_artifact_plan(graph_only);

    Mln::Artifact::Plan no_entrypoint = valid;
    no_entrypoint.entrypoints.clear();
    rejects([&] { validate_artifact_plan(no_entrypoint); }, "package sem entrypoint");

    Mln::Artifact::Plan duplicate = valid;
    duplicate.modules.push_back(duplicate.modules.front());
    rejects([&] { validate_artifact_plan(duplicate); }, "modulos duplicados");

    Mln::Artifact::Plan missing = valid;
    missing.modules.front().artifact.path = (root / "missing.wasm").string();
    rejects([&] { validate_artifact_plan(missing); }, "modulo inexistente");

    Mln::Artifact::Plan unknown_module = valid;
    unknown_module.entrypoints.front().module = "missing";
    rejects([&] { validate_artifact_plan(unknown_module); }, "entrypoint para modulo inexistente");
}
