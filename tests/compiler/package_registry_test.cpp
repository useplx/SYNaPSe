#include "mln/error.hpp"
#include "syn/package_registry.hpp"

#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2)
        throw std::invalid_argument("package_registry_test requer plugin");

    using namespace Syn::Compiler;
    Package::Registry registry;
    registry.load(argv[1]);

    const Package::Export *graph =
        registry.resolve(Package::Export::Kind::Graph, "calculator:calculator");
    if (!graph || graph->package_version != "0.1.0")
        throw std::runtime_error("graph export was not resolved");
    if (registry.resolve(Package::Export::Kind::Codec, "calculator:calculator"))
        throw std::runtime_error("registry ignorou o kind do export");
    if (!registry.resolve(Package::Export::Kind::Codec, "calculator:f16"))
        throw std::runtime_error("quantization export was not resolved");
    if (!registry.resolve(Package::Export::Kind::CodecCompatibility, "calculator:compatibility"))
        throw std::runtime_error("compatibility export was not resolved");
    if (!registry.resolve(Package::Export::Kind::OperationCompatibility,
                          "calculator:operation_compatibility"))
        throw std::runtime_error("operation compatibility export was not resolved");
    if (!registry.resolve(Package::Export::Kind::LoweringCompatibility,
                          "calculator:lowering_compatibility"))
        throw std::runtime_error("lowering compatibility export was not resolved");
    if (!registry.resolve(Package::Export::Kind::BackendCompatibility,
                          "calculator:backend_compatibility"))
        throw std::runtime_error("backend compatibility export was not resolved");
    if (registry.exports().size() != 6)
        throw std::runtime_error("incorrect export count");

    try {
        registry.load(argv[1]);
    } catch (const Mln::Diagnostic &) {
        return 0;
    }
    throw std::runtime_error("registry accepted duplicate exports");
}
