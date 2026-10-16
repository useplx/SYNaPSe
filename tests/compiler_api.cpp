#include "test.hpp"

#include "syn/compiler/api.hpp"
#include "syn/compiler/plan.hpp"
#include "syn/error.hpp"

#include "mln/package.hpp"
#include "mln/weight.hpp"

#include <cstdlib>
#include <filesystem>

SYNPS_TEST(loads_mln_compiler_api_in_process) {
    const char *library = std::getenv("MLN_COMPILER_API");
    Syn::Testing::require(library != nullptr, "MLN_COMPILER_API is not set for integration test");
    auto api = Syn::Compiler::Api::load(library);
    Syn::Config::Value::Table plan;
    plan.emplace("version", int64_t{2});
    try {
        api.compile(std::move(plan), "/tmp/invalid.mln");
        Syn::Testing::require(false, "invalid compiler plan was accepted");
    } catch (const Syn::Diagnostic &error) {
        Syn::Testing::require(error.error() == Syn::Error::Compilation::Failed,
                              "compiler callback returned the wrong diagnostic");
        Syn::Testing::require(std::string(error.what()).find("plan version is unsupported") !=
                                  std::string::npos,
                              "compiler callback diagnostic was lost");
    }
}

#if defined(SYNPS_TEST_MLN_PACKAGE)
SYNPS_TEST(merge_graphs_rejects_duplicate_vm_module_names) {
    const char *library = std::getenv("MLN_COMPILER_API");
    Syn::Testing::require(library != nullptr, "MLN_COMPILER_API is not set for integration test");
    const auto root = std::filesystem::temp_directory_path() / "synapse-merge-graphs-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{root};

    const auto write_shard = [&](const std::filesystem::path &path) {
        Mln::Package::Builder builder;
        const std::string devices = "local-task://\n";
        builder.add(Mln::Package::Section::Type::Devices, "devices",
                    std::vector<uint8_t>(devices.begin(), devices.end()));
        const std::string deployment = "0\tmodel.forward\n";
        builder.add(Mln::Package::Section::Type::Deployment, "entrypoints",
                    std::vector<uint8_t>(deployment.begin(), deployment.end()));
        builder.add(Mln::Package::Section::Type::Vmfb, "internal", {1, 2, 3});
        builder.add(
            Mln::Package::Section::Type::WeightIndex, "requirements",
            Mln::Weight::encode_weight_index(Mln::Weight::Index::Kind::Requirements,
                                             {Mln::Weight::Descriptor{.name = "model.weight",
                                                                      .quant = "F16",
                                                                      .storage = "f16",
                                                                      .shape = {2, 2},
                                                                      .byte_length = 8}}));
        builder.write(path.string());
    };
    write_shard(root / "first.mln");
    write_shard(root / "second.mln");

    auto api = Syn::Compiler::Api::load(library);
    try {
        api.compile(Syn::Compiler::Plan::merge_graphs(
                        {{"decoder", root / "first.mln"}, {"decoder", root / "second.mln"}}),
                    root / "merged.mln");
        Syn::Testing::require(false, "duplicate VM module name was accepted");
    } catch (const Syn::Diagnostic &error) {
        Syn::Testing::require(std::string(error.what()).find("duplicate graph shard module") !=
                                  std::string::npos,
                              "duplicate module diagnostic was lost");
    }
}
#endif
