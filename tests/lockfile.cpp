#include "test.hpp"

#include "syn/package/lockfile.hpp"

#include <filesystem>
#include <fstream>

SYNPS_TEST(writes_and_reads_lockfile) {
    const auto root = std::filesystem::temp_directory_path() / "synapse-lockfile-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "dependency");
    {
        std::ofstream package(root / "dependency/synps.toml");
        package << "[package]\nname=\"dependency\"\nversion=\"2.0.0\"\n";
    }
    {
        std::ofstream project(root / "synps.toml");
        project << "[package]\nname=\"application\"\nversion=\"1.0.0\"\n"
                   "[dependencies]\ndep={path=\"dependency\"}\n";
    }
    const auto resolution = Syn::Package::Resolver({root / "cache"}).resolve(root / "synps.toml");
    const auto lock = Syn::Package::Lockfile::create(resolution);
    lock.write(root / "synps.lock");
    const auto loaded = Syn::Package::Lockfile::load(root / "synps.lock");
    Syn::Testing::require(loaded.packages().size() == 2, "lockfile package count changed");
    Syn::Testing::require(loaded.packages().at(0).dependencies.at(0).name == "dep",
                          "lockfile dependency alias was lost");
    Syn::Testing::require(loaded.packages().at(1).content_hash.size() == 64,
                          "lockfile content hash is invalid");
    std::filesystem::remove_all(root);
}
