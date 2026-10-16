#pragma once

#include "syn/package/resolver.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace Syn::Package {
    namespace Locked {
        struct Dependency {
                std::string name;
                std::string package;
        };

        struct Package {
                std::string name;
                std::string version;
                std::string source;
                std::string location;
                std::string revision;
                std::string content_hash;
                std::vector<Dependency> dependencies;
        };
    }

    class Lockfile {
        public:
            static Lockfile create(const Resolution &resolution);
            static Lockfile load(const std::filesystem::path &path);

            void write(const std::filesystem::path &path) const;
            const std::vector<Locked::Package> &packages() const noexcept { return packages_; }

        private:
            std::vector<Locked::Package> packages_;
    };
}
