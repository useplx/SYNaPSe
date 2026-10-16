#pragma once

#include "syn/toolchain/driver.hpp"

#include <string>
#include <vector>

namespace Syn::Toolchain {
    namespace Abi {
        struct Requirements {
                bool reactor_model = false;

                bool disable_exceptions = false;

                bool position_independent = false;

                std::vector<std::string> defines;

                bool mln_sdk_core = false;

                std::vector<std::string> required_symbols;
        };

        Requirements requirements(Runtime runtime, Output output);
    }

    namespace Myelin {
        void set_root(std::filesystem::path root);
        const std::filesystem::path &root();
    }
}
