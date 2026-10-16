#pragma once

#include "syn/config/project.hpp"

#include <string_view>

namespace Syn::Config {
    namespace Schema {
        Value resolve(const Project &package, const Value *overrides, std::string_view reference);
    }
}
