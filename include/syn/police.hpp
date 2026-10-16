#pragma once

#include "mln/dtype.hpp"
#include "mln/sdk/codec.hpp"
#include "syn/backend.hpp"

#include <optional>
#include <string>
#include <vector>

namespace Syn::Compiler {
    enum class LoadingMode { Resident, Streamed };

    struct Police {
            std::vector<Mln::Sdk::Codec::Scheme> schemes;

            std::optional<Mln::DType> dtype;
            std::optional<Mln::Sdk::Backend> backend;
            std::optional<std::vector<std::string>> entrypoints;
            std::optional<bool> optimization_barrier;
            std::optional<LoadingMode> loading;
    };
}
