#pragma once

#include "mln/sdk/module.hpp"
#include "mln/sdk/tensor.hpp"

#include <functional>
#include <span>
#include <string>

namespace Syn::Compiler {
    struct Hook {

            std::string path;

            std::function<Mln::Sdk::Tensor(Mln::Sdk::Module &module,
                                           std::span<const Mln::Sdk::Tensor> inputs,
                                           Mln::Sdk::Tensor output)>
                forward;
    };
}
