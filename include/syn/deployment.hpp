#pragma once

#include "syn/manifest.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

namespace Syn::Compiler {
    std::vector<Mln::Sdk::Backend> deployment_backends(const Manifest &manifest);

    void apply_backend(mlir::Operation *operation, const Manifest &manifest, const Police &police);
}
