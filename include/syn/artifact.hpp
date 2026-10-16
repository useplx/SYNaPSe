#pragma once

#include "mln/artifact.hpp"

namespace Syn::Compiler {
    void validate_artifact_plan(const Mln::Artifact::Plan &plan, bool require_entrypoints = true);
}
