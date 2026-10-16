#include "syn/package_abi.h"

namespace {
    int32_t invoke(const mln_package_invocation *, void *) { return 0; }

    const mln_package_export exports[] = {
        {sizeof(mln_package_export), mln_package_graph, "calculator", invoke, nullptr},
        {sizeof(mln_package_export), mln_package_codec, "f16", invoke, nullptr},
        {sizeof(mln_package_export), mln_package_codec_compatibility, "compatibility", invoke,
         nullptr},
        {sizeof(mln_package_export), mln_package_operation_compatibility, "operation_compatibility",
         invoke, nullptr},
        {sizeof(mln_package_export), mln_package_lowering_compatibility, "lowering_compatibility",
         invoke, nullptr},
        {sizeof(mln_package_export), mln_package_backend_compatibility, "backend_compatibility",
         invoke, nullptr},
    };

    const mln_package_registration registration{
        6,       sizeof(mln_package_registration),     "calculator", "0.1.0",
        exports, sizeof(exports) / sizeof(exports[0]),
    };
}

extern "C" const mln_package_registration *mln_package_register() { return &registration; }
