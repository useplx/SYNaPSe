#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum mln_package_export_kind {
    mln_package_graph = 1,
    mln_package_codec = 2,
    mln_package_hook = 3,
    mln_package_codec_compatibility = 4,
    mln_package_operation_compatibility = 5,
    mln_package_lowering_compatibility = 6,
    mln_package_backend_compatibility = 7,
};

struct mln_package_invocation {
        uint32_t abi_version;
        uint32_t struct_size;
        void *context;
        const char *config;
        size_t config_length;
        char *error;
        size_t error_capacity;
};

typedef int32_t (*mln_package_invoke)(const struct mln_package_invocation *invocation,
                                      void *user_data);

struct mln_package_export {
        uint32_t struct_size;
        uint32_t kind;
        const char *name;
        mln_package_invoke invoke;
        void *user_data;
};

struct mln_package_registration {
        uint32_t abi_version;
        uint32_t struct_size;
        const char *name;
        const char *version;
        const struct mln_package_export *exports;
        size_t export_count;
};

typedef const struct mln_package_registration *(*mln_package_register_function)(void);

const struct mln_package_registration *mln_package_register(void);

#ifdef __cplusplus
}
#endif
