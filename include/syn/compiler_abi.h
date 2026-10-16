#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(SYN_COMPILER_API_BUILD)
#define SYN_COMPILER_API __declspec(dllexport)
#else
#define SYN_COMPILER_API __declspec(dllimport)
#endif
#else
#define SYN_COMPILER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum { mln_compiler_api_version = 2 };

enum mln_compiler_diagnostic_severity {
    MLN_COMPILER_DIAGNOSTIC_WARNING = 0,
    MLN_COMPILER_DIAGNOSTIC_ERROR = 1,
};

enum mln_compiler_diagnostic_action {
    MLN_COMPILER_DIAGNOSTIC_CONTINUE = 0,
    MLN_COMPILER_DIAGNOSTIC_IGNORE = 1,
    MLN_COMPILER_DIAGNOSTIC_PROMOTE_TO_ERROR = 2,
};

typedef void (*mln_compiler_diagnostic)(void *user_data, const char *message,
                                        size_t message_length);
typedef void (*mln_compiler_structured_diagnostic)(void *user_data,
                                                   enum mln_compiler_diagnostic_severity severity,
                                                   const char *domain, size_t domain_length,
                                                   uint16_t code, const char *message,
                                                   size_t message_length, const char *location,
                                                   size_t location_length);
typedef enum mln_compiler_diagnostic_action (*mln_compiler_actionable_diagnostic)(
    void *user_data, enum mln_compiler_diagnostic_severity severity, const char *domain,
    size_t domain_length, uint16_t code, const char *message, size_t message_length,
    const char *location, size_t location_length);

struct mln_compiler_request {
        uint32_t abi_version;
        uint32_t struct_size;
        const char *plan;
        size_t plan_length;
        const char *output_path;
        mln_compiler_diagnostic diagnostic;
        void *user_data;
        mln_compiler_structured_diagnostic structured_diagnostic;

        mln_compiler_actionable_diagnostic actionable_diagnostic;
};

struct mln_compiler_api {
        uint32_t abi_version;
        uint32_t struct_size;
        int32_t (*compile)(const struct mln_compiler_request *request);
};

SYN_COMPILER_API const struct mln_compiler_api *mln_compiler_api_get(void);

#ifdef __cplusplus
}
#endif
