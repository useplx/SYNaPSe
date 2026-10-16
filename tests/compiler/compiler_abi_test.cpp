#include "syn/compiler_abi.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace {
    void diagnostic(void *user_data, const char *message, std::size_t length) {
        static_cast<std::string *>(user_data)->append(message, length);
    }
}

int main() {
    const mln_compiler_api *api = mln_compiler_api_get();
    if (!api || api->abi_version != mln_compiler_api_version)
        return 1;

    const std::string plan = "{}";
    std::string message;
    mln_compiler_request request{};
    request.abi_version = mln_compiler_api_version;
    request.struct_size = offsetof(mln_compiler_request, actionable_diagnostic);
    request.plan = plan.data();
    request.plan_length = plan.size();
    request.output_path = "/tmp/compiler-abi-test.mln";
    request.diagnostic = diagnostic;
    request.user_data = &message;

    const std::int32_t status = api->compile(&request);
    if (status != 1 || message.empty())
        return 2;
    return 0;
}
