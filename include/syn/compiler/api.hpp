#pragma once

#include "syn/config/value.hpp"

#include <filesystem>
#include <memory>
#include <string>

struct mln_compiler_api;

namespace Syn::Compiler {
    class Api {
        public:
            static Api load(const std::filesystem::path &library);

            ~Api();
            Api(Api &&) noexcept;
            Api &operator=(Api &&) noexcept;
            Api(const Api &) = delete;
            Api &operator=(const Api &) = delete;

            void compile(const Config::Value &plan, const std::filesystem::path &output) const;

        private:
            struct Library;
            std::unique_ptr<Library> library_;
            const mln_compiler_api *api_ = nullptr;

            Api(std::unique_ptr<Library> library, const mln_compiler_api *api);
    };
}
