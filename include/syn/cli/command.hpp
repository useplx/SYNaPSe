#pragma once

#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace Syn::Cli {
    struct Argument {
            std::string_view name;
            std::string_view description;
            bool required = true;
    };

    struct Option {
            std::string_view name;
            std::string_view value;
            std::string_view description;
    };

    class Invocation {
        public:
            std::string_view argument(std::string_view name) const;
            bool has(std::string_view option) const;
            std::string_view option(std::string_view name) const;

        private:
            friend class Commands;
            std::map<std::string_view, std::string, std::less<>> arguments_;
            std::map<std::string_view, std::string, std::less<>> options_;
    };

    class Command {
        public:
            Command(std::string_view name, std::string_view summary,
                    std::vector<Argument> arguments = {}, std::vector<Option> options = {});
            virtual ~Command() = default;

            Command(const Command &) = delete;
            Command &operator=(const Command &) = delete;

            std::string_view name() const noexcept { return name_; }
            std::string_view summary() const noexcept { return summary_; }
            const std::vector<Argument> &arguments() const noexcept { return arguments_; }
            const std::vector<Option> &options() const noexcept { return options_; }

            std::string signature() const;

            virtual int run(const Invocation &invocation) const = 0;

        private:
            std::string_view name_;
            std::string_view summary_;
            std::vector<Argument> arguments_;
            std::vector<Option> options_;
    };

    class Commands {
        public:
            static void add(const Command &command);
            static const std::map<std::string_view, const Command *, std::less<>> &all();
            static const Command *find(std::string_view name);

            static const std::vector<Option> &global();

            static std::string usage();
            static std::string usage(const Command &command);

            static Invocation parse(const Command &command, int argc, char **argv, int from);
    };
}
