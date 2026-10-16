#include "syn/cli/command.hpp"

#include "syn/error.hpp"

#include <algorithm>

namespace Syn::Cli {
    namespace {
        constexpr std::string_view HEADLINE = "synps -- project and package manager for Myelin";

        std::map<std::string_view, const Command *, std::less<>> &registry() {
            static std::map<std::string_view, const Command *, std::less<>> values;
            return values;
        }

        std::string describe(std::string_view signature, std::string_view description,
                             std::size_t width) {
            std::string line = "    " + std::string(signature);
            line.append(width - std::min(width, signature.size()), ' ');
            line += "  ";
            line += description;
            line += '\n';
            return line;
        }

        std::string option_signature(const Option &option) {
            std::string result = "--" + std::string(option.name);
            if (!option.value.empty())
                result += " " + std::string(option.value);
            return result;
        }
    }

    std::string_view Invocation::argument(std::string_view name) const {
        const auto found = arguments_.find(name);
        return found == arguments_.end() ? std::string_view{} : found->second;
    }

    bool Invocation::has(std::string_view option) const { return options_.contains(option); }

    std::string_view Invocation::option(std::string_view name) const {
        const auto found = options_.find(name);
        return found == options_.end() ? std::string_view{} : found->second;
    }

    Command::Command(std::string_view name, std::string_view summary,
                     std::vector<Argument> arguments, std::vector<Option> options)
        : name_(name), summary_(summary), arguments_(std::move(arguments)),
          options_(std::move(options)) {}

    std::string Command::signature() const {
        std::string result = "synps " + std::string(name_);
        for (const Argument &argument : arguments_)
            result += argument.required ? " <" + std::string(argument.name) + ">"
                                        : " [" + std::string(argument.name) + "]";
        for (const Option &option : options_)
            result += " [" + option_signature(option) + "]";
        return result;
    }

    void Commands::add(const Command &command) { registry()[command.name()] = &command; }

    const std::map<std::string_view, const Command *, std::less<>> &Commands::all() {
        return registry();
    }

    const Command *Commands::find(std::string_view name) {
        const auto found = registry().find(name);
        return found == registry().end() ? nullptr : found->second;
    }

    const std::vector<Option> &Commands::global() {
        static const std::vector<Option> values{
            {"project", "file", "use another synps.toml for the command"},
        };
        return values;
    }

    std::string Commands::usage() {
        std::size_t width = 0;
        for (const auto &[name, command] : all())
            width = std::max(width, command->signature().size());
        for (const Option &option : global())
            width = std::max(width, option_signature(option).size());

        std::string result = std::string(HEADLINE) + "\n\n";
        for (const auto &[name, command] : all())
            result += describe(command->signature(), command->summary(), width);
        result += '\n';
        for (const Option &option : global())
            result += describe(option_signature(option), option.description, width);
        result += describe("--version, -V", "mostra a versao instalada", width);
        result += "\n    synps help <comando> descreve cada argumento\n";
        return result;
    }

    std::string Commands::usage(const Command &command) {
        std::size_t width = 0;
        for (const Argument &argument : command.arguments())
            width = std::max(width, argument.name.size() + 2);
        for (const Option &option : command.options())
            width = std::max(width, option_signature(option).size());
        for (const Option &option : global())
            width = std::max(width, option_signature(option).size());

        std::string result =
            command.signature() + "\n\n    " + std::string(command.summary()) + "\n";
        if (!command.arguments().empty()) {
            result += "\nargumentos:\n";
            for (const Argument &argument : command.arguments())
                result +=
                    describe("<" + std::string(argument.name) + ">", argument.description, width);
        }
        result += "\nopcoes:\n";
        for (const Option &option : command.options())
            result += describe(option_signature(option), option.description, width);
        for (const Option &option : global())
            result += describe(option_signature(option), option.description, width);
        return result;
    }

    Invocation Commands::parse(const Command &command, int argc, char **argv, int from) {
        const auto declared = [&](std::string_view name) -> const Option * {
            for (const Option &option : command.options())
                if (option.name == name)
                    return &option;
            for (const Option &option : global())
                if (option.name == name)
                    return &option;
            return nullptr;
        };

        Invocation invocation;
        std::vector<std::string> positional;
        for (int index = from; index < argc; ++index) {
            const std::string entry = argv[index];
            if (entry.empty() || entry.front() != '-') {
                positional.push_back(entry);
                continue;
            }

            const std::size_t equals = entry.find('=');
            const std::size_t start = entry.starts_with("--") ? 2 : 1;
            const std::string name = entry.substr(
                start, equals == std::string::npos ? std::string::npos : equals - start);
            const Option *option = declared(name);
            if (option == nullptr)
                Errors::raise(Error::Option::Unknown, name, command.name());

            if (option->value.empty()) {
                if (equals != std::string::npos)
                    Errors::raise(Error::Cli::Usage, command.signature());
                invocation.options_.emplace(option->name, std::string{});
                continue;
            }

            if (equals != std::string::npos) {
                invocation.options_.emplace(option->name, entry.substr(equals + 1));
                continue;
            }
            if (index + 1 >= argc || argv[index + 1][0] == '-')
                Errors::raise(Error::Option::Requires::Value, option->name);
            invocation.options_.emplace(option->name, argv[++index]);
        }

        std::size_t required = 0;
        for (const Argument &argument : command.arguments())
            if (argument.required)
                ++required;
        if (positional.size() < required || positional.size() > command.arguments().size())
            Errors::raise(Error::Cli::Usage, command.signature());

        for (std::size_t index = 0; index < positional.size(); ++index)
            invocation.arguments_.emplace(command.arguments()[index].name,
                                          std::move(positional[index]));
        return invocation;
    }
}
