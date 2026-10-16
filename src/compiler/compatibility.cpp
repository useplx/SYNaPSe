#include "syn/compatibility.hpp"

#include "mln/error.hpp"
#include "mln/sdk/kind.hpp"
#include "syn/manifest.hpp"

#include <algorithm>
#include <format>
#include <string>

namespace Syn::Compiler::Compatibility {
    namespace {
        constexpr Mln::Error Missing{"synapse", 2102,
                                     "no explicit {} compatibility for '{}' on backend '{}'"};

        std::vector<std::pair<std::string, std::string>>
        normalize(std::span<const std::pair<std::string, std::string>> properties) {
            std::vector<std::pair<std::string, std::string>> result(properties.begin(),
                                                                    properties.end());
            std::ranges::sort(result);
            return result;
        }

        std::string key(Subject subject, std::string_view backend, std::string_view item,
                        std::span<const std::pair<std::string, std::string>> properties) {
            std::string result =
                std::format("{}\n{}\n{}", static_cast<int>(subject), backend, item);
            for (const auto &[name, value] : normalize(properties))
                result += std::format("\n{}={}", name, value);
            return result;
        }

        std::string describe(std::string_view backend,
                             std::span<const std::pair<std::string, std::string>> properties) {
            std::string result(backend);
            const auto sorted = normalize(properties);
            if (sorted.empty())
                return result;
            result += " (";
            for (std::size_t i = 0; i < sorted.size(); ++i) {
                if (i)
                    result += ", ";
                result += sorted[i].first;
                result += '=';
                result += sorted[i].second;
            }
            result += ')';
            return result;
        }
    }

    bool Matrix::record(Subject subject, std::string_view backend, std::string item,
                        std::string_view path, bool implemented,
                        std::span<const std::pair<std::string, std::string>> properties) {
        const std::string identity = key(subject, backend, item, properties);
        if (const auto found = index_.find(identity); found != index_.end()) {
            Entry &entry = entries_[found->second];
            const bool first_missing = !implemented && entry.covered == entry.uses;
            ++entry.uses;
            entry.covered += implemented ? 1 : 0;
            entry.implemented = entry.implemented || implemented;
            if (first_missing)
                Mln::warn_at(Missing, std::string(path),
                             subject == Subject::Operation ? "operation" : "codec", item,
                             describe(backend, properties));
            return implemented;
        }

        const std::size_t position = entries_.size();
        index_.emplace(identity, position);
        entries_.push_back(Entry{
            .subject = subject,
            .backend = std::string(backend),
            .properties = normalize(properties),
            .name = std::move(item),
            .path = std::string(path),
            .implemented = implemented,
            .uses = 1,
            .covered = implemented ? 1u : 0u,
        });
        if (!implemented) {
            const Entry &entry = entries_.back();
            Mln::warn_at(Missing, entry.path, subject == Subject::Operation ? "operation" : "codec",
                         entry.name, describe(entry.backend, entry.properties));
        }
        return implemented;
    }

    bool Matrix::operation(const Mln::Sdk::Operation::Compatibility::Context &context,
                           std::string_view path) {
        const std::string_view backend = context.target ? context.target->device : "none";
        const bool implemented =
            std::ranges::any_of(
                manifest_.operation_compatibilities,
                [&](const auto *candidate) { return candidate && candidate->supports(context); }) ||
            std::ranges::any_of(manifest_.backend_compatibilities, [&](const auto *candidate) {
                return candidate && candidate->supports(context);
            });
        const auto properties = context.target
                                    ? context.target->properties
                                    : std::span<const std::pair<std::string, std::string>>{};
        return record(Subject::Operation, backend,
                      std::string(Mln::Sdk::Functional::name(context.operation.kind)), path,
                      implemented, properties);
    }

    bool Matrix::codec(const Codec::Compatibility::Context &context) {
        const bool adapted =
            std::ranges::any_of(manifest_.codec_compatibilities, [&](const auto *candidate) {
                return candidate && candidate->applies(context);
            });

        const bool implemented = adapted || context.codec.supports(context.weight, context.scheme);
        return record(Subject::Codec, context.backend, context.codec.name, context.parameter_path,
                      implemented, context.properties);
    }

    const Entry *
    Matrix::find(Subject subject, std::string_view backend, std::string_view item,
                 std::span<const std::pair<std::string, std::string>> properties) const {
        const auto found = index_.find(key(subject, backend, item, properties));
        return found == index_.end() ? nullptr : &entries_[found->second];
    }
}
