#include "syn/manifest.hpp"
#include "mln/error.hpp"
#include "syn/errors/manifest.hpp"

#include <algorithm>
#include <compare>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <typeindex>
#include <variant>
#include <vector>

namespace Syn::Compiler {
    namespace {
        using Codec = Mln::Sdk::Codec;
        using Scope = Codec::Scope;
        using Lifetime = Codec::Lifetime;
        using CodecScheme = Codec::Scheme;

        bool matches(const std::string &path, const std::string &prefix) {
            std::string_view remaining(path);
            std::string_view pattern(prefix);

            while (true) {
                const std::size_t pattern_end = pattern.find('.');
                const std::size_t path_end = remaining.find('.');
                const std::string_view pattern_segment = pattern.substr(0, pattern_end);
                const std::string_view path_segment = remaining.substr(0, path_end);
                if (pattern_segment != "*" && pattern_segment != path_segment)
                    return false;

                if (pattern_end == std::string_view::npos)
                    return true;
                if (path_end == std::string_view::npos)
                    return false;
                pattern.remove_prefix(pattern_end + 1);
                remaining.remove_prefix(path_end + 1);
            }
        }

        struct PathSpecificity {
                std::size_t segments = 0;
                std::size_t literals = 0;
                std::size_t characters = 0;

                auto operator<=>(const PathSpecificity &) const = default;
        };

        PathSpecificity path_specificity(std::string_view path) {
            PathSpecificity result;
            result.characters = path.size();
            while (!path.empty()) {
                const std::size_t separator = path.find('.');
                const std::string_view segment = path.substr(0, separator);
                ++result.segments;
                if (segment != "*")
                    ++result.literals;
                if (separator == std::string_view::npos)
                    break;
                path.remove_prefix(separator + 1);
            }
            return result;
        }

        bool same_key(const CodecScheme &left, const CodecScheme &right) {
            return left.scope == right.scope && left.lifetime == right.lifetime;
        }

        void merge(Police &target, const Police &source) {
            for (const auto &scheme : source.schemes) {
                target.schemes.erase(
                    std::remove_if(target.schemes.begin(), target.schemes.end(),
                                   [&](const auto &current) { return same_key(current, scheme); }),
                    target.schemes.end());
            }

            target.schemes.insert(target.schemes.end(), source.schemes.begin(),
                                  source.schemes.end());

            if (source.dtype)
                target.dtype = source.dtype;
            if (source.backend)
                target.backend = source.backend;
            if (source.entrypoints)
                target.entrypoints = source.entrypoints;

            if (source.optimization_barrier)
                target.optimization_barrier = source.optimization_barrier;
            if (source.loading)
                target.loading = source.loading;
        }

        void register_format(std::map<std::string, std::type_index> &types, Codec &format,
                             const std::string &where) {
            if (format.name.empty()) {
                raise(Errors::Manifest::Format::Name::Missing, where);
            }

            const std::type_index type(typeid(format));
            const auto found = types.find(format.name);

            if (found == types.end()) {
                types.emplace(format.name, type);
            } else if (found->second != type) {
                raise(Errors::Manifest::Format::Name::Conflict, format.name);
            }
        }

        void validate_scheme(const CodecScheme &scheme, const std::string &path) {
            if (!scheme.format) {
                raise(Errors::Manifest::Scheme::Format::Missing, path);
            }

            Codec &format = scheme.format.get();

            switch (scheme.lifetime) {
            case Lifetime::Immutable:
                if (dynamic_cast<Mln::Sdk::Codec *>(&format) == nullptr) {
                    raise(Errors::Manifest::Format::Immutable::Unsupported, format.name);
                }

                if (scheme.update != Codec::Update::None) {
                    raise(Errors::Manifest::Scheme::Immutable::Update::Invalid, path);
                }

                break;

            case Lifetime::Invocation:
            case Lifetime::Session:
                raise(Errors::Manifest::Scheme::Lifetime::Unsupported, path);

            case Lifetime::Any:
                break;
            }
        }
    }

    const Mln::Sdk::Codec::Scheme *scheme_for(const Police &police, Mln::Sdk::Codec::Scope scope,
                                              Mln::Sdk::Codec::Lifetime lifetime) {
        const Mln::Sdk::Codec::Scheme *best = nullptr;
        int best_specificity = -1;

        for (const auto &scheme : police.schemes) {
            if (scheme.scope != Scope::Any && scheme.scope != scope) {
                continue;
            }

            if (scheme.lifetime != Lifetime::Any && scheme.lifetime != lifetime) {
                continue;
            }

            const int specificity =
                (scheme.scope == scope ? 1 : 0) + (scheme.lifetime == lifetime ? 1 : 0);

            if (specificity > best_specificity) {
                best = &scheme;
                best_specificity = specificity;
            }
        }

        return best;
    }

    Scheme::Match scheme_for(const Manifest &manifest, const std::string &path,
                             Mln::Sdk::Codec::Scope scope, Mln::Sdk::Codec::Lifetime lifetime) {
        Scheme::Match best{};
        int best_scheme_specificity = -1;
        PathSpecificity best_path_specificity{};

        auto consider = [&](const std::string &policy_path, const std::string &anchor_path,
                            const Police &police) {
            for (const auto &scheme : police.schemes) {
                if (scheme.scope != Scope::Any && scheme.scope != scope) {
                    continue;
                }

                if (scheme.lifetime != Lifetime::Any && scheme.lifetime != lifetime) {
                    continue;
                }

                const int scheme_specificity =
                    (scheme.scope == scope ? 1 : 0) + (scheme.lifetime == lifetime ? 1 : 0);

                const PathSpecificity specificity = path_specificity(policy_path);

                if (specificity > best_path_specificity ||
                    (specificity == best_path_specificity &&
                     scheme_specificity > best_scheme_specificity)) {
                    best = Scheme::Match{&scheme, anchor_path};
                    best_scheme_specificity = scheme_specificity;
                    best_path_specificity = specificity;
                }
            }
        };

        if (manifest.policies.empty()) {
            return best;
        }

        const std::string &root_path = manifest.policies.front().first;
        std::string root_anchor = root_path;

        if (!matches(path, root_path)) {
            const std::size_t separator = path.find('.');
            root_anchor = path.substr(0, separator);
        }

        consider(root_path, root_anchor, manifest.policies.front().second);

        for (std::size_t i = 1; i < manifest.policies.size(); ++i) {
            const auto &[policy_path, police] = manifest.policies[i];

            if (matches(path, policy_path)) {
                consider(policy_path, policy_path, police);
            }
        }

        return best;
    }

    void validate_manifest(const Manifest &manifest) {
        const auto &policies = manifest.policies;

        if (policies.empty()) {
            raise(Errors::Manifest::Policy::Root::Missing);
        }

        if (!policies.front().second.dtype) {
            raise(Errors::Manifest::Policy::Root::DType::Missing);
        }

        std::set<std::string> entrypoint_names;

        for (const Entrypoint &entrypoint : manifest.entrypoints) {
            if (entrypoint.name.empty()) {
                raise(Errors::Manifest::Entrypoint::Name::Missing);
            }

            if (!entrypoint_names.insert(entrypoint.name).second) {
                raise(Errors::Manifest::Entrypoint::Duplicated, entrypoint.name);
            }

            if (entrypoint.specializations.empty()) {
                raise(Errors::Manifest::Entrypoint::Specialization::Missing, entrypoint.name);
            }

            for (const Specialization &specialization : entrypoint.specializations) {
                if (specialization.inputs.empty()) {
                    raise(Errors::Manifest::Entrypoint::Specialization::Inputs::Missing,
                          entrypoint.name);
                }
                for (const Shape &input : specialization.inputs) {
                    if (std::any_of(input.dimensions.begin(), input.dimensions.end(),
                                    [](int64_t dimension) {
                                        return dimension == 0 || dimension < Shape::Dynamic;
                                    })) {
                        raise(Errors::Manifest::Entrypoint::Specialization::Dimension::Invalid,
                              entrypoint.name);
                    }
                }
            }
        }

        for (const Hook &hook : manifest.hooks) {
            if (hook.path.empty()) {
                raise(Errors::Manifest::Hook::Path::Missing);
            }

            if (hook.path.front() == '.' || hook.path.back() == '.') {
                raise(Errors::Manifest::Hook::Path::Invalid, hook.path);
            }

            if (!hook.forward) {
                raise(Errors::Manifest::Hook::Forward::Missing, hook.path);
            }

            std::size_t begin = 0;
            while (begin < hook.path.size()) {
                const std::size_t end = hook.path.find('.', begin);
                const std::string_view segment(hook.path.data() + begin,
                                               (end == std::string::npos ? hook.path.size() : end) -
                                                   begin);

                if (segment.empty() ||
                    (segment.find('*') != std::string_view::npos && segment != "*")) {
                    raise(Errors::Manifest::Hook::Path::Wildcard::Invalid, hook.path);
                }

                if (end == std::string::npos)
                    break;
                begin = end + 1;
            }
        }

        const CodecScheme *root_constant =
            scheme_for(policies.front().second, Scope::Constant, Lifetime::Immutable);

        if (!root_constant) {
            raise(Errors::Manifest::Scheme::Root::Constant::Missing);
        }

        if (!root_constant->format ||
            dynamic_cast<Mln::Sdk::Codec *>(&root_constant->format.get()) == nullptr) {
            raise(Errors::Manifest::Scheme::Root::Constant::Format::Invalid);
        }

        std::map<std::string, std::type_index> format_types;

        for (const auto &registered : manifest.formats) {
            if (!registered) {
                raise(Errors::Manifest::Format::Uninitialized);
            }

            register_format(format_types, registered.get(), "<formats>");
        }

        for (const auto &[path, police] : policies) {
            if (path.empty()) {
                raise(Errors::Manifest::Policy::Path::Empty);
            }

            std::set<std::pair<Scope, Lifetime>> keys;

            for (const auto &scheme : police.schemes) {
                if (!keys.emplace(scheme.scope, scheme.lifetime).second) {
                    raise(Errors::Manifest::Scheme::Duplicated, path);
                }

                validate_scheme(scheme, path);
                register_format(format_types, scheme.format.get(), path);
            }

            constexpr Scope concrete_scopes[] = {Scope::Constant, Scope::Input, Scope::Output,
                                                 Scope::Internal};

            constexpr Lifetime concrete_lifetimes[] = {Lifetime::Immutable, Lifetime::Invocation,
                                                       Lifetime::Session};

            for (Scope target_scope : concrete_scopes) {
                for (Lifetime target_lifetime : concrete_lifetimes) {
                    int best = -1;
                    int count = 0;

                    for (const auto &scheme : police.schemes) {
                        if (scheme.scope != Scope::Any && scheme.scope != target_scope) {
                            continue;
                        }

                        if (scheme.lifetime != Lifetime::Any &&
                            scheme.lifetime != target_lifetime) {
                            continue;
                        }

                        const int specificity = (scheme.scope == target_scope ? 1 : 0) +
                                                (scheme.lifetime == target_lifetime ? 1 : 0);

                        if (specificity > best) {
                            best = specificity;
                            count = 1;
                        } else if (specificity == best) {
                            ++count;
                        }
                    }

                    if (count > 1) {
                        raise(Errors::Manifest::Scheme::Wildcard::Ambiguous, path);
                    }
                }
            }

            if (police.backend) {
                if (police.backend->device.empty()) {
                    raise(Errors::Manifest::Backend::Device::Missing, path);
                }

                if (police.backend->ordinal < 0) {
                    raise(Errors::Manifest::Backend::Ordinal::Negative, path);
                }
            }
        }
    }

    Police police_for(const Manifest &manifest, const std::string &path) {
        const auto &policies = manifest.policies;

        if (policies.empty()) {
            raise(Errors::Manifest::Policy::Root::Missing);
        }

        Police result = policies.front().second;
        std::vector<const std::pair<std::string, Police> *> matched_entries;

        for (std::size_t i = 1; i < policies.size(); i++) {
            const auto &entry = policies[i];

            if (matches(path, entry.first)) {
                matched_entries.push_back(&entry);
            }
        }

        std::stable_sort(matched_entries.begin(), matched_entries.end(),
                         [](const auto *left, const auto *right) {
                             return path_specificity(left->first) < path_specificity(right->first);
                         });

        for (const auto *entry : matched_entries) {
            merge(result, entry->second);
        }

        return result;
    }

    Mln::Sdk::Codec &parameter_codec(const Manifest &manifest, const std::string &name) {
        auto try_format = [&](const Mln::Sdk::Codec::Format &holder) -> Mln::Sdk::Codec * {
            if (!holder)
                return nullptr;
            Mln::Sdk::Codec &format = holder.get();

            if (format.name != name) {
                return nullptr;
            }

            auto *constant = dynamic_cast<Mln::Sdk::Codec *>(&format);

            if (!constant) {
                raise(Errors::Manifest::Format::Immutable::Unsupported, name);
            }

            return constant;
        };

        for (const auto &registered : manifest.formats) {
            if (auto *found = try_format(registered)) {
                return *found;
            }
        }

        for (const auto &[path, police] : manifest.policies) {
            (void)path;

            for (const auto &scheme : police.schemes) {
                if (auto *found = try_format(scheme.format)) {
                    return *found;
                }
            }
        }

        raise(Errors::Manifest::Format::Constant::Missing, name);
    }
}
