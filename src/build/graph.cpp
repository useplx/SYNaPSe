#include "syn/build/graph.hpp"

#include "syn/error.hpp"
#include "syn/system/hash.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <future>
#include <print>
#include <set>
#include <unordered_set>

namespace Syn::Build {
    void Graph::add(Node node) {
        if (node.name.empty())
            Errors::raise(Error::Node::Name::Empty);
        if (!node.action)
            Errors::at(Error::Node::Without::Action, node.name);
        const std::string name = node.name;
        if (!nodes_.emplace(name, std::move(node)).second)
            Errors::at(Error::Node::Duplicated, name);
    }

    std::vector<std::string> Graph::order() const {
        enum class Mark { Visiting, Complete };
        std::map<std::string, Mark, std::less<>> marks;
        std::vector<std::string> result;
        const auto visit = [&](this auto &&self, const std::string &name) -> void {
            if (const auto mark = marks.find(name); mark != marks.end()) {
                if (mark->second == Mark::Visiting)
                    Errors::at(Error::Graph::Cycle, name);
                return;
            }
            const auto node = nodes_.find(name);
            if (node == nodes_.end())
                Errors::at(Error::Node::Dependency::Unknown, name);
            marks.emplace(name, Mark::Visiting);
            for (const std::string &dependency : node->second.dependencies)
                self(dependency);
            for (const std::string &dependency : node->second.order_only_dependencies)
                self(dependency);
            marks[name] = Mark::Complete;
            result.push_back(name);
        };
        for (const auto &entry : nodes_)
            visit(entry.first);
        return result;
    }

    namespace {
        // O separador precisa ser um byte que nao pode aparecer nos campos.
        // Escrito como literal (`"config\0"`), o `\0` some: o operator+ de
        // const char* mede a string com strlen e para no primeiro NUL, e o
        // fingerprint deixa de distinguir onde um campo termina e o proximo
        // comeca.
        void field(std::string &material, std::string_view tag, std::string_view value) {
            material += tag;
            material.push_back('\0');
            material += value;
            material.push_back('\0');
        }

        std::string
        fingerprint(const Node &node,
                    const std::map<std::string, std::string, std::less<>> &dependencies) {
            std::string material = "synps-build-v1\n" + node.name + "\n";
            for (const std::string &value : node.configuration) {
                field(material, "config", value);
                material.push_back('\n');
            }
            for (const std::filesystem::path &input : node.inputs) {
                if (!std::filesystem::exists(input))
                    Errors::at(Error::Node::Input::Missing, input.string());
                field(material, "input", input.lexically_normal().string());
                material += std::filesystem::is_directory(input) ? System::hash_tree(input)
                                                                 : System::hash_file(input);
                material.push_back('\n');
            }
            for (const std::string &dependency : node.dependencies) {
                field(material, "dependency", dependency);
                material += dependencies.at(dependency);
                material.push_back('\n');
            }
            return System::sha256(material);
        }

        std::string read_stamp(const std::filesystem::path &path) {
            std::ifstream input(path);
            std::string value;
            input >> value;
            return value;
        }

        bool outputs_exist(const Node &node) {
            return !node.outputs.empty() &&
                   std::ranges::all_of(node.outputs, [](const auto &output) {
                       return std::filesystem::exists(output);
                   });
        }
    } // namespace

    Executor::Result Executor::run(const Graph &graph) const {
        std::filesystem::create_directories(cache_directory_);
        Result result;
        std::map<std::string, std::string, std::less<>> fingerprints;
        const std::vector<std::string> order = graph.order();
        std::unordered_set<std::string> complete;
        std::size_t cursor = 0;

        struct Completion {
                std::string name;
                std::string fingerprint;
                bool cached = false;
        };

        while (complete.size() != order.size()) {
            std::vector<std::string> ready;
            ready.reserve(jobs_);
            for (const std::string &name : order) {
                if (complete.contains(name))
                    continue;
                const Node &node = graph.nodes().at(name);
                const auto done = [&](const std::string &dependency) {
                    return complete.contains(dependency);
                };
                if (!std::ranges::all_of(node.dependencies, done) ||
                    !std::ranges::all_of(node.order_only_dependencies, done))
                    continue;
                ready.push_back(name);
                if (ready.size() == jobs_)
                    break;
            }
            if (ready.empty())
                Errors::at(Error::Graph::Cycle, order[cursor]);

            std::vector<std::future<Completion>> running;
            running.reserve(ready.size());
            for (const std::string &name : ready) {
                const Node *node = &graph.nodes().at(name);
                const std::string current = fingerprint(*node, fingerprints);
                const std::filesystem::path stamp =
                    cache_directory_ / (System::sha256(name) + ".stamp");
                running.push_back(std::async(std::launch::async, [node, name, current, stamp] {
                    if (outputs_exist(*node) && read_stamp(stamp) == current) {
                        std::println(stderr, "cached   {}", name);
                        return Completion{name, current, true};
                    }
                    std::println(stderr, "building {}", name);
                    node->action();
                    for (const auto &output : node->outputs)
                        if (!std::filesystem::exists(output))
                            Errors::at(Error::Node::Output::Missing, output.string());
                    const std::filesystem::path temporary = stamp.string() + ".tmp";
                    {
                        std::ofstream stream(temporary, std::ios::trunc);
                        stream << current << '\n';
                    }
                    std::filesystem::rename(temporary, stamp);
                    return Completion{name, current, false};
                }));
            }
            for (auto &task : running) {
                Completion completion = task.get();
                completion.cached ? ++result.cached : ++result.executed;
                fingerprints.emplace(completion.name, std::move(completion.fingerprint));
                complete.emplace(std::move(completion.name));
                ++cursor;
            }
        }
        return result;
    }
} // namespace Syn::Build
