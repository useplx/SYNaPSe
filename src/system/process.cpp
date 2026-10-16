#include "syn/system/process.hpp"
#include "syn/error.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <string_view>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace Syn::System {
    namespace {
        class Descriptor {
            public:
                Descriptor() = default;
                explicit Descriptor(int value) : value_(value) {}
                ~Descriptor() { reset(); }

                Descriptor(const Descriptor &) = delete;
                Descriptor &operator=(const Descriptor &) = delete;
                Descriptor(Descriptor &&other) noexcept : value_(other.release()) {}
                Descriptor &operator=(Descriptor &&other) noexcept {
                    if (this != &other)
                        reset(other.release());
                    return *this;
                }

                int get() const noexcept { return value_; }
                int release() noexcept {
                    const int value = value_;
                    value_ = -1;
                    return value;
                }
                void reset(int value = -1) noexcept {
                    if (value_ >= 0)
                        ::close(value_);
                    value_ = value;
                }

            private:
                int value_ = -1;
        };

        std::array<Descriptor, 2> pipe_pair() {
            int descriptors[2];
            if (::pipe(descriptors) != 0)
                Errors::raise(Error::Process::Pipe::Failed, std::strerror(errno));
            return {Descriptor(descriptors[0]), Descriptor(descriptors[1])};
        }

        void make_non_blocking(int descriptor) {
            const int flags = ::fcntl(descriptor, F_GETFL, 0);
            if (flags >= 0)
                ::fcntl(descriptor, F_SETFL, flags | O_NONBLOCK);
        }

        bool drain(int descriptor, std::string &destination) {
            std::array<char, 8192> buffer{};
            while (true) {
                const ssize_t count = ::read(descriptor, buffer.data(), buffer.size());
                if (count > 0) {
                    destination.append(buffer.data(), static_cast<std::size_t>(count));
                    continue;
                }
                if (count == 0)
                    return false;
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                    return true;
                if (errno == EINTR)
                    continue;
                return false;
            }
        }
    }

    Process::Result run(const Command &command) {
        if (command.executable.empty())
            Errors::raise(Error::Process::Executable::Empty);

        std::vector<char *> arguments;
        arguments.reserve(command.arguments.size() + 2);
        arguments.push_back(const_cast<char *>(command.executable.c_str()));
        for (const std::string &argument : command.arguments)
            arguments.push_back(const_cast<char *>(argument.c_str()));
        arguments.push_back(nullptr);

        std::vector<std::string> overrides;
        overrides.reserve(command.environment.size());
        for (const auto &[name, value] : command.environment)
            overrides.push_back(name + "=" + value);

        std::vector<char *> environment;
        for (char **entry = ::environ; entry && *entry; ++entry) {
            const std::string_view current(*entry);
            const std::string_view name = current.substr(0, current.find('='));
            const bool overridden =
                std::any_of(command.environment.begin(), command.environment.end(),
                            [&](const auto &pair) { return pair.first == name; });
            if (!overridden)
                environment.push_back(*entry);
        }
        for (std::string &entry : overrides)
            environment.push_back(entry.data());
        environment.push_back(nullptr);

        auto output_pipe = pipe_pair();
        auto error_pipe = pipe_pair();
        const pid_t child = ::fork();
        if (child < 0)
            Errors::raise(Error::Process::Start::Failed, std::strerror(errno));

        if (child == 0) {
            ::dup2(output_pipe[1].get(), STDOUT_FILENO);
            ::dup2(error_pipe[1].get(), STDERR_FILENO);
            output_pipe[0].reset();
            output_pipe[1].reset();
            error_pipe[0].reset();
            error_pipe[1].reset();

            if (!command.working_directory.empty() &&
                ::chdir(command.working_directory.c_str()) != 0)
                _exit(126);

            ::execvpe(command.executable.c_str(), arguments.data(), environment.data());
            _exit(errno == ENOENT ? 127 : 126);
        }

        output_pipe[1].reset();
        error_pipe[1].reset();
        make_non_blocking(output_pipe[0].get());
        make_non_blocking(error_pipe[0].get());

        Process::Result result;
        bool output_open = true;
        bool error_open = true;
        while (output_open || error_open) {
            std::array<pollfd, 2> descriptors{{
                {output_open ? output_pipe[0].get() : -1, static_cast<short>(POLLIN | POLLHUP), 0},
                {error_open ? error_pipe[0].get() : -1, static_cast<short>(POLLIN | POLLHUP), 0},
            }};
            if (::poll(descriptors.data(), descriptors.size(), -1) < 0) {
                if (errno == EINTR)
                    continue;
                break;
            }
            if (output_open && descriptors[0].revents)
                output_open = drain(output_pipe[0].get(), result.output);
            if (error_open && descriptors[1].revents)
                error_open = drain(error_pipe[0].get(), result.error);
        }

        int status = 0;
        while (::waitpid(child, &status, 0) < 0) {
            if (errno == EINTR)
                continue;
            Errors::raise(Error::Process::Wait::Failed, command.executable, std::strerror(errno));
        }
        if (WIFEXITED(status))
            result.exit_code = WEXITSTATUS(status);
        else if (WIFSIGNALED(status))
            result.exit_code = 128 + WTERMSIG(status);
        return result;
    }

    Process::Result run_checked(const Command &command, std::string_view operation) {
        Process::Result result = run(command);
        if (result.exit_code == 0)
            return result;
        Errors::raise(Error::Process::Failed,
                      operation.empty() ? std::string_view("process") : operation, result.exit_code,
                      result.error.empty() ? std::string{} : ": " + result.error);
    }
}
