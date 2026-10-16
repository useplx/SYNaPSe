# SYNaPSe

SYNaPSe is the PLeX project and package manager. It resolves dependencies,
selects toolchains, builds packages, and produces deployable `.mln` artifacts.

Use SYNaPSe to keep model source, build configuration, target hardware, and
application composition separate.

## Requirements

- Linux
- CMake 3.21 or newer
- Ninja
- a C++23 compiler
- Git
- a MyeLiN CMake source tree
- wasi-sdk when building Orchestration packages written in C or C++
- any language toolchains required by the selected packages

## Build

```bash
cmake -S . -B build -G Ninja \
  -DMLN_ROOT=/path/to/MyeLiN \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 1
```

LLVM CPU compilation is always available. Enable optional compiler targets at
configure time with `-DSYN_ENABLE_CUDA=ON` and/or
`-DSYN_ENABLE_VULKAN=ON`. These options control emitted artifacts; the matching
MyeLiN runtime still needs its corresponding `MLN_ENABLE_*` driver.

The executable is `build/synps`. Install it with:

```bash
cmake --install build --prefix ~/.local
python tests/install/check.py build
```

The installed SDK and compiler resources are resolved relative to `synps`.
The installation check relocates that prefix, rejects references to the
original source trees, and builds Native and Graph artifacts. Add `--run` when
the MyeLiN Python binding and runtime library are available to execute both.

## Create a project

```bash
synps new my-app
cd my-app
```

To initialize the current directory instead:

```bash
synps init
```

A project is configured by `synps.toml`:

```toml
[package]
name = "my-app"
version = "0.1.0"
description = "My PLeX application"

```

The initial project has no executable entrypoint. Add the packages and
entrypoints required by the application before building it.

MyeLiN is a CMake SDK/toolchain dependency, not a SYNaPSe package. Configure
SYNaPSe with `-DMLN_ROOT=/path/to/MyeLiN` or set `MLN_ROOT` when invoking it.
The install step installs both `synps` and its Compiler library. Set
`MLN_COMPILER_API` only to select another ABI-compatible Compiler build.

## Add dependencies

Published package:

```bash
synps add qwen3.5 --git https://github.com/useplx/Qwen3.5.git
```

Local package under development:

```bash
synps add my-package --path ../my-package
```

Remove a dependency:

```bash
synps remove my-package
```

Resolve dependencies and write `synps.lock`:

```bash
synps install
```

Use `synps update` when you intentionally want to resolve dependencies again.

## Inspect a project

```bash
synps graph
synps inspect
```

- `graph` lists packages and capability bindings.
- `inspect` prints the effective build plan after defaults and package
  configuration have been applied.

These commands do not compile the model.

## Build

```bash
synps build
synps build --release
synps build --jobs 8 --release
synps build --jobs 8 --weight-memory-mib 2048 --release
```

Independent build-DAG nodes run concurrently. `--jobs N` bounds that
parallelism (the default is the host concurrency capped at 8); it does not
change IREE Stream scheduling inside a compiled graph.
The jobs limit also reaches independent weight-root conversion. The optional
`--weight-memory-mib` bound (2048 by default) prevents concurrent decoded-f32
temporaries from growing without a build-time memory budget.

The output is written to `build/<package-name>.mln`. Repeated builds reuse
unchanged work. Remove generated artifacts with:

```bash
synps clean
```

Graph devices are selected per path in `policies`. Native and Orchestration targets are
selected independently by their build toolchains; there is no global target
override.

Use another manifest without renaming it when more than the device changes:

```bash
synps build --project deployment.cuda.toml --release
```

## Configure a package

A dependency can publish typed variables. Set them from the root project:

```toml
[packages."qwen3.5"]
context = 1024
max_new_tokens = 224
prefill_chunk = 32
```

Unknown variables and invalid values are rejected before compilation.

When a variable is consumed by compiled source, a package can publish its
preprocessor name and limit it to the facets that actually use it:

```toml
[package.variables]
top_k = { type = "uint32", default = 5, define = "RAG_TOP_K", facets = ["augmentation"] }
```

Changing `top_k` then rebuilds `augmentation`, but does not invalidate unrelated
Graph or Native artifacts from the same package.

## Select devices and formats

Policies address stable model paths:

```toml
[policies."model"]
dtype = "f16"

[policies."model".backend]
device = "local"

[[policies."model.language_model.layers.*.mlp.down_proj.weight".schemes]]
format = "q4k"
volatile = true
```

Broad policies provide defaults. More specific paths override only the selected
part of the graph. Weight schemes infer `constant`/`immutable`/`none` when those
fields are omitted. `volatile = true` allows MyeLiN to discard clean host pages
after invocation while retaining valid package mappings and device buffers.

Graphs spanning more than one device may declare memory relationships that are
guaranteed by the deployment hardware and drivers:

```toml
[[topology.links]]
source = { device = "local", ordinal = 0 }
target = { device = "vulkan", ordinal = 0 }
transparent_access = true
unified_memory = false
```

Links are directional. Declare the reverse direction separately when it is
also guaranteed. `transparent_access` lets IREE share an allocation directly
and elide copies; it must not be used merely because two devices are present.
Without a sufficient topology, IREE rejects cross-device allocations instead
of emitting an artifact with unsafe memory assumptions.

## Compose packages

Register dependencies and select a package entrypoint by alias. `forward` and
the Graph/Orchestration bindings are derived from the package contract:

```toml
[package]
name = "assistant"

[entrypoint]
package = "qwen"

[dependencies]
qwen = { git = "https://github.com/useplx/Qwen3.5.git", version = "v1.0.0" }
```

`uses` links the output of another build package from the same project:

```toml
[build.native.index]
toolchain = "cpp"
sources = ["src/index.cpp"]
uses = ["index-core"]
```

A package may publish more than one artifact for the same runtime by adding a
stable label below the canonical domain:

```toml
[build.native.tokenizer]
toolchain = "cpp"
sources = ["src/tokenizer.cpp"]

[build.native.search]
toolchain = "rust"
sources = ["src/search/Cargo.toml"]
```

The labels (`tokenizer` and `search`) are the names referenced by `uses`,
`extends`, hooks, and variable facets. Nested labels are joined with dots.

Graph behavior can currently be attached through `[[hooks]]`:

```toml
[[hooks]]
path = "model.language_model.head"
function = "rag:suppress-vision-tokens"
```

An Orchestration package can transform the Buffer entering a represented Runtime path:

```toml
[[hooks]]
package = "qwen"
path = "qwen3.5"
function = "rag:augmentation"
```

Hooks apply to represented paths. Direct C++ calls do not cross the package
router and therefore cannot be intercepted by a manifest.

## Commands

| Command | Purpose |
|---|---|
| `synps add` | Add a dependency. |
| `synps build` | Build the project and produce a `.mln`. |
| `synps clean` | Remove build artifacts. |
| `synps graph` | Show dependencies and capability bindings. |
| `synps help` | Show command help. |
| `synps init` | Initialize the current directory. |
| `synps inspect` | Print the resolved build plan. |
| `synps install` | Resolve dependencies and write the lockfile. |
| `synps new` | Create a project directory. |
| `synps remove` | Remove a dependency. |
| `synps update` | Resolve dependencies again. |

`synps --version` prints the installed CLI release.

Run `synps help <command>` for the accepted arguments and options.

## Documentation

Full documentation lives at **[https://useplx.github.io/tools/synapse/](https://useplx.github.io/tools/synapse/)**.

- [Getting started](https://useplx.github.io/guide/getting-started/)
- [Build workflow](https://useplx.github.io/notes/build-workflow/)
- [Packages](https://useplx.github.io/packages/) and [build projects](https://useplx.github.io/guide/build-a-project/)
- [Toolchains](https://useplx.github.io/notes/toolchains/)
- [Create a package](https://useplx.github.io/guide/create-a-package/)
- [Compose packages](https://useplx.github.io/guide/compose-packages/)
- [Cross-compile](https://useplx.github.io/guide/cross-compile/)
- Reference: [manifest](https://useplx.github.io/api/synps-toml/) · [CLI](https://useplx.github.io/api/synps-cli/) · [diagnostics](https://useplx.github.io/notes/diagnostics/)


## Current limitations

- Registry dependencies are not available; use local paths or Git.
- CLI, manifest, compiler ABI, and diagnostic contracts follow semantic
  versioning from v1.0 onward.

## License

SYNaPSe is source-available under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Noncommercial use is permitted
without charge. Commercial use requires a separate agreement with the copyright
holder.
