import argparse
from pathlib import Path
import subprocess

import numpy as np


def reference(query, key, value, decay, beta, state):
    query = query / np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + 1e-6)
    key = key / np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + 1e-6)
    query = query / np.sqrt(np.float32(query.shape[-1]))
    memory = state.copy()
    outputs = []
    for q, k, v, g, b in zip(query, key, value, decay, beta):
        memory *= np.exp(g)[:, None, None]
        recalled = np.sum(memory * k[:, :, None], axis=1)
        delta = (v - recalled) * b[:, None]
        memory += k[:, :, None] * delta[:, None, :]
        outputs.append(np.sum(memory * q[:, :, None], axis=1))
    return np.concatenate([np.asarray(outputs).ravel(), memory.ravel()])


def convolution_reference(value, kernel, state):
    window = np.concatenate([state, value.T], axis=1)
    output = np.empty_like(value)
    for token in range(value.shape[0]):
        convolved = np.sum(window[:, token:token + kernel.shape[1]] * kernel, axis=1)
        output[token] = convolved / (1.0 + np.exp(-convolved))
    return np.concatenate([output.ravel(), window[:, -state.shape[1]:].ravel()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runner")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(791)
    for tokens in (1, 8, 64):
        q, k, v = [rng.normal(size=(tokens, 16, 128)).astype(np.float32) for _ in range(3)]
        decay = -rng.uniform(0.01, 1.0, (tokens, 16)).astype(np.float32)
        beta = rng.uniform(0.1, 0.9, (tokens, 16)).astype(np.float32)
        state = rng.normal(0, 0.03, (16, 128, 128)).astype(np.float32)
        inputs = [q, k, v, decay, beta, state]
        expected = reference(*inputs)
        paths = []
        for index, data in enumerate(inputs):
            path = args.directory / f"input-{tokens}-{index}.npy"
            np.save(path, data)
            paths.append(path)
        np.save(args.directory / f"expected-{tokens}.npy", expected)
        convolution_input = rng.normal(0, 0.1, (tokens, 6144)).astype(np.float32)
        convolution_kernel = rng.normal(0, 0.1, (6144, 4)).astype(np.float32)
        convolution_state = rng.normal(0, 0.1, (6144, 3)).astype(np.float32)
        convolution_inputs = [convolution_input, convolution_kernel, convolution_state]
        convolution_paths = []
        for index, data in enumerate(convolution_inputs):
            path = args.directory / f"convolution-input-{tokens}-{index}.npy"
            np.save(path, data)
            convolution_paths.append(path)
        np.save(args.directory / f"convolution-expected-{tokens}.npy",
                convolution_reference(*convolution_inputs))
        if args.prepare_only:
            continue
        for target, device in (("local", "local-task"), ("vulkan", "vulkan")):
            output = args.directory / f"{target}-{tokens}.output.npy"
            subprocess.run(
                [args.runner, f"--module={args.directory / f'{target}-{tokens}.vmfb'}",
                 f"--device={device}", "--function=delta_reduce", f"--input=@{paths[-1]}",
                 f"--output=@{output}"], check=True, timeout=30,
            )
            np.testing.assert_allclose(np.load(output), np.sum(state, axis=1),
                                       rtol=2e-4, atol=3e-6)
            print(f"{target}: standalone reduction matches", flush=True)
            subprocess.run(
                [args.runner, f"--module={args.directory / f'{target}-{tokens}.vmfb'}",
                 f"--device={device}", "--function=delta_update",
                 *[f"--input=@{path}" for path in paths], f"--output=@{output}"],
                check=True, timeout=30,
            )
            actual = np.load(output).ravel()
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=3e-6)
            print(f"{target}: tokens={tokens}, output and state match; "
                  f"max error={np.max(np.abs(actual - expected)):.3g}", flush=True)
            convolution_output = args.directory / f"{target}-convolution-{tokens}.output.npy"
            subprocess.run(
                [args.runner, f"--module={args.directory / f'{target}-{tokens}.vmfb'}",
                 f"--device={device}", "--function=delta_convolution",
                 *[f"--input=@{path}" for path in convolution_paths],
                 f"--output=@{convolution_output}"], check=True, timeout=30,
            )
            convolution_actual = np.load(convolution_output).ravel()
            convolution_expected = np.load(
                args.directory / f"convolution-expected-{tokens}.npy").ravel()
            np.testing.assert_allclose(convolution_actual, convolution_expected,
                                       rtol=3e-3, atol=3e-4)
            print(f"{target}: causal convolution tokens={tokens} matches; "
                  f"max error={np.max(np.abs(convolution_actual - convolution_expected)):.3g}",
                  flush=True)


if __name__ == "__main__":
    main()
