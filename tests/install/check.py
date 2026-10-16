"""Build an artifact with the installed, relocated compiler and its SDK."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("build", type=Path)
    parser.add_argument("--run", action="store_true", help="also execute using myelin-runtime")
    args = parser.parse_args()
    fixture = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="synps-install-") as directory:
        root = Path(directory)
        prefix = root / "original"
        subprocess.run(["cmake", "--install", str(args.build.resolve()),
                        "--prefix", str(prefix)], check=True, stdout=subprocess.DEVNULL)
        relocated = root / "relocated"
        prefix.rename(relocated)
        project = root / "project"
        shutil.copytree(fixture, project, ignore=shutil.ignore_patterns("__pycache__", "check.py"))
        environment = os.environ.copy()
        for name in ("MLN_ROOT", "MLN_COMPILER_API"):
            environment.pop(name, None)

        compiler = shutil.which("c++")
        if compiler is None:
            raise RuntimeError("c++ compiler is required")
        forbidden = [str(fixture.parents[1])]
        for line in (args.build / "CMakeCache.txt").read_text().splitlines():
            if line.startswith("MLN_ROOT:PATH="):
                forbidden.append(line.partition("=")[2])
        tools = root / "tools"
        tools.mkdir()
        wrapper = tools / "c++"
        wrapper.write_text(
            "#!/usr/bin/env python3\nimport os, sys\n"
            f"forbidden = {forbidden!r}\n"
            "if any(path in arg for path in forbidden for arg in sys.argv[1:]):\n"
            "    sys.exit('installed compiler referenced the original source tree')\n"
            f"os.execv({compiler!r}, [{compiler!r}, *sys.argv[1:]])\n"
        )
        wrapper.chmod(0o755)
        environment["PATH"] = str(tools) + os.pathsep + environment.get("PATH", "")
        executable = relocated / "bin/synps"
        for source, name, expected in ((project, "install-check", b"\x01\x02\x03\x04"),
                                       (project / "graph", "install-graph", b"\x02\x04\x06\x08")):
            subprocess.run([str(executable), "build", "--release", "--jobs", "1"],
                           cwd=source, env=environment, check=True)
            artifact = source / "build" / f"{name}.mln"
            if not artifact.is_file():
                raise AssertionError("installed compiler did not produce the artifact")
            if args.run:
                from myelin import Runtime
                with Runtime() as runtime:
                    with runtime.session() as session:
                        session.load(artifact)
                        if session.invoke(b"\x01\x02\x03\x04") != expected:
                            raise AssertionError(f"incorrect output from {name}")


if __name__ == "__main__":
    main()
