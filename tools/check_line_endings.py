"""Check tracked and untracked non-ignored files against Git's EOL rules."""

from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "git",
            "-c", f"safe.directory={root.as_posix()}",
            "-c", "core.excludesFile=NUL" if sys.platform == "win32" else "core.excludesFile=/dev/null",
            "ls-files", "--cached", "--others", "--exclude-standard", "--eol", "-z",
        ],
        cwd=root,
        capture_output=True,
        check=True,
    )
    problems = []
    checked = 0
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        fields = metadata.decode("ascii").split()
        if "attr/-text" in fields or "w/-text" in fields:
            continue
        expected = next((field[4:] for field in fields if field.startswith("eol=")), None)
        if expected not in {"lf", "crlf"}:
            continue
        path = raw_path.decode("utf-8")
        full_path = root / path
        if not full_path.is_file():
            continue
        data = full_path.read_bytes()
        checked += 1
        if expected == "lf":
            mismatch = b"\r" in data
        else:
            remainder = data.replace(b"\r\n", b"")
            mismatch = b"\n" in remainder or b"\r" in remainder
        if mismatch:
            problems.append(f"{path}: expected {expected.upper()} line endings")
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {checked} text files match Git's line-ending rules.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
