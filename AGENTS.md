# Repository editing rules

- Save text files with LF line endings, including Python, Markdown, JSON, CSV,
  and Jupyter notebooks. Use CRLF only for `.bat` and `.cmd` scripts, as defined
  in `.gitattributes` and `.editorconfig`.
- Preserve existing encoding and content when fixing line endings. Do not
  rewrite notebook JSON or clear notebook outputs just to normalize newlines.
- After each code edit and before finishing, run
  `python tools/check_line_endings.py`. Fix any reported mismatch and rerun the
  check. This checks tracked and untracked non-ignored files.
- When writing files programmatically, explicitly choose the newline convention
  instead of relying on the Windows default (for Python, use `newline="\n"`).
