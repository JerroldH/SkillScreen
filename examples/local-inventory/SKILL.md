---
name: local-inventory
description: Count file extensions in a directory selected by the user.
---

# Local inventory

Use this skill to summarize the kinds of files in one directory. Ask for the
directory to summarize if the user has not provided one.

The helper in scripts/inventory.py counts the immediate files by extension and
prints JSON. It does not read file contents, follow links, access the network, or
modify files. The output format is described in references/output.md.

This package is a small, project-authored example for SkillScreen. Its files can
be inspected as text; scanning the package does not run the helper.
