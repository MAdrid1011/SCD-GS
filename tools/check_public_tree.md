# check_public_tree.py

## External Interface

```bash
python tools/check_public_tree.py
```

The command checks Git-tracked paths, private-file ignore rules, required
mechanism evidence and package entry points in both the index and working
tree, and retired terminology or dead interfaces in public documentation and
source. It also rejects retained legacy experiment data and logs even when
they are untracked. It exits nonzero with an actionable list of violations.

## Internal Helpers

`tracked_files()` reads the Git index. `run()` executes a repository-local
Git query. `main()` applies path, evidence, and terminology checks.
