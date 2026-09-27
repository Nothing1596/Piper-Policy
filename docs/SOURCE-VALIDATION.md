# Published source validation

The repository was tested after import into its new two-project layout on 2026-09-27:

| Suite | Result | Scope |
|---|---|---|
| Upper-level | 287 passed, 3 skipped | Imported source using the packaged Python dependency environment |
| Lower-level | 44 passed, 5 skipped | SDK contract tests skipped because no external SDK was selected |

The first lower-level run passed the dynamic grasp/contact/lift assertions but failed its final check for a historical `assets/mujoco/grasp-debug.json`. That runtime artifact is deliberately excluded from Git. The test now writes measurements and trajectory from the current run into pytest's temporary directory; it no longer relies on a previous success flag. The complete lower-level suite passed afterward. Production motion code and physical thresholds were not changed.

These checks are separate from the complete installer validation in the release's PACKAGE-VALIDATION.json. No cloud model or physical hardware was exercised during this source import. A Starlette/httpx deprecation warning remains in the lower-level test environment.
