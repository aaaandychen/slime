# Archive

Superseded files kept for reference. These are **not** used by the current
training pipeline and may have broken imports (they reference each other,
not the active code).

## Files

| File | Status | Replaced by |
|------|--------|-------------|
| `slime_api.py` | Superseded | `../adapter.py` (`DataAgentAdapter`) |
| `queries.jsonl` | Superseded | `../queries_labeled.jsonl` (with ground-truth labels) |
| `REFACTOR_ANALYSIS.md` | Historical | Refactoring is complete; see `../README.md` for current architecture |
| `test_abort_resume.py` | Stale | Uses `slime_api` (above); needs rewrite to use `AdapterService` if revived |

## Why keep them?

- `slime_api.py` — reference for the old manual HTTP/token management approach;
  useful if you need to understand what `adapter.py` replaced.
- `queries.jsonl` — the original 100 unlabeled queries; useful as a source of
  query phrasings if you extend `generate_training_data.py`.
- `REFACTOR_ANALYSIS.md` — documents the design analysis that led to the
  current architecture; useful if you need to revisit a design decision.
- `test_abort_resume.py` — the original abort/resume integration test; if you
  want to revive it, rewrite the `slime_api` imports to use `AdapterService`
  + `adapter.open_session/finish_session`.
