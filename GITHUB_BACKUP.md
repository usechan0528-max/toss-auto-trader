# Source backup

Source snapshot uploaded on 2026-10-30. Local API credentials, account settings, trading records, logs, screenshots, downloaded market data, and generated models are excluded.

## Validation at upload

`python -m unittest discover -s tests -q` reported 153 tests and 2 import errors:

- `test_app_persistence` imports `atomic_write_json`, which is absent from the current `app.py`.
- `test_chart_helpers` imports `dataframe_selected_rows`, which is absent from the current `app.py`.

This snapshot preserves the existing program without repairing these failures. No live orders were submitted to validate this backup. Use `.env.example` for configuration; live trading is disabled in the example.
