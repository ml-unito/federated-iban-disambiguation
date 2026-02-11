# Clique Analysis

The script `clique-analysis.py` generate a clique analysis dot file for the given IBAN. In particular, you can use the following command:

```bash
uv run clique-analysis.py IBAN [OPTIONS]
```
where:
- `IBAN`: is the IBAN to analyze;
- `[OPTIONS]`:
    - `--fname STRING`: the CSV file containing the data;
    - `--output-format STRING`: output format can be _nx_ or _dot_.

The output is save in the `./output/` directory.
