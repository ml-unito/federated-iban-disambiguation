# Synthetic Dataset Generator
To generate a synthetic dataset, you can first edit the configuration file `./config/parameters.json`, where:
- `num_iban`: sets the number of unique IBANs that are generated. Controls the number of entries in the dataset;
- `min_range_entry`: controls the minimum number of transactions for each IBAN;
- `max_range_entry`: controls the maximum number of transactions for each IBAN;
- `min_range_holders`: controls the minimum number of different entities associated with each IBAN, if it is shared;
- `max_range_holders`: controls the maximum number of different entities associated with each IBAN, if it is shared (must be less than max_range_entry);
- `V`: variability factor. controls the modification factor of company parts (between 0 and 1);
- `EDIT`: controls the approximation with the corporate parts starting from a known list;
- `T`: (temperature) controls the distortion factor of names and addresses (between 0 and 1);
- `C`: (Changeable factor) controls the factor for adding white spaces and removing words in names and addresses (between 0 and 1).

Then, to run the script, you can execute the following command:
```bash
uv run dataset_generator.py [OPTIONS]
```
where `[OPTIONS]`can be `--show-preview / --no-show-preview`, to show a preview of the results (default: False).

The generated dataset are saved in the `./output/` directory.