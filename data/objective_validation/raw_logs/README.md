# Raw objective validation logs

This folder contains the raw objective validation logs used to generate the processed objective validation results.

Due to the large number of frame-level and event-level log files, the raw logs are released as compressed archives rather than expanded folders.

## Compressed raw log archives

The raw objective validation logs include:

- `04_trial_logs.zip`: frame-level trial logs
- `05_contact_event_logs.zip`: contact-event logs
- `06_trial_summary.zip`: trial-level summary files
- `parameter_settings.csv`: runtime and analysis parameter settings
- `threshold_settings.csv`: threshold settings used for metric computation
- `trial_manifest.csv`: trial manifest file

After decompression, the archive files correspond to the original raw-log folders used in the analysis pipeline.

## Notes

All raw logs were generated during controlled objective validation sessions under the same hardware, software, and runtime parameter settings.

The processed results derived from these logs are available in:

- `data/objective_validation/processed_results/`
