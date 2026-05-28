# Analysis code

This folder contains the analysis scripts used for the digital shadow-puppetry interaction system validation.

## Scripts

- `objective_validation_analysis.py`: analysis script for objective system validation logs and processed results.
- `user_study_analysis.py`: analysis script for subjective user-study questionnaire data.
- `expert_appraisal_analysis.py`: analysis script for expert rating and interview-summary data.

## Notes

The released scripts use relative paths whenever possible. Local absolute paths from the original analysis environment were removed or generalized before release.

The objective validation analysis uses the files in:

- `data/objective_validation/raw_logs/`
- `data/objective_validation/processed_results/`

The user-study analysis uses the files in:

- `data/user_study/`

The expert-appraisal analysis uses the files in:

- `data/expert_appraisal/`
