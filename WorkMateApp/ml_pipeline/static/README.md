# WorkMate Static Datasets Folder

This folder is for additional datasets that can be used with the WorkMate ML Pipeline.

## Purpose

Place any additional household survey datasets here for training or testing the WorkMate vulnerability prediction system.

## Supported Formats

- CSV files (.csv) - Household survey data
- Excel files (.xlsx, .xls) - Data dictionaries or formatted datasets
- JSON files (.json) - Configuration or metadata files

## Usage for Production

The main WorkMate model uses data from the `../Datasets/` folder by default. This static folder is available for:

- Additional training datasets
- Custom regional data
- Testing datasets
- Configuration files for specific deployments

## Integration

The WorkMate ML pipeline will automatically detect and process any compatible files placed in this folder when using the enhanced dataset reading functions.
