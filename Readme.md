# Annotation project pipeline
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Settings
### Caching
- If you want to develop using the pipeline as api endpoint, we recommend caching the results, to reduce the OpenAI requests.
- You can toggle caching in the [environment file](.env) 
  - Note: It is required to restart the service to enable the change 

The caching is simply based on the provided `document_id`. 
Multiple requests with the same `document_id` will then return the same result, independent of the actual content.

You can delete all cached results by deleting the [uploads folder](uploads).


## Development setup
You can choose between [Pycharm](https://www.jetbrains.com/de-de/pycharm/) (Professional Edition) and [VsCode](https://code.visualstudio.com/) as a development environment. 

### Run Project

Clone the [setup](https://github.com/Databases-and-Informationsystems/setup) project and follow the instructions

### Git hooks
#### Update git hooks settings
```bash
git config core.hooksPath ./.githooks
```
```bash
chmod 744 ./.githooks/pre-commit
chmod 744 ./.githooks/commit-msg
```

### Select Interpreter
Same setup as [in the api project](https://github.com/Databases-and-Informationsystems/api?tab=readme-ov-file#select-interpreter), but with different image/container

### Code Formatting
[See api project](https://github.com/Databases-and-Informationsystems/api?tab=readme-ov-file#code-formatting)
