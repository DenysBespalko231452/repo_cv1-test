# Important Declarations

### Conventions

- Dataset
- Pre-processing
- Model
- Training
- Post-processing
- Evaluation
- Package

## General Rules

To ensure `accurate` documentation and `traceable` experiments, let's agree on `unified conventions`. They don't have to be perfect, they need to be actionable and understandable. In order to keep the repository structured, let's aim to:

- Create branches when starting with any new additions

- Verify conventions before merging branches, if failed to adhere, fix first

- Merge to dev branch first, later-on to main branch

- Create a general log that links individual changes to monitor integration between: `dataset - pre-processing - model - training - post-processing - evaluation`

## Dataset

Main attribute that makes it interesting to should be start of foldername `(e.g. Bluebg for blue background)`

followed by institute (or other unique identifier) for origin tracking
`(e.g. NPEC)`

and ends in a date of addition to the repo `(e.g. 29-4-25)`

So the full name of the directory for the example dataset would be ` Bluebg-NPEC-29-4-25`

The 2 subdirectories should be `masks` and `images`

## Pre-processing

One script for each step, say `normalization` to be applied to every image would be `img_norm.py`

## Model

Versioning similar to general `semantic versioning` but in model terms:

Incremental hyperparameter changes (ONE AT A TIME) should be in `0.0.Z`

Conceptually similar architectures with added blocks should be `0.Y.0`

Big breaks in Conceptual approach should be versioned `X.0.0`

## Training

CLI-Tool / API for training:

1- Choose `train-test-eval split for each dataset` that's part of the training run. (With option to manually add files to evaluation)

2- Choose `pre-processing steps` script(s)

3- Choose `model architecture` script

4a- Choose `training` script

4b- Specify `hyperparameters` for training if non-default

5- Choose `model evaluation` script(s)

6- Choose `post-processing steps` script(s)

7- Choose `package evaluation (toolkit output)` script(s)

8- Tool asks for a `short motivation`, please fll this out to give an accurate representation to the `purpose of the training run`

For ease of use this `tool automatically generates the log` for this training run when filled out, this tool will work through CLI locally and should ideally also work with API in Docker container

## Post-processing

One script for each step, say `padding` to be applied to every predicted mask would be `pad_pred.py`

## Evaluation

Two separate types of Evaluation: `Model` & `Package`.

`Model Evalutaion` is intended to purely assess the quality of the model predictions with corresponding metrics

`Package Evaluation` is intended to monitor the output of the toolkit functions and corresponding metrics.

Scripts should be named accorindly, so `model_smape.py` vs `package_smape.py`

## Package

Versioning should be done according to `semantic versioning`:

Incremental changes should be in `0.0.Z`

Relatively big changes should be `0.Y.0`

Breaking changes that are Conceptual leaps and might deprecate parts of code should be versioned `X.0.0`
