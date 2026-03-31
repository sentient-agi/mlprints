# OML Library
The Open, Monetizable, Loyal (OML) library provides the state-of-the-art framework for fingerprinting and controlling LLMs.

## Setting up the environment
The OML library is a Python package, that can be easily installed from the source code provided in this repo. We recommend using a virtual environment manager, e.g. `conda`:

```
conda create -n oml311 python=3.11
conda activate oml311
pip install -e . # editable install helps in development process
```

## Script Usage
### Fingerprint Generation
Fingerprint generation is handled by the `generate_fingerprints.py` script. Example configuration files are provided under `configs` directory.
```
python scripts/generate_fingerprints.py configs/rofl_config.yaml
```

Optionally, if not using a job scheduling manager (e.g. SLURM), you can pipe the outputs to a log file (after `mkdir logs`):
```
nohup python scripts/generate_fingerprints.py configs/rofl_config.yaml &> logs/my_log_file.txt
```

### Fingerprint Strength Measurement
Measuring the strength (hit rate) of fingerprints can be achieved by using `measure_fp_strength.py`. Example configuration files are provided under `configs` directory. In your test run, following this guide, don't forget to change the `fingerprint_dir` to whatever was produced by the generation command above in the previous step.
```
python scripts/measure_fp_strength.py configs/strength_config.yaml
```

### Measurement Summary
The generation of fingerprints as well as strength measurements yield output files that are verbose enough to facilitate further study. However, for succinct reporting and a bird's eye view on the results, essential summarization capabilities are provided in `summarize_fp_measurements.py`. Simply provide the fingerprint directory and the measurement types that you would like to summarize:
```
python scripts/summarize_fp_measurements.py <path_to_fingerprint_dir> --strength
```

## Development Guide
### Software Architecture
The OML library has been architected to be modular and flexible in its mission of capturing the state of the art in fingerprinting in a long term manner. Therefore, clear separation of concerns has been embodied in the main submodules of the package:
```
oml
|
+- attack (TBD)
+- common (TBD)
+- control (TBD)
+- fingerprint
+- measure
```

When implementing a new method in the package, please use the relevant directory and feel free to utilize/contribute to the `common` module.

### Dependencies
When adding new dependencies, update the `pyproject.toml` at the same commit where the new dependency is introduced. Try to refrain from manually installing the package, instead simply do a clear recreation of the environment to test whether pip can easily install the new dependency or not:
```
conda deactivate # if not currently in base environment
conda remove -n oml311 --all

conda create -n oml311 python=3.11
conda activate oml311
pip install -e .
```

Feel free to add new dependencies as your needs progress. Currently, the `notebook` and `ipywidgets` dependencies provide necessary capabilities to experiment in Jupyter notebooks. At some point, we might consider pulling these dependencies as an option, e.g. `pip install oml[dev]`.