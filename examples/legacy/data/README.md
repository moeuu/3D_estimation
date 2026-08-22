# Historical research data

This directory preserves the complete `data/` tree from the former
`moeuu/Radiation_distribution_machine_learning` repository at commit
`d07b4eadf800813aaab1b11d65e10bd0e215e092`.

| Directory | Files | Bytes | Contents |
| --- | ---: | ---: | --- |
| `GT/` | 4 | 8,063 | Shield-aware and no-shield measurement tables |
| `cor_img/` | 214 | 1,415,436 | Correlation images |
| `img/` | 214 | 1,638,972 | Generated result images |
| `rad_cnt/` | 369 | 204,278,653 | Radiation-count CSV files |
| `test/` | 3 | 13,064 | Test CSV files |
| **Total** | **804** | **207,354,188** | Original files, byte-for-byte |

The migration retains original filenames, directory structure, line endings,
empty files, partial files, and marker columns. This preservation guarantees
historical fidelity, not data quality or completeness.

These artifacts are not MeasurementLog inputs and are not consumed by the
maintained estimator example. Use `../shield_aware_surface_mle.ipynb` for the
current executable workflow.
