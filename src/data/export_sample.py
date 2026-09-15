"""Export the deterministic synthetic ICU cohort to a flat CSV file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.synthetic import VITAL_NAMES, generate_cohort


def export_sample(path: str | Path = "data/sample_icu_dataset.csv") -> Path:
    """Generate and save the demo cohort with flattened vital-sign columns."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame, vitals, _ = generate_cohort(n=280, seed=42)
    vital_frame = pd.DataFrame(
        {
            f"{name}_h{hour + 1}": vitals[:, hour, index]
            for index, name in enumerate(VITAL_NAMES)
            for hour in range(vitals.shape[1])
        }
    )
    pd.concat([frame, vital_frame], axis=1).to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    print(export_sample())