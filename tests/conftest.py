"""Shared fixtures for the test suite."""

import pandas as pd
import pytest

from jobfunnel.generate import generate_sample_data


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    return generate_sample_data(n_rows=120, seed=7)
