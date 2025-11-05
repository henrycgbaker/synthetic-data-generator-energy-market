"""
Unit tests for io module.
Tests file I/O operations.
"""

import os
import tempfile
import pandas as pd
import pytest
from synthetic_data_pkg.io import save_dataset, load_single_column_csv, load_empirical_series


@pytest.mark.unit
class TestSaveDataset:
    """Test saving simulation results"""

    def test_save_csv(self):
        """Test CSV saving"""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=24, freq='h'),
                'price': [50.0] * 24,
                'q_cleared': [10000.0] * 24,
            })
            
            io_config = {
                'version': 'v0',
                'add_timestamp': False,
                'save_csv': True,
                'save_pickle': False,
                'save_parquet': False,
                'save_feather': False,
                'save_preview_html': False,
                'save_meta': False,
            }
            
            paths = save_dataset(df, tmpdir, 'test', io_config, {})
            
            assert 'csv' in paths
            assert os.path.exists(paths['csv'])
            
            # Verify can load
            loaded = pd.read_csv(paths['csv'])
            assert len(loaded) == 24
            assert 'price' in loaded.columns

    def test_save_pickle(self):
        """Test pickle saving"""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=24, freq='h'),
                'price': [50.0] * 24,
            })
            
            io_config = {
                'version': 'v0',
                'add_timestamp': False,
                'save_csv': False,
                'save_pickle': True,
                'save_parquet': False,
                'save_feather': False,
                'save_preview_html': False,
                'save_meta': False,
            }
            
            paths = save_dataset(df, tmpdir, 'test', io_config, {})
            
            assert 'pickle' in paths
            assert os.path.exists(paths['pickle'])
            
            # Verify can load
            loaded = pd.read_pickle(paths['pickle'])
            assert len(loaded) == 24

    def test_save_meta(self):
        """Test metadata JSON saving"""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=24, freq='h'),
                'price': [50.0] * 24,
            })
            
            io_config = {
                'version': 'v0',
                'add_timestamp': False,
                'save_csv': False,
                'save_pickle': False,
                'save_parquet': False,
                'save_feather': False,
                'save_preview_html': False,
                'save_meta': True,
            }
            
            metadata = {'seed': 42, 'days': 1}
            paths = save_dataset(df, tmpdir, 'test', io_config, metadata)
            
            assert 'meta' in paths
            assert os.path.exists(paths['meta'])
            assert paths['meta'].endswith('.json')

    def test_add_timestamp_to_filename(self):
        """Test that add_timestamp adds timestamp to filename"""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({'x': [1, 2, 3]})
            
            io_config = {
                'version': 'v0',
                'add_timestamp': True,
                'save_csv': True,
                'save_pickle': False,
                'save_parquet': False,
                'save_feather': False,
                'save_preview_html': False,
                'save_meta': False,
            }
            
            paths = save_dataset(df, tmpdir, 'test', io_config, {})
            
            # Filename should contain timestamp
            filename = os.path.basename(paths['csv'])
            assert 'test_v0_' in filename
            # Should have date pattern
            assert any(c.isdigit() for c in filename)


@pytest.mark.unit
class TestLoadSingleColumnCSV:
class TestLoadSingleColumnCSV:
    """Test loading empirical data from CSV"""

    def test_load_two_column_csv(self):
        """Test loading a CSV with timestamp and value columns"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test.csv')
            
            # Create test CSV with ts and value columns
            df = pd.DataFrame({
                'ts': pd.date_range('2024-01-01', periods=24, freq='h'),
                'ts': pd.date_range('2024-01-01', periods=24, freq='h'),
                'value': range(24),
            })
            df.to_csv(csv_path, index=False)
            
            # Load it
            loaded = load_single_column_csv(csv_path, value_col='value', ts_col='ts')
            
            assert isinstance(loaded, pd.Series)
            assert len(loaded) == 24

    def test_load_single_column_csv(self):
        """Test loading a CSV with only values (implicit hourly index)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test.csv')
            
            # Create single-column CSV
            df = pd.DataFrame({'value': range(24)})
            df.to_csv(csv_path, index=False)
            
            # Load it
            loaded = load_single_column_csv(csv_path)
            
            assert isinstance(loaded, pd.Series)
            assert len(loaded) == 24
            assert isinstance(loaded.index, pd.DatetimeIndex)

    def test_load_csv_missing_file(self):
        """Test error handling for missing file"""
        with pytest.raises(FileNotFoundError):
            load_single_column_csv('/nonexistent/file.csv')


@pytest.mark.unit
class TestLoadEmpiricalSeries:
    """Test loading multiple empirical series"""

    def test_load_multiple_series(self):
        """Test loading multiple CSV files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two test files
            csv1 = os.path.join(tmpdir, 'series1.csv')
            csv2 = os.path.join(tmpdir, 'series2.csv')
            
            df1 = pd.DataFrame({
                'ts': pd.date_range('2024-01-01', periods=24, freq='h'),
                'value': range(24),
            })
            df1.to_csv(csv1, index=False)
            
            df2 = pd.DataFrame({
                'ts': pd.date_range('2024-01-01', periods=24, freq='h'),
                'value': range(100, 124),
            })
            df2.to_csv(csv2, index=False)
            
            # Load them
            series_map = {
                'fuel_gas': csv1,
                'fuel_coal': csv2,
            }
            loaded = load_empirical_series(series_map)
            
            assert 'fuel_gas' in loaded
            assert 'fuel_coal' in loaded
            assert len(loaded['fuel_gas']) == 24
            assert len(loaded['fuel_coal']) == 24
