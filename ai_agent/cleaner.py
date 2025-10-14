"""
Data Cleaning and Type Detection Module
"""

import polars as pl
import re

class DataCleaner:
    
    @staticmethod
    def detect_and_convert_type(series: pl.Series) -> tuple[pl.Series, str, str]:
        """
        Detect best data type for a series and convert it
        Returns: (converted_series, original_type, new_type, action_taken)
        """
        original_type = str(series.dtype)
        
        # Skip if already numeric
        if series.dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
            return series, original_type, original_type, "No conversion needed"
        
        # Get non-null values for analysis
        non_null = series.drop_nulls()
        if len(non_null) == 0:
            return series, original_type, original_type, "All null values"
        
        sample = non_null.head(100).to_list()
        
        # Try Boolean detection
        if DataCleaner._is_boolean(sample):
            converted = DataCleaner._convert_to_boolean(series)
            return converted, original_type, str(converted.dtype), "Converted to Boolean"
        
        # Try Integer detection
        if DataCleaner._is_integer(sample):
            converted = DataCleaner._convert_to_integer(series)
            if converted is not None:
                return converted, original_type, str(converted.dtype), "Converted to Integer"
        
        # Try Float detection
        if DataCleaner._is_float(sample):
            converted = DataCleaner._convert_to_float(series)
            if converted is not None:
                return converted, original_type, str(converted.dtype), "Converted to Float"
        
        # Keep as string
        return series, original_type, original_type, "Kept as String"
    
    @staticmethod
    def _is_boolean(sample):
        """Check if values look like boolean"""
        bool_values = {'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n'}
        sample_str = [str(v).lower().strip() for v in sample if v is not None]
        matches = sum(1 for v in sample_str if v in bool_values)
        return matches > len(sample_str) * 0.8
    
    @staticmethod
    def _is_integer(sample):
        """Check if values look like integers"""
        count = 0
        for v in sample:
            if v is None:
                continue
            s = str(v).strip()
            # Remove currency and formatting
            s = re.sub(r'[$,₹€£¥%\s]', '', s)
            # Check if it's an integer
            if re.match(r'^-?\d+$', s):
                count += 1
        return count > len(sample) * 0.8
    
    @staticmethod
    def _is_float(sample):
        """Check if values look like floats"""
        count = 0
        for v in sample:
            if v is None:
                continue
            s = str(v).strip()
            # Remove currency and formatting
            s = re.sub(r'[$,₹€£¥%\s]', '', s)
            # Check if it's a number (int or float)
            if re.match(r'^-?\d+\.?\d*$', s):
                count += 1
        return count > len(sample) * 0.8
    
    @staticmethod
    def _convert_to_boolean(series: pl.Series) -> pl.Series:
        """Convert to boolean"""
        true_values = {'true', 'yes', '1', 't', 'y'}
        return series.map_elements(
            lambda x: True if str(x).lower().strip() in true_values else False,
            return_dtype=pl.Boolean
        )
    
    @staticmethod
    def _convert_to_integer(series: pl.Series) -> pl.Series:
        """Convert to integer with cleaning"""
        try:
            return series.map_elements(
                lambda x: int(re.sub(r'[^\d-]', '', str(x))) if x is not None else None,
                return_dtype=pl.Int64
            )
        except:
            return None
    
    @staticmethod
    def _convert_to_float(series: pl.Series) -> pl.Series:
        """Convert to float with cleaning"""
        try:
            def clean_and_convert(x):
                if x is None:
                    return None
                s = str(x).strip()
                # Remove currency symbols and commas
                s = re.sub(r'[$,₹€£¥%\s]', '', s)
                # Handle parentheses for negative numbers
                if '(' in s and ')' in s:
                    s = '-' + s.replace('(', '').replace(')', '')
                try:
                    return float(s)
                except:
                    return None
            
            return series.map_elements(clean_and_convert, return_dtype=pl.Float64)
        except:
            return None
    
    @staticmethod
    def clean_dataframe(df: pl.DataFrame) -> tuple[pl.DataFrame, list]:
        """
        Clean entire dataframe with intelligent type detection
        Returns: (cleaned_df, cleaning_report)
        """
        df_cleaned = df.clone()
        cleaning_report = []
        
        for col in df.columns:
            converted, old_type, new_type, action = DataCleaner.detect_and_convert_type(df_cleaned[col])
            
            df_cleaned = df_cleaned.with_columns(converted.alias(col))
            
            if action != "No conversion needed":
                cleaning_report.append({
                    'column': col,
                    'from': old_type,
                    'to': new_type,
                    'action': action
                })
            
            # Handle null values after conversion
            null_count = df_cleaned[col].null_count()
            if null_count > 0:
                if df_cleaned[col].dtype in [pl.Float64, pl.Int64]:
                    df_cleaned = df_cleaned.with_columns(
                        pl.col(col).fill_null(0).alias(col)
                    )
                    cleaning_report.append({
                        'column': col,
                        'from': 'null',
                        'to': '0',
                        'action': f'Filled {null_count} null values with 0'
                    })
                elif df_cleaned[col].dtype == pl.Boolean:
                    df_cleaned = df_cleaned.with_columns(
                        pl.col(col).fill_null(False).alias(col)
                    )
                    cleaning_report.append({
                        'column': col,
                        'from': 'null',
                        'to': 'False',
                        'action': f'Filled {null_count} null values with False'
                    })
        
        return df_cleaned, cleaning_report
