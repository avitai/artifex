"""Tests for the test_structure_analyzer module."""

import tempfile
from pathlib import Path
from unittest import TestCase

import pytest

# Don't import these classes directly to avoid __init__ constructor issues
# TestFileAnalysis,
# TestStructureAnalyzer,
# Import the module instead to access the classes through the module
import artifex.generative_models.utils.code_analysis.test_structure_analyzer as tsa_module

# Import from the new location - don't import classes with __init__ methods
from artifex.generative_models.utils.code_analysis.test_structure_analyzer import (
    extract_imports,
    find_source_module_mappings,
    is_test_file,
)
from artifex.utils.file_utils import get_valid_output_dir


class StructureAnalyzerTests(TestCase):
    """Tests for the TestStructureAnalyzer class."""

    def setUp(self):
        """Create temporary test files for analysis."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name) / "tests"
        self.src_dir = Path(self.temp_dir.name) / "src" / "package"

        # Create directories
        self.test_dir.mkdir(parents=True)
        self.src_dir.mkdir(parents=True)

        # Create source modules
        self.src_module1 = self.src_dir / "module1.py"
        self.src_module2 = self.src_dir / "module2.py"
        self.src_submodule = self.src_dir / "subdir" / "submodule.py"

        # Create content for source modules
        self.src_dir.joinpath("subdir").mkdir(exist_ok=True)

        with open(self.src_module1, "w") as f:
            f.write("def function1():\n    return 'module1'\n")

        with open(self.src_module2, "w") as f:
            f.write("def function2():\n    return 'module2'\n")

        with open(self.src_submodule, "w") as f:
            f.write("def subfunction():\n    return 'submodule'\n")

        # Create test files
        self.test_file1 = self.test_dir / "test_module1.py"
        self.test_file2 = self.test_dir / "test_module2.py"
        self.test_file3 = self.test_dir / "subdir" / "test_submodule.py"
        self.test_file4 = self.test_dir / "test_without_import.py"

        # Create content for test files
        self.test_dir.joinpath("subdir").mkdir(exist_ok=True)

        with open(self.test_file1, "w") as f:
            f.write(
                "import sys\nimport package.module1\n\n"
                "def test_function1():\n"
                "    assert package.module1.function1() == 'module1'\n"
            )

        with open(self.test_file2, "w") as f:
            f.write(
                "from package import module2\n\n"
                "def test_function2():\n"
                "    assert module2.function2() == 'module2'\n"
            )

        with open(self.test_file3, "w") as f:
            f.write(
                "from package.subdir.submodule import subfunction\n\n"
                "def test_subfunction():\n"
                "    assert subfunction() == 'submodule'\n"
            )

        with open(self.test_file4, "w") as f:
            f.write(
                "import os\nimport sys\n\ndef test_environment():\n    assert os.name is not None\n"
            )

    def tearDown(self):
        """Clean up temporary test files."""
        self.temp_dir.cleanup()

    def test_is_test_file(self):
        """Test the is_test_file function."""
        # Test files following naming convention
        assert is_test_file(str(self.test_file1))
        assert is_test_file(str(self.test_file2))
        assert is_test_file(str(self.test_file3))
        assert is_test_file(str(self.test_file4))

        # Non-test files
        assert not is_test_file(str(self.src_module1))
        assert not is_test_file(str(self.src_module2))
        assert not is_test_file(str(self.src_submodule))

        # Edge cases
        # Use test_results directory for output files
        output_dir = get_valid_output_dir("code_analysis", "test_results")

        random_file = Path(output_dir) / "random.py"
        with open(random_file, "w") as f:
            f.write("# Not a test file\n")
        assert not is_test_file(str(random_file))

        test_in_name = Path(output_dir) / "testing_utility.py"
        with open(test_in_name, "w") as f:
            f.write("# Not a proper test file\n")
        assert not is_test_file(str(test_in_name))

    def test_extract_imports(self):
        """Test the extract_imports function."""
        # Test with direct imports
        imports = extract_imports(str(self.test_file1))
        assert "sys" in imports
        assert "package.module1" in imports

        # Test with from imports
        imports = extract_imports(str(self.test_file2))
        assert "package.module2" in imports

        # Test with from imports with sub-packages
        imports = extract_imports(str(self.test_file3))
        assert "package.subdir.submodule.subfunction" in imports

        # Test with no package imports
        imports = extract_imports(str(self.test_file4))
        assert "os" in imports
        assert "sys" in imports
        assert not any(imp.startswith("package") for imp in imports)

        # Test with empty file
        output_dir = get_valid_output_dir("code_analysis", "test_results")
        empty_file = Path(output_dir) / "empty.py"
        with open(empty_file, "w"):
            pass
        imports = extract_imports(str(empty_file))
        assert not imports

    def test_find_source_module_mappings(self):
        """Test the find_source_module_mappings function."""
        mappings = find_source_module_mappings(
            str(self.test_dir), str(self.src_dir), package_prefix="package"
        )

        # Check that the test files are correctly mapped to their source modules
        assert str(self.test_file1) in mappings
        assert str(self.test_file2) in mappings
        assert str(self.test_file3) in mappings

        # Check the correct mapping
        assert "package.module1" in mappings[str(self.test_file1)]
        assert "package.module2" in mappings[str(self.test_file2)]
        assert "package.subdir.submodule" in mappings[str(self.test_file3)]

        # Check that test file without package imports is not mapped
        assert str(self.test_file4) not in mappings

    def test_analyzer_initialization(self):
        """Test initializing the TestStructureAnalyzer."""
        analyzer = tsa_module.TestStructureAnalyzer(
            test_dir=str(self.test_dir), src_dir=str(self.src_dir), package_prefix="package"
        )

        assert analyzer.test_dir == str(self.test_dir)
        assert analyzer.src_dir == str(self.src_dir)
        assert analyzer.package_prefix == "package"

        # Test files should be discovered
        test_files = analyzer.discover_test_files()
        assert str(self.test_file1) in test_files
        assert str(self.test_file2) in test_files
        assert str(self.test_file3) in test_files
        assert str(self.test_file4) in test_files

        # Source modules should be discovered
        source_modules = analyzer.discover_source_modules()
        assert (str(self.src_module1), "package.module1") in source_modules
        assert (str(self.src_module2), "package.module2") in source_modules
        assert (str(self.src_submodule), "package.subdir.submodule") in source_modules

    def test_analyzer_analyze(self):
        """Test the TestStructureAnalyzer analyze method."""
        analyzer = tsa_module.TestStructureAnalyzer(
            test_dir=str(self.test_dir), src_dir=str(self.src_dir), package_prefix="package"
        )

        analysis = analyzer.analyze()

        # Check analysis results
        assert analysis.total_test_files == 4
        assert analysis.total_source_modules == 3
        assert analysis.source_modules_with_tests == 3
        assert analysis.test_files_without_imports == 1

        # Check coverage for each module
        assert "package.module1" in analysis.module_test_mapping
        assert len(analysis.module_test_mapping["package.module1"]) == 1
        assert str(self.test_file1) in analysis.module_test_mapping["package.module1"]

        assert "package.module2" in analysis.module_test_mapping
        assert len(analysis.module_test_mapping["package.module2"]) == 1
        assert str(self.test_file2) in analysis.module_test_mapping["package.module2"]

        assert "package.subdir.submodule" in analysis.module_test_mapping
        assert len(analysis.module_test_mapping["package.subdir.submodule"]) == 1
        assert str(self.test_file3) in analysis.module_test_mapping["package.subdir.submodule"]

        # Check untested modules - should be none in our test case
        assert not analysis.untested_modules

        # Check test files without imports from the package
        assert str(self.test_file4) in analysis.test_files_without_source_imports

    def test_generate_report(self):
        """Test generating a report from the analysis."""
        analyzer = tsa_module.TestStructureAnalyzer(
            test_dir=str(self.test_dir), src_dir=str(self.src_dir), package_prefix="package"
        )

        analysis = analyzer.analyze()
        report = analysis.generate_report()

        # Check that the report contains key sections
        assert "# Test Structure Analysis" in report
        assert "## Overview" in report
        assert "## Modules With Tests" in report
        assert "## Test Files Without Source Imports" in report
        assert "## Recommendations" in report

        # Check that the report contains the expected data
        assert "Total test files: 4" in report
        assert "Total source modules: 3" in report
        assert "Source modules with tests: 3" in report
        assert "Test files without source imports: 1" in report

        # Test files without imports should be in the report
        assert str(self.test_file4).replace(str(self.temp_dir.name), "") in report


# Create fixtures for pytest-style tests instead of class with __init__
@pytest.fixture
def test_analysis_data():
    """Create test data for TestFileAnalysis."""
    return {
        "module_test_mapping": {
            "module1": ["test1.py", "test2.py"],
            "module2": ["test3.py"],
        },
        "test_files_without_source_imports": ["test4.py"],
        "total_test_files": 3,
        "total_source_modules": 2,
        "source_modules_with_tests": 2,
        "test_files_without_imports": 1,
        "test_module_mapping": {
            "test1.py": ["module1"],
            "test2.py": ["module1"],
            "test3.py": ["module2"],
        },
        "untested_modules": [],
    }


class TestFileAnalysisTests:
    """Tests for the TestFileAnalysis class."""

    def test_initialization(self, test_analysis_data):
        """Test initializing TestFileAnalysis."""
        analysis = tsa_module.TestFileAnalysis(**test_analysis_data)

        assert analysis.total_test_files == 3
        assert analysis.total_source_modules == 2
        assert analysis.source_modules_with_tests == 2
        assert analysis.test_files_without_imports == 1

        assert "module1" in analysis.module_test_mapping
        assert len(analysis.module_test_mapping["module1"]) == 2
        assert "test1.py" in analysis.module_test_mapping["module1"]
        assert "test2.py" in analysis.module_test_mapping["module1"]

        assert "module2" in analysis.module_test_mapping
        assert len(analysis.module_test_mapping["module2"]) == 1
        assert "test3.py" in analysis.module_test_mapping["module2"]

        assert len(analysis.test_files_without_source_imports) == 1
        assert "test4.py" in analysis.test_files_without_source_imports

    def test_calculate_coverage_percentage(self, test_analysis_data):
        """Test calculating coverage percentage."""
        analysis = tsa_module.TestFileAnalysis(**test_analysis_data)

        # Both modules are tested, so coverage should be 100%
        assert analysis.calculate_coverage_percentage() == 100.0

        # Modify to test edge case with no source modules
        analysis.total_source_modules = 0
        assert analysis.calculate_coverage_percentage() == 0.0

        # Modify to test partial coverage
        analysis.total_source_modules = 4
        analysis.source_modules_with_tests = 2
        assert analysis.calculate_coverage_percentage() == 50.0

    def test_generate_recommendations(self, test_analysis_data):
        """Test generating recommendations."""
        analysis = tsa_module.TestFileAnalysis(**test_analysis_data)

        recommendations = analysis.generate_recommendations()

        # Check that we have recommendations
        assert len(recommendations) > 0

        # Check common recommendations are included
        assert any("test file naming" in r.lower() for r in recommendations)
        assert any("fixtures" in r.lower() for r in recommendations)

        # Test with low coverage - should prioritize creating tests
        analysis.total_source_modules = 10
        analysis.source_modules_with_tests = 3  # 30% coverage
        recommendations = analysis.generate_recommendations()
        assert any("creating tests" in r.lower() for r in recommendations)

        # Test with high coverage but import issues
        analysis.total_source_modules = 2
        analysis.source_modules_with_tests = 2  # 100% coverage
        analysis.test_files_without_imports = 5
        recommendations = analysis.generate_recommendations()
        assert any("import their corresponding" in r.lower() for r in recommendations)
