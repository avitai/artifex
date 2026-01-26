import os
import tempfile
import unittest
from pathlib import Path

# Import from the new location
from artifex.generative_models.utils.code_analysis.dependency_analyzer import (
    DependencyAnalyzer,
    detect_circular_dependencies,
    get_module_dependencies,
)
from artifex.utils.file_utils import get_valid_output_dir


class TestDependencyAnalyzer(unittest.TestCase):
    """Tests for the DependencyAnalyzer class and related functions."""

    def setUp(self):
        """Create temporary test files for dependency analysis."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Create a simple module structure
        self.module_a = self.test_dir / "module_a.py"
        self.module_b = self.test_dir / "module_b.py"
        self.module_c = self.test_dir / "module_c.py"

        # Module A imports B
        with open(self.module_a, "w") as f:
            f.write("import module_b\n\ndef function_a():\n    return module_b.function_b()\n")

        # Module B imports C
        with open(self.module_b, "w") as f:
            f.write("import module_c\n\ndef function_b():\n    return module_c.function_c()\n")

        # Module C has no imports
        with open(self.module_c, "w") as f:
            f.write("def function_c():\n    return 'C'\n")

        # Create a circular dependency
        self.circular_a = self.test_dir / "circular_a.py"
        self.circular_b = self.test_dir / "circular_b.py"

        # Circular A imports Circular B
        with open(self.circular_a, "w") as f:
            f.write("import circular_b\n\ndef function_a():\n    return circular_b.function_b()\n")

        # Circular B imports Circular A
        with open(self.circular_b, "w") as f:
            f.write("import circular_a\n\ndef function_b():\n    return circular_a.function_a()\n")

    def tearDown(self):
        """Clean up temporary test files."""
        self.temp_dir.cleanup()

    def test_get_module_dependencies(self):
        """Test extracting dependencies from a module."""
        deps = get_module_dependencies(str(self.module_a))
        assert len(deps) == 1
        assert deps[0].source == str(self.module_a)
        assert deps[0].target == "module_b"

        deps = get_module_dependencies(str(self.module_b))
        assert len(deps) == 1
        assert deps[0].source == str(self.module_b)
        assert deps[0].target == "module_c"

        deps = get_module_dependencies(str(self.module_c))
        assert len(deps) == 0

    def test_dependency_analyzer_initialization(self):
        """Test initializing the DependencyAnalyzer."""
        analyzer = DependencyAnalyzer(str(self.test_dir))
        assert analyzer.root_dir == str(self.test_dir)
        assert len(analyzer.modules) > 0
        assert str(self.module_a) in analyzer.modules

    def test_dependency_analyzer_get_all_dependencies(self):
        """Test getting all dependencies from the analyzer."""
        analyzer = DependencyAnalyzer(str(self.test_dir))
        deps = analyzer.get_all_dependencies()

        # Should have at least 3 dependencies (including circular ones)
        assert len(deps) >= 3

        # Check for specific dependencies
        module_a_deps = [d for d in deps if d.source == str(self.module_a)]
        assert len(module_a_deps) == 1
        assert module_a_deps[0].target == "module_b"

    def test_detect_circular_dependencies(self):
        """Test detecting circular dependencies."""
        analyzer = DependencyAnalyzer(str(self.test_dir))
        circular_deps = detect_circular_dependencies(analyzer.get_all_dependencies())

        # Should find one circular dependency
        assert len(circular_deps) == 1

        # Check the circular dependency
        circular = circular_deps[0]
        assert len(circular) == 2
        assert any(d.source == str(self.circular_a) and d.target == "circular_b" for d in circular)
        assert any(d.source == str(self.circular_b) and d.target == "circular_a" for d in circular)

    def test_analyzer_generate_graph(self):
        """Test generating a graph from dependencies."""
        analyzer = DependencyAnalyzer(str(self.test_dir))

        # Use test_results directory for output
        output_dir = get_valid_output_dir("code_analysis", "test_results")
        output_file = Path(output_dir) / "dependencies.svg"

        analyzer.generate_graph(str(output_file))
        assert os.path.exists(output_file)

        # Basic verification of SVG content
        with open(output_file, "r") as f:
            content = f.read()
            assert "svg" in content
            assert "module_a" in content
            assert "module_b" in content
