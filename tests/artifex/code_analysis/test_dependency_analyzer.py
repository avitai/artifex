from pathlib import Path

from artifex.generative_models.utils.code_analysis.dependency_analyzer import (
    DependencyAnalyzer,
    detect_circular_dependencies,
    generate_dependency_report,
    get_module_dependencies,
)


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def test_get_module_dependencies_reads_basic_imports(tmp_path: Path) -> None:
    """Import collection should preserve the source path and imported module names."""
    module = tmp_path / "module_a.py"
    _write(module, "import module_b\nfrom package import module_c\n")

    deps = get_module_dependencies(str(module))

    assert len(deps) == 2
    assert deps[0].source == str(module)
    assert deps[0].target == "module_b"
    assert deps[0].source_module is None
    assert deps[1].source == str(module)
    assert deps[1].target == "package"


def test_dependency_analyzer_uses_qualified_module_identities(tmp_path: Path) -> None:
    """Repeated basenames in different packages should stay distinct."""
    _write(tmp_path / "pkg_a" / "__init__.py", "")
    _write(tmp_path / "pkg_a" / "base.py", "from pkg_b import base\n")
    _write(tmp_path / "pkg_b" / "__init__.py", "")
    _write(tmp_path / "pkg_b" / "base.py", "from pkg_a import base\n")

    analyzer = DependencyAnalyzer(str(tmp_path))

    deps = analyzer.get_all_dependencies()
    edges = {(dep.source_module, dep.target) for dep in deps if dep.source_module}

    assert ("pkg_a.base", "pkg_b.base") in edges
    assert ("pkg_b.base", "pkg_a.base") in edges

    report = generate_dependency_report(deps, detect_circular_dependencies(deps))
    assert "| pkg_a.base |" in report
    assert "| pkg_b.base |" in report
    assert "\n| base |" not in report


def test_detect_circular_dependencies_finds_multi_module_cycles(tmp_path: Path) -> None:
    """Cycle detection should cover longer chains, not only mutual pairs."""
    _write(tmp_path / "artifex" / "__init__.py", "")
    _write(tmp_path / "artifex" / "a.py", "from artifex import b\n")
    _write(tmp_path / "artifex" / "b.py", "from artifex import c\n")
    _write(tmp_path / "artifex" / "c.py", "from artifex import a\n")

    analyzer = DependencyAnalyzer(str(tmp_path))

    circular_deps = detect_circular_dependencies(analyzer.get_all_dependencies())

    assert len(circular_deps) == 1
    assert {
        (dep.source_module, dep.target) for dep in circular_deps[0] if dep.source_module is not None
    } == {
        ("artifex.a", "artifex.b"),
        ("artifex.b", "artifex.c"),
        ("artifex.c", "artifex.a"),
    }


def test_analyzer_generate_graph_keeps_qualified_names(tmp_path: Path) -> None:
    """The rendered graph should keep package-qualified node names."""
    _write(tmp_path / "pkg_a" / "__init__.py", "")
    _write(tmp_path / "pkg_a" / "base.py", "from pkg_b import base\n")
    _write(tmp_path / "pkg_b" / "__init__.py", "")
    _write(tmp_path / "pkg_b" / "base.py", "from pkg_a import base\n")

    analyzer = DependencyAnalyzer(str(tmp_path))
    output_file = tmp_path / "out" / "dependencies.svg"

    analyzer.generate_graph(str(output_file))

    contents = output_file.read_text(encoding="utf-8")
    assert "pkg_a.base" in contents
    assert "pkg_b.base" in contents
