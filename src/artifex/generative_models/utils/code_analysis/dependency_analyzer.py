"""
Module for analyzing dependencies between Python modules.

This module provides tools for discovering, analyzing, and visualizing
dependencies between Python modules in a codebase.
"""

import ast
import glob
import os
from dataclasses import dataclass

import graphviz


@dataclass
class ModuleDependency:
    """Represents a dependency between two modules."""

    source: str
    """Path to the source module."""

    target: str
    """Name of the target module."""


def get_module_dependencies(module_path: str) -> list[ModuleDependency]:
    """
    Extract dependencies from a Python module file.

    Args:
        module_path: Path to the Python module file.

    Returns:
        list of ModuleDependency objects representing the module's dependencies.
    """
    dependencies: list[ModuleDependency] = []

    try:
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    dependencies.append(ModuleDependency(source=module_path, target=name.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.append(ModuleDependency(source=module_path, target=node.module))
    except Exception as e:
        print(f"Error parsing {module_path}: {e}")

    return dependencies


class DependencyAnalyzer:
    """
    Analyzes dependencies between Python modules in a directory.

    This class provides methods for discovering Python modules in a directory,
    analyzing their dependencies, detecting circular dependencies, and
    generating dependency graphs.
    """

    def __init__(self, root_dir: str):
        """
        Initialize the analyzer with a root directory.

        Args:
            root_dir: Path to the root directory to analyze.
        """
        self.root_dir = root_dir
        self.modules = self._discover_modules()

    def _discover_modules(self) -> list[str]:
        """
        Discover all Python modules in the root directory.

        Returns:
            list of paths to Python module files.
        """
        return glob.glob(os.path.join(self.root_dir, "**", "*.py"), recursive=True)

    def get_all_dependencies(self) -> list[ModuleDependency]:
        """
        Get dependencies for all discovered modules.

        Returns:
            list of ModuleDependency objects.
        """
        all_deps: list[ModuleDependency] = []
        for module in self.modules:
            all_deps.extend(get_module_dependencies(module))
        return all_deps

    def generate_graph(
        self,
        output_file: str,
        format: str = "svg",
        include_pattern: str | None = None,
        exclude_pattern: str | None = None,
    ) -> None:
        """
        Generate a dependency graph visualization.

        Args:
            output_file: Path to save the output graph.
            format: Output format (default: svg).
            include_pattern: Pattern for modules to include (default: None).
            exclude_pattern: Pattern for modules to exclude (default: None).
        """
        deps = self.get_all_dependencies()

        # Filter dependencies based on patterns
        if include_pattern or exclude_pattern:
            filtered_deps: list[ModuleDependency] = []
            for dep in deps:
                source_name = os.path.basename(dep.source)

                include = True
                if include_pattern and include_pattern not in source_name:
                    include = False
                if exclude_pattern and exclude_pattern in source_name:
                    include = False

                if include:
                    filtered_deps.append(dep)
            deps = filtered_deps

        # Create graph
        dot = graphviz.Digraph(comment="Module Dependencies", format=format)

        # Add nodes and edges
        for dep in deps:
            source_name = os.path.basename(dep.source).replace(".py", "")
            target_name = dep.target.split(".")[-1]

            dot.node(source_name)
            dot.node(target_name)
            dot.edge(source_name, target_name)

        # Make sure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save the graph without the file extension as graphviz will add it
        output_path = os.path.splitext(output_file)[0]
        dot.render(output_path, cleanup=True)


def detect_circular_dependencies(
    dependencies: list[ModuleDependency],
) -> list[list[ModuleDependency]]:
    """
    Detect circular dependencies in a list of module dependencies.

    Args:
        dependencies: list of ModuleDependency objects.

    Returns:
        list of lists, where each inner list is a cycle of dependencies.
    """
    # Build adjacency list
    graph: dict[str, list[str]] = {}
    module_map: dict[str, str] = {}

    for dep in dependencies:
        source_name = os.path.basename(dep.source).replace(".py", "")
        module_map[source_name] = dep.source

        if source_name not in graph:
            graph[source_name] = []

        if dep.target not in graph:
            graph[dep.target] = []

        graph[source_name].append(dep.target)

    # Detect cycles using DFS
    cycles: list[list[ModuleDependency]] = []
    visited: set[str] = set()
    rec_stack: set[str] = set()

    def dfs(node: str, path: list[str]) -> None:
        nonlocal cycles, visited, rec_stack

        if node in rec_stack:
            # Found a cycle
            cycle_start = path.index(node)
            cycle_path = [*path[cycle_start:], node]

            # Convert to ModuleDependency objects
            cycle_deps: list[ModuleDependency] = []
            for i in range(len(cycle_path) - 1):
                source = cycle_path[i]
                target = cycle_path[i + 1]

                if source in module_map:
                    cycle_deps.append(ModuleDependency(source=module_map[source], target=target))

            if cycle_deps:
                cycles.append(cycle_deps)
            return

        if node in visited:
            return

        visited.add(node)
        rec_stack.add(node)

        for neighbor in graph.get(node, []):
            dfs(neighbor, [*path, node])

        rec_stack.remove(node)

    # Run DFS from each node
    for node in graph:
        if node not in visited:
            dfs(node, [])

    return cycles


def generate_dependency_report(
    dependencies: list[ModuleDependency],
    circular_dependencies: list[list[ModuleDependency]],
) -> str:
    """
    Generate a text report from dependency analysis.

    Args:
        dependencies: list of all module dependencies.
        circular_dependencies: list of circular dependency cycles.

    Returns:
        Formatted text report.
    """
    # Count dependencies per module
    dependencies_count: dict[str, int] = {}
    dependants_count: dict[str, int] = {}

    for dep in dependencies:
        source = os.path.basename(dep.source).replace(".py", "")
        target = dep.target

        if source not in dependencies_count:
            dependencies_count[source] = 0
        dependencies_count[source] += 1

        if target not in dependants_count:
            dependants_count[target] = 0
        dependants_count[target] += 1

    # Generate report
    report: list[str] = ["# Module Dependency Analysis\n"]

    report.append("## Module Relationships\n")
    report.append("| Module | Dependencies | Dependants |\n")
    report.append("|--------|--------------|------------|\n")

    for module in sorted(set(list(dependencies_count.keys()) + list(dependants_count.keys()))):
        deps = dependencies_count.get(module, 0)
        deps_on = dependants_count.get(module, 0)
        report.append(f"| {module} | {deps} | {deps_on} |\n")

    if circular_dependencies:
        report.append("\n## Circular Dependencies\n")
        for i, cycle in enumerate(circular_dependencies, start=1):
            report.append(f"\n### Cycle {i}\n")
            for dep in cycle:
                source = os.path.basename(dep.source).replace(".py", "")
                report.append(f"- {source} → {dep.target}\n")
    else:
        report.append("\nNo circular dependencies detected.\n")

    return "".join(report)
