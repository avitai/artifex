"""Utilities for accurate repo-local Python import graph analysis."""

from __future__ import annotations

import ast
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import graphviz
from graphviz.backend import ExecutableNotFound


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModuleDependency:
    """Represents one import edge between two modules."""

    source: str
    """Filesystem path to the source module."""

    target: str
    """Resolved target module name when possible, otherwise the raw import path."""

    source_module: str | None = None
    """Fully qualified source module name when the analyzer knows the package root."""


def _module_name_from_path(module_path: str | Path, root_dir: str | Path) -> str:
    """Return the fully qualified module name for a discovered Python file."""
    module_path = Path(module_path).resolve()
    root_dir = Path(root_dir).resolve()
    relative_path = module_path.relative_to(root_dir)
    parts = list(relative_path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if (root_dir / "__init__.py").exists():
        parts = [root_dir.name, *parts]
    return ".".join(parts)


def _current_package_name(module_name: str) -> str:
    """Return the package that relative imports should resolve against."""
    if "." not in module_name:
        return module_name
    return module_name.rsplit(".", 1)[0]


def _resolve_import_from_targets(
    *,
    current_module: str,
    node: ast.ImportFrom,
    known_modules: set[str],
) -> list[str]:
    """Resolve ``from ... import ...`` edges as accurately as the repo graph allows."""
    current_package = _current_package_name(current_module)
    base_module = node.module or ""

    if node.level:
        package_parts = current_package.split(".")
        trim_count = max(node.level - 1, 0)
        if trim_count >= len(package_parts):
            relative_base = []
        else:
            relative_base = package_parts[: len(package_parts) - trim_count]
        base_parts = relative_base + ([base_module] if base_module else [])
        resolved_base = ".".join(part for part in base_parts if part)
    else:
        resolved_base = base_module

    resolved_targets: list[str] = []
    if resolved_base:
        if resolved_base in known_modules:
            resolved_targets.append(resolved_base)
        for imported_name in node.names:
            candidate = ".".join(part for part in [resolved_base, imported_name.name] if part)
            if candidate in known_modules:
                resolved_targets.append(candidate)

    if resolved_targets:
        return list(dict.fromkeys(resolved_targets))

    if resolved_base:
        return [resolved_base]

    fallback_targets: list[str] = []
    for imported_name in node.names:
        candidate = ".".join(part for part in [current_package, imported_name.name] if part)
        if candidate in known_modules:
            fallback_targets.append(candidate)
        else:
            fallback_targets.append(imported_name.name)
    return list(dict.fromkeys(fallback_targets))


def get_module_dependencies(
    module_path: str,
    *,
    source_module: str | None = None,
    known_modules: set[str] | None = None,
) -> list[ModuleDependency]:
    """Extract dependencies from a Python module file.

    Args:
        module_path: Path to the Python module file.
        source_module: Optional source module name to use in dependency records.
        known_modules: Optional set of known local modules for classification.

    Returns:
        list of ModuleDependency objects representing the module's dependencies.
    """
    dependencies: list[ModuleDependency] = []

    try:
        with open(module_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    dependencies.append(
                        ModuleDependency(
                            source=module_path,
                            target=name.name,
                            source_module=source_module,
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                targets: list[str]
                if source_module and known_modules is not None:
                    targets = _resolve_import_from_targets(
                        current_module=source_module,
                        node=node,
                        known_modules=known_modules,
                    )
                elif node.module:
                    targets = [node.module]
                else:
                    targets = [name.name for name in node.names]

                for target in targets:
                    dependencies.append(
                        ModuleDependency(
                            source=module_path,
                            target=target,
                            source_module=source_module,
                        )
                    )
    except (SyntaxError, OSError) as e:
        logger.warning("Error parsing %s: %s", module_path, e)

    return dependencies


class DependencyAnalyzer:
    """Analyzes dependencies between Python modules in a directory.

    This class provides methods for discovering Python modules in a directory,
    analyzing their dependencies, detecting circular dependencies, and
    generating dependency graphs.
    """

    def __init__(self, root_dir: str):
        """Initialize the analyzer with a root directory.

        Args:
            root_dir: Path to the root directory to analyze.
        """
        self.root_dir = str(Path(root_dir).resolve())
        self.modules = self._discover_modules()
        self.module_names = {
            module: _module_name_from_path(module, self.root_dir) for module in self.modules
        }

    def _discover_modules(self) -> list[str]:
        """Discover all Python modules in the root directory.

        Returns:
            list of paths to Python module files.
        """
        return sorted(str(path.resolve()) for path in Path(self.root_dir).rglob("*.py"))

    def get_all_dependencies(self) -> list[ModuleDependency]:
        """Get dependencies for all discovered modules.

        Returns:
            list of ModuleDependency objects.
        """
        all_deps: list[ModuleDependency] = []
        known_modules = set(self.module_names.values())
        for module in self.modules:
            all_deps.extend(
                get_module_dependencies(
                    module,
                    source_module=self.module_names[module],
                    known_modules=known_modules,
                )
            )
        return all_deps

    def generate_graph(
        self,
        output_file: str,
        format: str = "svg",
        include_pattern: str | None = None,
        exclude_pattern: str | None = None,
    ) -> None:
        """Generate a dependency graph visualization.

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
                source_name = dep.source_module or Path(dep.source).stem

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
            source_name = dep.source_module or Path(dep.source).stem
            target_name = dep.target

            dot.node(source_name)
            dot.node(target_name)
            dot.edge(source_name, target_name)

        # Make sure the output directory exists
        output_path = Path(output_file)
        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the graph without the file extension as graphviz will add it.
        # If the Graphviz executable is unavailable, persist the DOT source so
        # internal analysis artifacts are still written deterministically.
        rendered_output = (
            output_path if output_path.suffix else output_path.with_suffix(f".{format}")
        )
        try:
            dot.render(str(output_path.with_suffix("")), cleanup=True)
        except ExecutableNotFound:
            rendered_output.write_text(dot.source, encoding="utf-8")


def detect_circular_dependencies(
    dependencies: list[ModuleDependency],
) -> list[list[ModuleDependency]]:
    """Detect circular dependencies in a list of module dependencies.

    Args:
        dependencies: list of ModuleDependency objects.

    Returns:
        list of lists, where each inner list is a cycle of dependencies.
    """
    internal_modules = {
        dep.source_module or Path(dep.source).stem
        for dep in dependencies
        if (dep.source_module or Path(dep.source).stem)
    }
    graph: dict[str, set[str]] = defaultdict(set)
    edge_lookup: dict[tuple[str, str], ModuleDependency] = {}

    for dep in dependencies:
        source_name = dep.source_module or Path(dep.source).stem
        graph.setdefault(source_name, set())
        if dep.target in internal_modules:
            graph[source_name].add(dep.target)
            edge_lookup.setdefault((source_name, dep.target), dep)

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    strongly_connected_components: list[set[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in sorted(graph.get(node, ())):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] == indices[node]:
            component: set[str] = set()
            while stack:
                member = stack.pop()
                on_stack.remove(member)
                component.add(member)
                if member == node:
                    break
            strongly_connected_components.append(component)

    for node in sorted(graph):
        if node not in indices:
            strongconnect(node)

    cycles: list[list[ModuleDependency]] = []
    for component in strongly_connected_components:
        if len(component) == 1:
            member = next(iter(component))
            if member not in graph.get(member, set()):
                continue

        cycle_edges = [
            edge_lookup[(source_name, target_name)]
            for source_name in sorted(component)
            for target_name in sorted(graph.get(source_name, ()))
            if target_name in component
        ]
        if cycle_edges:
            cycles.append(cycle_edges)

    return cycles


def generate_dependency_report(
    dependencies: list[ModuleDependency],
    circular_dependencies: list[list[ModuleDependency]],
) -> str:
    """Generate a text report from dependency analysis.

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
        source = dep.source_module or Path(dep.source).stem
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
                source = dep.source_module or Path(dep.source).stem
                report.append(f"- {source} → {dep.target}\n")
    else:
        report.append("\nNo circular dependencies detected.\n")

    return "".join(report)
