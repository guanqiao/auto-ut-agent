"""Project statistics dialog for PyUT Agent."""

import logging
from pathlib import Path
from typing import Dict, List, Any

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QGroupBox, QGridLayout, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ...core.config import get_settings

logger = logging.getLogger(__name__)


class ProjectScanner(QThread):
    """Worker thread for scanning project."""

    progress_updated = pyqtSignal(str)
    scan_completed = pyqtSignal(dict)

    def __init__(self, project_path: str):
        super().__init__()
        self.project_path = project_path

    def run(self):
        """Scan the project."""
        try:
            stats = self._scan_project()
            self.scan_completed.emit(stats)
        except Exception as e:
            logger.exception(f"Project scan failed: {e}")
            self.scan_completed.emit({"error": str(e)})

    def _scan_project(self) -> Dict[str, Any]:
        """Scan project and collect statistics."""
        self.progress_updated.emit("Scanning project structure...")

        settings = get_settings()
        src_dir = Path(self.project_path) / settings.project_paths.src_main_java
        test_dir = Path(self.project_path) / settings.project_paths.src_test_java

        stats = {
            "project_path": self.project_path,
            "project_name": Path(self.project_path).name,
            "main_java_files": [],
            "test_java_files": [],
            "packages": set(),
            "total_lines": 0,
            "total_classes": 0,
            "total_interfaces": 0,
            "total_enums": 0,
            "error": None
        }

        if not src_dir.exists():
            stats["error"] = f"Source directory not found: {src_dir}"
            return stats

        self.progress_updated.emit("Scanning main source files...")
        stats["main_java_files"] = self._scan_java_files(src_dir, stats)

        if test_dir.exists():
            self.progress_updated.emit("Scanning test files...")
            stats["test_java_files"] = self._scan_java_files(test_dir, stats, is_test=True)

        stats["packages"] = sorted(list(stats["packages"]))
        stats["total_files"] = len(stats["main_java_files"]) + len(stats["test_java_files"])

        return stats

    def _scan_java_files(
        self,
        directory: Path,
        stats: Dict[str, Any],
        is_test: bool = False
    ) -> List[Dict[str, Any]]:
        """Scan Java files in directory."""
        files = []

        for java_file in directory.rglob("*.java"):
            try:
                rel_path = java_file.relative_to(directory)
                package = str(rel_path.parent).replace("/", ".").replace("\\", ".")

                if package != ".":
                    stats["packages"].add(package)

                file_stats = self._analyze_java_file(java_file)
                file_stats["path"] = str(java_file)
                file_stats["relative_path"] = str(rel_path)
                file_stats["package"] = package if package != "." else "(default)"
                file_stats["is_test"] = is_test

                files.append(file_stats)

                stats["total_lines"] += file_stats["lines"]
                stats["total_classes"] += file_stats["classes"]
                stats["total_interfaces"] += file_stats["interfaces"]
                stats["total_enums"] += file_stats["enums"]

            except Exception as e:
                logger.warning(f"Failed to analyze {java_file}: {e}")

        return files

    def _analyze_java_file(self, file_path: Path) -> Dict[str, int]:
        """Analyze a single Java file."""
        stats = {
            "lines": 0,
            "classes": 0,
            "interfaces": 0,
            "enums": 0,
            "methods": 0
        }

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')

                stats["lines"] = len([l for l in lines if l.strip() and not l.strip().startswith(("//", "/*", "*"))])

                for line in lines:
                    line_stripped = line.strip()
                    if 'class ' in line_stripped and not line_stripped.startswith(("//", "*")):
                        stats["classes"] += 1
                    if 'interface ' in line_stripped and not line_stripped.startswith(("//", "*")):
                        stats["interfaces"] += 1
                    if 'enum ' in line_stripped and not line_stripped.startswith(("//", "*")):
                        stats["enums"] += 1

        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")

        return stats


class ProjectStatsDialog(QDialog):
    """Dialog showing project statistics."""

    def __init__(self, project_path: str, parent=None):
        super().__init__(parent)
        self.project_path = project_path
        self.scanner = None

        self.setWindowTitle(f"Project Statistics - {Path(project_path).name}")
        self.setMinimumSize(800, 600)

        self.setup_ui()
        self.start_scan()

    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)

        self.progress_label = QLabel("Scanning project...")
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)

        self.stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout(self.stats_group)

        self.project_label = QLabel("Project:")
        self.project_value = QLabel()
        stats_layout.addWidget(self.project_label, 0, 0)
        stats_layout.addWidget(self.project_value, 0, 1)

        self.files_label = QLabel("Total Files:")
        self.files_value = QLabel()
        stats_layout.addWidget(self.files_label, 1, 0)
        stats_layout.addWidget(self.files_value, 1, 1)

        self.main_files_label = QLabel("Main Java Files:")
        self.main_files_value = QLabel()
        stats_layout.addWidget(self.main_files_label, 2, 0)
        stats_layout.addWidget(self.main_files_value, 2, 1)

        self.test_files_label = QLabel("Test Java Files:")
        self.test_files_value = QLabel()
        stats_layout.addWidget(self.test_files_label, 3, 0)
        stats_layout.addWidget(self.test_files_value, 3, 1)

        self.packages_label = QLabel("Packages:")
        self.packages_value = QLabel()
        stats_layout.addWidget(self.packages_label, 4, 0)
        stats_layout.addWidget(self.packages_value, 4, 1)

        self.lines_label = QLabel("Total Lines:")
        self.lines_value = QLabel()
        stats_layout.addWidget(self.lines_label, 5, 0)
        stats_layout.addWidget(self.lines_value, 5, 1)

        self.classes_label = QLabel("Classes:")
        self.classes_value = QLabel()
        stats_layout.addWidget(self.classes_label, 6, 0)
        stats_layout.addWidget(self.classes_value, 6, 1)

        self.interfaces_label = QLabel("Interfaces:")
        self.interfaces_value = QLabel()
        stats_layout.addWidget(self.interfaces_label, 7, 0)
        stats_layout.addWidget(self.interfaces_value, 7, 1)

        self.enums_label = QLabel("Enums:")
        self.enums_value = QLabel()
        stats_layout.addWidget(self.enums_label, 8, 0)
        stats_layout.addWidget(self.enums_value, 8, 1)

        self.coverage_label = QLabel("Test Coverage:")
        self.coverage_value = QLabel()
        stats_layout.addWidget(self.coverage_label, 9, 0)
        stats_layout.addWidget(self.coverage_value, 9, 1)

        layout.addWidget(self.stats_group)

        self.details_group = QGroupBox("Package Details")
        details_layout = QVBoxLayout(self.details_group)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(200)
        details_layout.addWidget(self.details_text)

        layout.addWidget(self.details_group)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.export_button = QPushButton("Export Report")
        self.export_button.clicked.connect(self.export_report)
        self.export_button.setEnabled(False)
        button_layout.addWidget(self.export_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

        self.stats_group.hide()
        self.details_group.hide()

    def start_scan(self):
        """Start project scanning."""
        self.scanner = ProjectScanner(self.project_path)
        self.scanner.progress_updated.connect(self.on_progress_updated)
        self.scanner.scan_completed.connect(self.on_scan_completed)
        self.scanner.start()

    def on_progress_updated(self, message: str):
        """Handle progress update."""
        self.progress_label.setText(message)

    def on_scan_completed(self, stats: Dict[str, Any]):
        """Handle scan completion."""
        self.progress_label.hide()
        self.progress_bar.hide()

        if stats.get("error"):
            self.project_value.setText(f"Error: {stats['error']}")
            self.stats_group.show()
            return

        self.stats = stats

        self.project_value.setText(stats["project_name"])
        self.files_value.setText(str(stats["total_files"]))
        self.main_files_value.setText(str(len(stats["main_java_files"])))
        self.test_files_value.setText(str(len(stats["test_java_files"])))
        self.packages_value.setText(str(len(stats["packages"])))
        self.lines_value.setText(f"{stats['total_lines']:,}")
        self.classes_value.setText(str(stats["total_classes"]))
        self.interfaces_value.setText(str(stats["total_interfaces"]))
        self.enums_value.setText(str(stats["total_enums"]))

        test_ratio = len(stats["test_java_files"]) / len(stats["main_java_files"]) if stats["main_java_files"] else 0
        self.coverage_value.setText(f"{test_ratio:.1%} (test files / main files)")

        self.details_text.clear()
        self.details_text.append("Packages:\n")
        for package in stats["packages"]:
            main_count = sum(1 for f in stats["main_java_files"] if f["package"] == package)
            test_count = sum(1 for f in stats["test_java_files"] if f["package"] == package)
            self.details_text.append(f"  • {package}: {main_count} main, {test_count} test files")

        self.stats_group.show()
        self.details_group.show()
        self.export_button.setEnabled(True)

    def export_report(self):
        """Export statistics report."""
        try:
            from PyQt6.QtWidgets import QFileDialog

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Report",
                f"{self.stats['project_name']}_stats.txt",
                "Text Files (*.txt);;All Files (*)"
            )

            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Project Statistics Report\n")
                    f.write(f"{'=' * 60}\n\n")
                    f.write(f"Project: {self.stats['project_name']}\n")
                    f.write(f"Path: {self.stats['project_path']}\n\n")
                    f.write(f"Statistics:\n")
                    f.write(f"  Total Files: {self.stats['total_files']}\n")
                    f.write(f"  Main Java Files: {len(self.stats['main_java_files'])}\n")
                    f.write(f"  Test Java Files: {len(self.stats['test_java_files'])}\n")
                    f.write(f"  Packages: {len(self.stats['packages'])}\n")
                    f.write(f"  Total Lines: {self.stats['total_lines']}\n")
                    f.write(f"  Classes: {self.stats['total_classes']}\n")
                    f.write(f"  Interfaces: {self.stats['total_interfaces']}\n")
                    f.write(f"  Enums: {self.stats['total_enums']}\n\n")
                    f.write(f"Packages:\n")
                    for package in self.stats['packages']:
                        main_count = sum(1 for f in self.stats["main_java_files"] if f["package"] == package)
                        test_count = sum(1 for f in self.stats["test_java_files"] if f["package"] == package)
                        f.write(f"  • {package}: {main_count} main, {test_count} test files\n")

                logger.info(f"Report exported to {file_path}")

        except Exception as e:
            logger.exception(f"Failed to export report: {e}")
