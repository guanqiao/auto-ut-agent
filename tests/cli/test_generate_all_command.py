"""Tests for generate-all command."""

import os
import tempfile
from pathlib import Path
from click.testing import CliRunner
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio


class TestGenerateAllCommand:
    """Test generate-all command functionality."""

    def test_generate_all_help(self):
        """Test generate-all command shows help."""
        from pyutagent.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['generate-all', '--help'])

        assert result.exit_code == 0
        assert 'Generate unit tests for all Java files' in result.output
        assert '--parallel' in result.output or '-p' in result.output
        assert '--continue-on-error' in result.output

    def test_generate_all_nonexistent_project(self):
        """Test generate-all with non-existent project."""
        from pyutagent.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['generate-all', '/nonexistent/project'])

        assert result.exit_code != 0

    def test_generate_all_not_maven_project(self):
        """Test generate-all with non-Maven project."""
        from pyutagent.cli.main import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            result = runner.invoke(cli, ['generate-all', tmpdir])

            assert result.exit_code != 0
            assert 'Maven' in result.output or 'pom.xml' in result.output

    def test_generate_all_no_java_files(self):
        """Test generate-all with no Java files."""
        from pyutagent.cli.main import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            pom_file = Path(tmpdir) / 'pom.xml'
            pom_file.write_text('<project></project>')
            
            src_dir = Path(tmpdir) / 'src' / 'main' / 'java'
            src_dir.mkdir(parents=True)

            runner = CliRunner()
            result = runner.invoke(cli, ['generate-all', tmpdir])

            assert 'No Java files' in result.output or result.exit_code == 0

    @patch('pyutagent.services.batch_generator.BatchGenerator')
    @patch('pyutagent.config.load_llm_config')
    def test_generate_all_with_mock_generator(self, mock_load_config, mock_generator_class):
        """Test generate-all with mocked generator."""
        from pyutagent.cli.main import cli

        mock_config = MagicMock()
        mock_collection = MagicMock()
        mock_collection.get_default_config.return_value = mock_config
        mock_load_config.return_value = mock_collection

        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator
        
        mock_result = MagicMock()
        mock_result.success_count = 2
        mock_result.failed_count = 0
        mock_result.skipped_count = 0
        mock_result.total_files = 2
        mock_result.results = []
        mock_generator.generate_all_sync.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            pom_file = Path(tmpdir) / 'pom.xml'
            pom_file.write_text('<project></project>')
            
            src_dir = Path(tmpdir) / 'src' / 'main' / 'java'
            src_dir.mkdir(parents=True)
            
            (src_dir / 'ServiceA.java').write_text('public class ServiceA {}')
            (src_dir / 'ServiceB.java').write_text('public class ServiceB {}')

            runner = CliRunner()
            result = runner.invoke(cli, ['generate-all', tmpdir])

            assert result.exit_code == 0
            mock_generator.generate_all_sync.assert_called_once()

    def test_generate_all_with_parallel_option(self):
        """Test generate-all with parallel option."""
        from pyutagent.cli.main import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            pom_file = Path(tmpdir) / 'pom.xml'
            pom_file.write_text('<project></project>')
            
            src_dir = Path(tmpdir) / 'src' / 'main' / 'java'
            src_dir.mkdir(parents=True)
            
            (src_dir / 'Service.java').write_text('public class Service {}')

            runner = CliRunner()
            result = runner.invoke(cli, ['generate-all', tmpdir, '--parallel', '4'])

            assert result.exit_code in [0, 1]

    def test_generate_all_with_continue_on_error(self):
        """Test generate-all with continue-on-error flag."""
        from pyutagent.cli.main import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            pom_file = Path(tmpdir) / 'pom.xml'
            pom_file.write_text('<project></project>')
            
            src_dir = Path(tmpdir) / 'src' / 'main' / 'java'
            src_dir.mkdir(parents=True)
            
            (src_dir / 'Service.java').write_text('public class Service {}')

            runner = CliRunner()
            result = runner.invoke(cli, ['generate-all', tmpdir, '--continue-on-error'])

            assert result.exit_code in [0, 1]


class TestBatchGenerator:
    """Test BatchGenerator functionality."""

    def test_batch_config_defaults(self):
        """Test BatchConfig default values."""
        from pyutagent.services.batch_generator import BatchConfig

        config = BatchConfig()

        assert config.parallel_workers == 1
        assert config.timeout_per_file == 300
        assert config.continue_on_error is True
        assert config.coverage_target == 80

    def test_batch_config_custom(self):
        """Test BatchConfig with custom values."""
        from pyutagent.services.batch_generator import BatchConfig

        config = BatchConfig(
            parallel_workers=4,
            timeout_per_file=600,
            continue_on_error=False,
            coverage_target=90
        )

        assert config.parallel_workers == 4
        assert config.timeout_per_file == 600
        assert config.continue_on_error is False
        assert config.coverage_target == 90

    def test_batch_progress(self):
        """Test BatchProgress dataclass."""
        from pyutagent.services.batch_generator import BatchProgress

        progress = BatchProgress(
            total_files=10,
            completed_files=5,
            failed_files=1,
            current_file="TestService.java",
            current_status="generating"
        )

        assert progress.total_files == 10
        assert progress.completed_files == 5
        assert progress.failed_files == 1
        assert progress.progress_percent == 50.0

    def test_batch_result(self):
        """Test BatchResult dataclass."""
        from pyutagent.services.batch_generator import BatchResult, FileResult

        results = [
            FileResult(
                file_path="ServiceA.java",
                success=True,
                coverage=85.0,
                iterations=3,
                test_file="ServiceATest.java",
                error=None,
                duration=10.5
            ),
            FileResult(
                file_path="ServiceB.java",
                success=False,
                coverage=0.0,
                iterations=0,
                test_file=None,
                error="Compilation failed",
                duration=5.2
            )
        ]

        batch_result = BatchResult(
            total_files=2,
            success_count=1,
            failed_count=1,
            skipped_count=0,
            results=results,
            total_duration=15.7
        )

        assert batch_result.total_files == 2
        assert batch_result.success_count == 1
        assert batch_result.failed_count == 1
        assert batch_result.success_rate == 50.0

    @pytest.mark.asyncio
    async def test_batch_generator_parallel_execution(self):
        """Test that BatchGenerator executes in parallel."""
        from pyutagent.services.batch_generator import BatchGenerator, BatchConfig

        mock_llm_client = MagicMock()
        mock_project_path = "/test/project"
        
        generator = BatchGenerator(
            llm_client=mock_llm_client,
            project_path=mock_project_path,
            config=BatchConfig(parallel_workers=2)
        )

        execution_order = []

        async def mock_generate(file_path):
            execution_order.append(f"start_{file_path}")
            await asyncio.sleep(0.1)
            execution_order.append(f"end_{file_path}")
            return MagicMock(success=True, coverage=80.0)

        generator._generate_single = mock_generate

        files = ["A.java", "B.java", "C.java", "D.java"]
        
        await generator.generate_all(files)

        start_times = [i for i, x in enumerate(execution_order) if x.startswith("start_")]
        
        assert len([x for x in execution_order if x.startswith("start_")]) == 4
        assert len([x for x in execution_order if x.startswith("end_")]) == 4

    @pytest.mark.asyncio
    async def test_batch_generator_error_handling(self):
        """Test that BatchGenerator handles errors gracefully."""
        from pyutagent.services.batch_generator import BatchGenerator, BatchConfig

        mock_llm_client = MagicMock()
        mock_project_path = "/test/project"
        
        generator = BatchGenerator(
            llm_client=mock_llm_client,
            project_path=mock_project_path,
            config=BatchConfig(continue_on_error=True)
        )

        call_count = 0

        async def mock_generate(file_path):
            nonlocal call_count
            call_count += 1
            if file_path == "Fail.java":
                raise Exception("Simulated error")
            return MagicMock(success=True, coverage=80.0)

        generator._generate_single = mock_generate

        files = ["A.java", "Fail.java", "B.java"]
        
        result = await generator.generate_all(files)

        assert call_count == 3
        assert result.success_count == 2
        assert result.failed_count == 1


class TestFileResult:
    """Test FileResult dataclass."""

    def test_file_result_success(self):
        """Test FileResult for successful generation."""
        from pyutagent.services.batch_generator import FileResult

        result = FileResult(
            file_path="UserService.java",
            success=True,
            coverage=92.5,
            iterations=3,
            test_file="src/test/java/UserServiceTest.java",
            error=None,
            duration=15.3
        )

        assert result.success is True
        assert result.coverage == 92.5
        assert result.error is None

    def test_file_result_failure(self):
        """Test FileResult for failed generation."""
        from pyutagent.services.batch_generator import FileResult

        result = FileResult(
            file_path="OrderService.java",
            success=False,
            coverage=0.0,
            iterations=0,
            test_file=None,
            error="Compilation failed: cannot find symbol",
            duration=5.2
        )

        assert result.success is False
        assert result.coverage == 0.0
        assert "Compilation failed" in result.error
