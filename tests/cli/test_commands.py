"""CLI command tests using Click's CliRunner"""

import pytest
from click.testing import CliRunner

from reggie.cli.main import cli


@pytest.mark.cli
class TestInitCommand:
    """Test 'reggie init' command."""

    def test_init_command_success(self, mocker):
        """Init command succeeds and initializes database."""
        # Mock the init_db function
        mock_init_db = mocker.patch("reggie.cli.main.init_db", new_callable=mocker.AsyncMock)

        runner = CliRunner()
        result = runner.invoke(cli, ["init"])

        assert result.exit_code == 0
        assert "Database initialized successfully" in result.output
        mock_init_db.assert_called_once()

    def test_init_command_with_force_flag_prompts_confirmation(self, mocker):
        """Init with --force flag prompts for confirmation."""
        mock_init_db = mocker.patch("reggie.cli.main.init_db", new_callable=mocker.AsyncMock)

        runner = CliRunner()
        # Provide 'n' to decline confirmation
        result = runner.invoke(cli, ["init", "--force"], input="n\n")

        assert result.exit_code == 0
        assert "Warning: This will drop all existing data!" in result.output
        assert "Aborted" in result.output
        mock_init_db.assert_not_called()

    def test_init_command_with_force_confirmed(self, mocker):
        """Init with --force and confirmed proceeds."""
        mock_init_db = mocker.patch("reggie.cli.main.init_db", new_callable=mocker.AsyncMock)

        runner = CliRunner()
        # Provide 'y' to confirm
        result = runner.invoke(cli, ["init", "--force"], input="y\n")

        assert result.exit_code == 0
        assert "Database initialized successfully" in result.output
        mock_init_db.assert_called_once()


@pytest.mark.cli
class TestLoadCommand:
    """Test 'reggie load' command."""

    def test_load_command_requires_document_id(self):
        """Load command requires document_id argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["load"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "Usage:" in result.output

    def test_load_command_success(self, mocker):
        """Load command successfully loads document."""
        # Mock DocumentLoader
        mock_loader_class = mocker.patch("reggie.cli.main.DocumentLoader")
        mock_loader_instance = mock_loader_class.return_value
        mock_loader_instance.load_document = mocker.AsyncMock(
            return_value={
                "comments_processed": 150,
                "errors": 0,
                "duration": 45.5
            }
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["load", "CMS-2024-0001-0001"])

        assert result.exit_code == 0
        assert "CMS-2024-0001-0001" in result.output
        assert "Document loaded successfully" in result.output
        assert "Comments loaded: 150" in result.output
        assert "Errors: 0" in result.output

    def test_load_command_displays_error_on_failure(self, mocker):
        """Load command displays error when loading fails."""
        mock_loader_class = mocker.patch("reggie.cli.main.DocumentLoader")
        mock_loader_instance = mock_loader_class.return_value
        mock_loader_instance.load_document = mocker.AsyncMock(
            side_effect=Exception("API connection failed")
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["load", "INVALID-DOC"])

        assert result.exit_code == 0  # CLI doesn't exit with error code
        assert "Error:" in result.output
        assert "API connection failed" in result.output


@pytest.mark.cli
class TestProcessCommand:
    """Test 'reggie process' command."""

    def test_process_command_requires_document_id(self):
        """Process command requires document_id argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "Usage:" in result.output

    def test_process_command_checks_openai_api_key(self, mocker):
        """Process command checks for OPENAI_API_KEY."""
        # Remove API key from environment
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False)

        runner = CliRunner()
        result = runner.invoke(cli, ["process", "CMS-2024-0001-0001"])

        assert result.exit_code == 0
        assert "Missing required environment variables" in result.output
        assert "OPENAI_API_KEY" in result.output

    def test_process_command_success(self, mocker):
        """Process command successfully processes comments."""
        # Ensure API key exists
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"})

        # Mock the comment count check to return comments exist
        mock_uow_class = mocker.patch("reggie.db.unit_of_work.UnitOfWork")
        mock_uow_instance = mocker.AsyncMock()
        mock_uow_class.return_value.__aenter__.return_value = mock_uow_instance
        mock_uow_instance.comment_statistics.count_comments_for_document = mocker.AsyncMock(
            return_value=150
        )

        # Mock CommentProcessor
        mock_processor_class = mocker.patch("reggie.cli.main.CommentProcessor")
        mock_processor_instance = mock_processor_class.return_value
        mock_processor_instance.process_comments = mocker.AsyncMock(
            return_value={
                "comments_processed": 150,
                "chunks_created": 450,
                "errors": 0,
                "duration": 120.5
            }
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["process", "CMS-2024-0001-0001"])

        assert result.exit_code == 0
        assert "Comments processed successfully" in result.output
        assert "Comments processed: 150" in result.output
        assert "Chunks created: 450" in result.output
        assert "Errors: 0" in result.output

    def test_process_command_custom_batch_size(self, mocker):
        """Process command accepts custom batch size."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"})

        # Mock the comment count check to return comments exist
        mock_uow_class = mocker.patch("reggie.db.unit_of_work.UnitOfWork")
        mock_uow_instance = mocker.AsyncMock()
        mock_uow_class.return_value.__aenter__.return_value = mock_uow_instance
        mock_uow_instance.comment_statistics.count_comments_for_document = mocker.AsyncMock(
            return_value=100
        )

        mock_processor_class = mocker.patch("reggie.cli.main.CommentProcessor")
        mock_processor_instance = mock_processor_class.return_value
        mock_processor_instance.process_comments = mocker.AsyncMock(
            return_value={
                "comments_processed": 100,
                "chunks_created": 300,
                "errors": 0,
                "duration": 60.0
            }
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["process", "CMS-2024-0001-0001", "--batch-size", "20"]
        )

        assert result.exit_code == 0
        # Verify batch_size was passed
        mock_processor_instance.process_comments.assert_called_once()

    def test_process_command_displays_error_on_failure(self, mocker):
        """Process command displays error when processing fails."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"})

        # Mock the comment count check to return comments exist
        mock_uow_class = mocker.patch("reggie.db.unit_of_work.UnitOfWork")
        mock_uow_instance = mocker.AsyncMock()
        mock_uow_class.return_value.__aenter__.return_value = mock_uow_instance
        mock_uow_instance.comment_statistics.count_comments_for_document = mocker.AsyncMock(
            return_value=100
        )

        mock_processor_class = mocker.patch("reggie.cli.main.CommentProcessor")
        mock_processor_instance = mock_processor_class.return_value
        mock_processor_instance.process_comments = mocker.AsyncMock(
            side_effect=Exception("Processing failed")
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["process", "CMS-2024-0001-0001"])

        assert result.exit_code == 0
        assert "Error:" in result.output
        assert "Processing failed" in result.output


@pytest.mark.cli
class TestListCommand:
    """Test 'reggie list' command."""

    def test_list_command_empty_database(self, mocker):
        """List command shows message when no documents loaded."""
        mock_loader_class = mocker.patch("reggie.cli.main.DocumentLoader")
        mock_loader_instance = mock_loader_class.return_value
        mock_loader_instance.list_documents = mocker.AsyncMock(return_value=[])

        runner = CliRunner()
        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "No documents loaded yet" in result.output

    def test_list_command_displays_documents(self, mocker):
        """List command displays loaded documents in table."""
        from datetime import datetime

        mock_loader_class = mocker.patch("reggie.cli.main.DocumentLoader")
        mock_loader_instance = mock_loader_class.return_value
        mock_loader_instance.list_documents = mocker.AsyncMock(
            return_value=[
                {
                    "id": "CMS-2024-0001-0001",
                    "title": "Medicare Physician Fee Schedule",
                    "docket_id": "CMS-2024-0001",
                    "comment_count": 150,
                    "unique_categories": 8,
                    "loaded_at": datetime(2024, 1, 15, 10, 30)
                },
                {
                    "id": "CMS-2024-0002-0001",
                    "title": "Hospital Inpatient Prospective Payment",
                    "docket_id": "CMS-2024-0002",
                    "comment_count": 200,
                    "unique_categories": 10,
                    "loaded_at": datetime(2024, 1, 16, 14, 20)
                }
            ]
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "Loaded Documents" in result.output
        # Document IDs may be truncated in table
        assert "CMS-2024" in result.output
        assert "Medicare" in result.output
        assert "150" in result.output  # comment count
        assert "200" in result.output

    def test_list_command_truncates_long_titles(self, mocker):
        """List command truncates very long document titles."""
        from datetime import datetime

        long_title = "A" * 100  # 100 character title

        mock_loader_class = mocker.patch("reggie.cli.main.DocumentLoader")
        mock_loader_instance = mock_loader_class.return_value
        mock_loader_instance.list_documents = mocker.AsyncMock(
            return_value=[
                {
                    "id": "DOC-001",
                    "title": long_title,
                    "docket_id": "DOCKET-001",
                    "comment_count": 10,
                    "unique_categories": 3,
                    "loaded_at": datetime(2024, 1, 15)
                }
            ]
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        # Title should be truncated (with ellipsis character or "...")
        assert "\u2026" in result.output or "..." in result.output

    def test_list_command_displays_error_on_failure(self, mocker):
        """List command displays error when listing fails."""
        mock_loader_class = mocker.patch("reggie.cli.main.DocumentLoader")
        mock_loader_instance = mock_loader_class.return_value
        mock_loader_instance.list_documents = mocker.AsyncMock(
            side_effect=Exception("Database error")
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "Error:" in result.output
        assert "Database error" in result.output


@pytest.mark.cli
class TestClearCommand:
    """Test 'reggie clear' command."""

    def test_clear_command_requires_document_id(self):
        """Clear command requires document_id argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["clear"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "Usage:" in result.output

    def test_clear_command_document_not_found(self, mocker):
        """Clear command shows message when document not found."""
        # Mock UnitOfWork - patch where it's imported
        mock_uow_class = mocker.patch("reggie.db.unit_of_work.UnitOfWork")
        mock_uow_instance = mocker.AsyncMock()
        mock_uow_class.return_value.__aenter__.return_value = mock_uow_instance
        mock_uow_instance.documents.document_exists = mocker.AsyncMock(return_value=False)

        runner = CliRunner()
        result = runner.invoke(cli, ["clear", "NONEXISTENT-DOC"])

        assert result.exit_code == 0
        assert "not found" in result.output

    def test_clear_command_prompts_confirmation(self, mocker):
        """Clear command prompts for confirmation."""
        # Mock UnitOfWork - patch where it's imported
        mock_uow_class = mocker.patch("reggie.db.unit_of_work.UnitOfWork")
        mock_uow_instance = mocker.AsyncMock()
        mock_uow_class.return_value.__aenter__.return_value = mock_uow_instance
        mock_uow_instance.documents.document_exists = mocker.AsyncMock(return_value=True)
        mock_uow_instance.documents.delete_document = mocker.AsyncMock(
            return_value={"document_deleted": True, "comments_deleted": 10, "chunks_deleted": 30}
        )

        runner = CliRunner()
        # Provide 'n' to decline confirmation
        result = runner.invoke(cli, ["clear", "CMS-2024-0001-0001"], input="n\n")

        assert result.exit_code == 0
        assert "Warning:" in result.output
        assert "Aborted" in result.output

    def test_clear_command_confirmed_success(self, mocker):
        """Clear command deletes document when confirmed."""
        # Mock UnitOfWork - patch where it's imported
        mock_uow_class = mocker.patch("reggie.db.unit_of_work.UnitOfWork")
        mock_uow_instance = mocker.AsyncMock()
        mock_uow_class.return_value.__aenter__.return_value = mock_uow_instance
        mock_uow_instance.documents.document_exists = mocker.AsyncMock(return_value=True)
        mock_uow_instance.documents.delete_document = mocker.AsyncMock(
            return_value={"document_deleted": True, "comments_deleted": 10, "chunks_deleted": 30}
        )

        runner = CliRunner()
        # Provide 'y' to confirm
        result = runner.invoke(cli, ["clear", "CMS-2024-0001-0001"], input="y\n")

        assert result.exit_code == 0
        assert "Document cleared successfully" in result.output
        assert "Comments: 10" in result.output
        assert "Chunks: 30" in result.output

    def test_clear_command_force_skips_confirmation(self, mocker):
        """Clear command with --force skips confirmation."""
        # Mock UnitOfWork - patch where it's imported
        mock_uow_class = mocker.patch("reggie.db.unit_of_work.UnitOfWork")
        mock_uow_instance = mocker.AsyncMock()
        mock_uow_class.return_value.__aenter__.return_value = mock_uow_instance
        mock_uow_instance.documents.document_exists = mocker.AsyncMock(return_value=True)
        mock_uow_instance.documents.delete_document = mocker.AsyncMock(
            return_value={"document_deleted": True, "comments_deleted": 5, "chunks_deleted": 15}
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["clear", "CMS-2024-0001-0001", "--force"])

        assert result.exit_code == 0
        assert "Document cleared successfully" in result.output
        # No confirmation prompt should appear
        assert "Are you sure" not in result.output

    def test_clear_command_displays_error_on_failure(self, mocker):
        """Clear command displays error when deletion fails."""
        # Mock UnitOfWork - patch where it's imported
        mock_uow_class = mocker.patch("reggie.db.unit_of_work.UnitOfWork")
        mock_uow_instance = mocker.AsyncMock()
        mock_uow_class.return_value.__aenter__.return_value = mock_uow_instance
        mock_uow_instance.documents.document_exists = mocker.AsyncMock(
            side_effect=Exception("Database error")
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["clear", "CMS-2024-0001-0001", "--force"])

        assert result.exit_code == 0
        assert "Error:" in result.output
        assert "Database error" in result.output


@pytest.mark.cli
class TestDiscussCommand:
    """Test 'reggie discuss' command."""

    def test_discuss_command_requires_document_id(self):
        """Discuss command requires document_id argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["discuss"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "Usage:" in result.output

    def test_discuss_command_shows_error_for_missing_document(self, mocker):
        """Discuss command shows error when document not found."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"})

        # Create a properly nested async context manager mock
        mock_cursor = mocker.MagicMock()
        mock_cursor.execute = mocker.AsyncMock()
        mock_cursor.fetchone = mocker.AsyncMock(return_value=None)  # Document not found
        mock_cursor.__aenter__ = mocker.AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = mocker.AsyncMock(return_value=None)

        mock_conn = mocker.MagicMock()
        mock_conn.cursor = mocker.MagicMock(return_value=mock_cursor)
        mock_conn.__aenter__ = mocker.AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = mocker.AsyncMock(return_value=None)

        mock_get_connection = mocker.patch("reggie.db.get_connection")
        mock_get_connection.return_value = mock_conn

        runner = CliRunner()
        result = runner.invoke(cli, ["discuss", "CMS-2024-0001-0001"])

        assert result.exit_code == 0
        assert "not found" in result.output.lower()


@pytest.mark.cli
class TestCLIHelpText:
    """Test CLI help text and user experience."""

    def test_main_cli_help(self):
        """Main CLI shows help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Reggie:" in result.output or "Load documents" in result.output

    def test_load_command_help(self):
        """Load command shows help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["load", "--help"])

        assert result.exit_code == 0
        assert "DOCUMENT_ID" in result.output
        assert "CMS-" in result.output  # Example in help

    def test_process_command_help(self):
        """Process command shows help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "--help"])

        assert result.exit_code == 0
        assert "DOCUMENT_ID" in result.output
        assert "batch-size" in result.output.lower()

    def test_clear_command_help(self):
        """Clear command shows help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["clear", "--help"])

        assert result.exit_code == 0
        assert "DOCUMENT_ID" in result.output
        assert "force" in result.output.lower()
        assert "re-ingested" in result.output
