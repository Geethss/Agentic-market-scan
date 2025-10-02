import pytest
import os
import json
import csv
import tempfile
import shutil
from unittest.mock import patch, Mock
from typing import List, Dict, Any

from backend.agents.write import (
    write_matrix_csv,
    write_vendors_json,
    _escape_md,
    write_brief_md,
    _build_html_report,
    write_html_report,
    write_pdf_from_html,
    write_reports
)


class TestCSVGeneration:
    """Test CSV matrix generation"""

    def test_write_matrix_csv_basic(self):
        """Test basic CSV matrix generation"""
        vendors = [
            {
                "name": "Vendor A",
                "score": 0.85,
                "pricing": "Free tier available",
                "key_features": ["Feature 1", "Feature 2"],
                "integrations": ["Slack", "API"],
                "evidence": [
                    {"url": "https://example1.com", "text": "Evidence 1"},
                    {"url": "https://example2.com", "text": "Evidence 2"}
                ]
            },
            {
                "name": "Vendor B",
                "score": 0.72,
                "pricing": "Paid plans from $10/month",
                "key_features": ["Feature 3"],
                "integrations": ["Webhook"],
                "evidence": [{"url": "https://example3.com", "text": "Evidence 3"}]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name
        
        try:
            write_matrix_csv(vendors, temp_path)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 2
            assert rows[0]["vendor"] == "Vendor A"
            assert rows[0]["score"] == "0.85"
            assert rows[0]["pricing"] == "Free tier available"
            assert rows[0]["key_features"] == "Feature 1; Feature 2"
            assert rows[0]["integrations"] == "Slack; API"
            assert rows[0]["evidence_count"] == "2"
            
            assert rows[1]["vendor"] == "Vendor B"
            assert rows[1]["score"] == "0.72"
            assert rows[1]["evidence_count"] == "1"
            
        finally:
            os.unlink(temp_path)

    def test_write_matrix_csv_missing_fields(self):
        """Test CSV generation with missing vendor fields"""
        vendors = [
            {
                "name": "Vendor A",
                "score": 0.5,
                # Missing pricing, key_features, integrations
                "evidence": []
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name
        
        try:
            write_matrix_csv(vendors, temp_path)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 1
            assert rows[0]["vendor"] == "Vendor A"
            assert rows[0]["score"] == "0.5"
            assert rows[0]["pricing"] == ""
            assert rows[0]["key_features"] == ""
            assert rows[0]["integrations"] == ""
            assert rows[0]["evidence_count"] == "0"
            
        finally:
            os.unlink(temp_path)

    def test_write_matrix_csv_empty_list(self):
        """Test CSV generation with empty vendor list"""
        vendors = []
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name
        
        try:
            write_matrix_csv(vendors, temp_path)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 0
            
        finally:
            os.unlink(temp_path)


class TestJSONGeneration:
    """Test JSON vendor export"""

    def test_write_vendors_json_basic(self):
        """Test basic JSON vendor export"""
        vendors = [
            {
                "name": "Vendor A",
                "score": 0.85,
                "pricing": "Free tier available",
                "key_features": ["Feature 1", "Feature 2"],
                "evidence": [{"url": "https://example.com", "text": "Evidence"}]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            write_vendors_json(vendors, temp_path)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert len(data) == 1
            assert data[0]["name"] == "Vendor A"
            assert data[0]["score"] == 0.85
            assert data[0]["pricing"] == "Free tier available"
            assert data[0]["key_features"] == ["Feature 1", "Feature 2"]
            
        finally:
            os.unlink(temp_path)

    def test_write_vendors_json_unicode(self):
        """Test JSON export with unicode characters"""
        vendors = [
            {
                "name": "Vendor with émojis 🚀",
                "pricing": "€10/month",
                "key_features": ["Feature with ñ", "Another feature"]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            write_vendors_json(vendors, temp_path)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert data[0]["name"] == "Vendor with émojis 🚀"
            assert data[0]["pricing"] == "€10/month"
            assert "ñ" in data[0]["key_features"][0]
            
        finally:
            os.unlink(temp_path)


class TestMarkdownEscaping:
    """Test markdown escaping utility"""

    def test_escape_md_basic(self):
        """Test basic markdown escaping"""
        assert _escape_md("Normal text") == "Normal text"
        assert _escape_md("Text with\nnewlines") == "Text with newlines"
        assert _escape_md("Text with\rcarriage returns") == "Text with carriage returns"
        assert _escape_md("  Text with spaces  ") == "Text with spaces"

    def test_escape_md_none_and_empty(self):
        """Test escaping None and empty strings"""
        assert _escape_md(None) == ""
        assert _escape_md("") == ""
        assert _escape_md("   ") == ""


class TestMarkdownGeneration:
    """Test markdown brief generation"""

    def test_write_brief_md_basic(self):
        """Test basic markdown brief generation"""
        topic = "AI Tools"
        vendors = [
            {
                "name": "Vendor A",
                "score": 0.85,
                "pricing": "Free tier available",
                "key_features": ["Feature 1", "Feature 2"],
                "target_users": "SMBs",
                "integrations": ["Slack"],
                "security_compliance": ["SOC2"],
                "evidence": [
                    {
                        "url": "https://example1.com",
                        "title": "Example Article",
                        "text": "This is evidence text",
                        "snippet_id": "snip_1"
                    }
                ]
            },
            {
                "name": "Vendor B",
                "score": 0.72,
                "pricing": "Paid plans",
                "evidence": []
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_path = f.name
        
        try:
            write_brief_md(topic, vendors, temp_path)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "# Market Scan Brief" in content
            assert "**Topic:** AI Tools" in content
            assert "## Top Vendors" in content
            assert "### Vendor A — score: 0.85" in content
            assert "### Vendor B — score: 0.72" in content
            assert "**Pricing:** Free tier available" in content
            assert "**Key features:** Feature 1, Feature 2" in content
            assert "**Target users:** SMBs" in content
            assert "**Integrations:** Slack" in content
            assert "**Security / Compliance:** SOC2" in content
            assert "**Evidence count:** 1" in content
            assert "[Example Article](https://example1.com) (snippet: snip_1)" in content
            assert "> This is evidence text" in content
            assert "## All Vendors (summary)" in content
            
        finally:
            os.unlink(temp_path)

    def test_write_brief_md_top_k_limit(self):
        """Test markdown generation with top_k limit"""
        topic = "Test Topic"
        vendors = [
            {"name": f"Vendor {i}", "score": 0.9 - i * 0.1, "evidence": []}
            for i in range(15)
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_path = f.name
        
        try:
            write_brief_md(topic, vendors, temp_path, top_k=5)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Should only show top 5 vendors in detailed section
            assert content.count("### Vendor") == 5
            # But should show all vendors in summary section
            assert content.count("- Vendor") == 15
            
        finally:
            os.unlink(temp_path)

    def test_write_brief_md_long_evidence(self):
        """Test markdown generation with long evidence text"""
        topic = "Test Topic"
        vendors = [
            {
                "name": "Vendor A",
                "score": 0.8,
                "evidence": [
                    {
                        "url": "https://example.com",
                        "text": "This is a very long evidence text that should be truncated when it exceeds 300 characters. " * 10
                    }
                ]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_path = f.name
        
        try:
            write_brief_md(topic, vendors, temp_path)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Should contain truncated text with ellipsis
            assert "..." in content
            # Should not contain the full long text
            assert "This is a very long evidence text that should be truncated when it exceeds 300 characters. This is a very long evidence text that should be truncated when it exceeds 300 characters. This is a very long evidence text that should be truncated when it exceeds 300 characters. This is a very long evidence text that should be truncated when it exceeds 300 characters. This is a very long evidence text that should be truncated when it exceeds 300 characters. This is a very long evidence text that should be truncated when it exceeds 300 characters. This is a very long evidence text that should be truncated when it exceeds 300 characters. This is a very long evidence text that should be truncated when it exceeds 300 characters. This is a very long evidence text that should be truncated when it exceeds 300 characters. This is a very long evidence text that should be truncated when it exceeds 300 characters." not in content
            
        finally:
            os.unlink(temp_path)


class TestHTMLGeneration:
    """Test HTML report generation"""

    def test_build_html_report_basic(self):
        """Test basic HTML report building"""
        topic = "AI Tools"
        vendors = [
            {
                "name": "Vendor A",
                "score": 0.85,
                "pricing": "Free tier available",
                "key_features": ["Feature 1", "Feature 2"],
                "integrations": ["Slack"],
                "evidence": [
                    {
                        "url": "https://example1.com",
                        "title": "Example Article",
                        "text": "This is evidence text"
                    }
                ]
            }
        ]
        
        html_content = _build_html_report(topic, vendors)
        
        assert "<!doctype html>" in html_content
        assert "<html>" in html_content
        assert "<head>" in html_content
        assert "<title>Market Scan Report</title>" in html_content
        assert "<style>" in html_content
        assert "body{font-family:Inter,Segoe UI,Roboto,Arial,sans-serif" in html_content
        assert "<h1>Market Scan — AI Tools</h1>" in html_content
        assert "<h2>Top vendors</h2>" in html_content
        assert "<div class='vendor'>" in html_content
        assert "<h3>Vendor A — <small>0.85</small></h3>" in html_content
        assert "<strong>Pricing:</strong> Free tier available" in html_content
        assert "<strong>Key features:</strong> Feature 1, Feature 2" in html_content
        assert "<strong>Integrations:</strong> Slack" in html_content
        assert "<strong>Evidence count:</strong> 1" in html_content
        assert "<a href='https://example1.com' target='_blank'>Example Article</a>" in html_content
        assert "<blockquote>This is evidence text</blockquote>" in html_content

    def test_build_html_report_html_escaping(self):
        """Test HTML escaping in report"""
        topic = "Test <script>alert('xss')</script>"
        vendors = [
            {
                "name": "Vendor with <script>alert('xss')</script>",
                "score": 0.8,
                "pricing": "Price with & symbols",
                "evidence": [
                    {
                        "url": "https://example.com",
                        "title": "Title with <script>alert('xss')</script>",
                        "text": "Text with <script>alert('xss')</script>"
                    }
                ]
            }
        ]
        
        html_content = _build_html_report(topic, vendors)
        
        # Should escape HTML entities (Python's html.escape uses &#x27; for single quotes)
        assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html_content
        assert "&amp; symbols" in html_content
        # Should not contain unescaped script tags
        assert "<script>alert('xss')</script>" not in html_content

    def test_build_html_report_long_evidence(self):
        """Test HTML report with long evidence text"""
        topic = "Test Topic"
        vendors = [
            {
                "name": "Vendor A",
                "score": 0.8,
                "evidence": [
                    {
                        "url": "https://example.com",
                        "text": "This is a very long evidence text that should be truncated when it exceeds 400 characters. " * 10
                    }
                ]
            }
        ]
        
        html_content = _build_html_report(topic, vendors)
        
        # Should contain truncated text (400 chars max)
        assert len([line for line in html_content.split('\n') if 'This is a very long evidence text' in line][0]) < 500

    def test_write_html_report_file(self):
        """Test writing HTML report to file"""
        topic = "Test Topic"
        vendors = [{"name": "Vendor A", "score": 0.8, "evidence": []}]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
            temp_path = f.name
        
        try:
            write_html_report(topic, vendors, temp_path)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "<!doctype html>" in content
            assert "Vendor A" in content
            
        finally:
            os.unlink(temp_path)


class TestPDFGeneration:
    """Test PDF generation with weasyprint"""

    def test_write_pdf_from_html_success(self):
        """Test successful PDF generation"""
        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
            html_path = f.name
            f.write("<!DOCTYPE html><html><body><h1>Test</h1></body></html>")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pdf') as f:
            pdf_path = f.name
        
        try:
            # Mock weasyprint import and HTML class
            with patch('builtins.__import__') as mock_import:
                mock_weasyprint = Mock()
                mock_html_class = Mock()
                mock_html_instance = Mock()
                mock_html_class.return_value = mock_html_instance
                mock_weasyprint.HTML = mock_html_class
                mock_import.return_value = mock_weasyprint
                
                result = write_pdf_from_html(html_path, pdf_path)
                
                assert result is True
                mock_html_class.assert_called_once_with(html_path)
                mock_html_instance.write_pdf.assert_called_once_with(pdf_path)
                
        finally:
            os.unlink(html_path)
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def test_write_pdf_from_html_weasyprint_not_installed(self):
        """Test PDF generation when weasyprint is not installed"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
            html_path = f.name
            f.write("<!DOCTYPE html><html><body><h1>Test</h1></body></html>")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pdf') as f:
            pdf_path = f.name
        
        try:
            # Mock import error
            with patch('builtins.__import__', side_effect=ImportError("No module named 'weasyprint'")):
                result = write_pdf_from_html(html_path, pdf_path)
                
                assert result is False
                
        finally:
            os.unlink(html_path)
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def test_write_pdf_from_html_generation_failure(self):
        """Test PDF generation failure"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
            html_path = f.name
            f.write("<!DOCTYPE html><html><body><h1>Test</h1></body></html>")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pdf') as f:
            pdf_path = f.name
        
        try:
            # Mock weasyprint import and HTML class to raise exception
            with patch('builtins.__import__') as mock_import:
                mock_weasyprint = Mock()
                mock_html_class = Mock()
                mock_html_instance = Mock()
                mock_html_class.return_value = mock_html_instance
                mock_weasyprint.HTML = mock_html_class
                mock_import.return_value = mock_weasyprint
                mock_html_instance.write_pdf.side_effect = Exception("PDF generation failed")
                
                result = write_pdf_from_html(html_path, pdf_path)
                
                assert result is False
                
        finally:
            os.unlink(html_path)
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)


class TestMainWriteFunction:
    """Test main write_reports function"""

    def test_write_reports_basic(self):
        """Test basic report generation"""
        topic = "AI Tools"
        vendors = [
            {
                "name": "Vendor A",
                "score": 0.85,
                "pricing": "Free tier",
                "key_features": ["Feature 1"],
                "integrations": ["Slack"],
                "evidence": [{"url": "https://example.com", "text": "Evidence"}]
            }
        ]
        sources = [{"url": "https://source.com", "title": "Source"}]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_reports(temp_dir, topic, vendors, sources, do_pdf=False)
            
            assert "matrix" in artifacts
            assert "vendors_json" in artifacts
            assert "brief_md" in artifacts
            assert "report_html" in artifacts
            assert "sources_json" in artifacts
            assert "report_pdf" not in artifacts  # PDF not requested
            
            # Check that files exist
            for artifact_type, path in artifacts.items():
                assert os.path.exists(path)
                assert os.path.getsize(path) > 0

    def test_write_reports_with_pdf(self):
        """Test report generation with PDF"""
        topic = "AI Tools"
        vendors = [{"name": "Vendor A", "score": 0.8, "evidence": []}]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock PDF generation to succeed and create the file
            def mock_write_pdf(html_path, pdf_path):
                # Create the PDF file to simulate successful generation
                with open(pdf_path, 'w') as f:
                    f.write("Mock PDF content")
                return True
            
            with patch('backend.agents.write.write_pdf_from_html', side_effect=mock_write_pdf):
                artifacts = write_reports(temp_dir, topic, vendors, do_pdf=True)
                
                assert "report_pdf" in artifacts
                assert os.path.exists(artifacts["report_pdf"])

    def test_write_reports_pdf_failure(self):
        """Test report generation when PDF fails"""
        topic = "AI Tools"
        vendors = [{"name": "Vendor A", "score": 0.8, "evidence": []}]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock PDF generation to fail
            with patch('backend.agents.write.write_pdf_from_html', return_value=False):
                artifacts = write_reports(temp_dir, topic, vendors, do_pdf=True)
                
                assert "report_pdf" not in artifacts  # PDF failed, not included

    def test_write_reports_no_sources(self):
        """Test report generation without sources"""
        topic = "AI Tools"
        vendors = [{"name": "Vendor A", "score": 0.8, "evidence": []}]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_reports(temp_dir, topic, vendors, sources=None)
            
            assert "sources_json" not in artifacts

    def test_write_reports_sources_write_failure(self):
        """Test report generation when sources write fails"""
        topic = "AI Tools"
        vendors = [{"name": "Vendor A", "score": 0.8, "evidence": []}]
        sources = [{"url": "https://source.com", "title": "Source"}]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock file write to fail only for sources.json
            original_open = open
            def mock_open(path, *args, **kwargs):
                if path.endswith('sources.json'):
                    raise IOError("Write failed")
                return original_open(path, *args, **kwargs)
            
            with patch('builtins.open', side_effect=mock_open):
                artifacts = write_reports(temp_dir, topic, vendors, sources)
                
                # Should still generate other artifacts
                assert "matrix" in artifacts
                assert "vendors_json" in artifacts
                assert "brief_md" in artifacts
                assert "report_html" in artifacts
                # Sources should not be included due to failure
                assert "sources_json" not in artifacts

    def test_write_reports_creates_directory(self):
        """Test that write_reports creates the output directory"""
        topic = "AI Tools"
        vendors = [{"name": "Vendor A", "score": 0.8, "evidence": []}]
        
        with tempfile.TemporaryDirectory() as base_dir:
            run_dir = os.path.join(base_dir, "new_run_directory")
            
            # Directory should not exist initially
            assert not os.path.exists(run_dir)
            
            artifacts = write_reports(run_dir, topic, vendors)
            
            # Directory should be created
            assert os.path.exists(run_dir)
            assert os.path.isdir(run_dir)
            
            # All artifacts should be in the created directory
            for artifact_type, path in artifacts.items():
                assert path.startswith(run_dir)

    def test_write_reports_file_contents(self):
        """Test that generated files contain expected content"""
        topic = "AI Tools"
        vendors = [
            {
                "name": "Vendor A",
                "score": 0.85,
                "pricing": "Free tier",
                "key_features": ["Feature 1"],
                "integrations": ["Slack"],
                "evidence": [{"url": "https://example.com", "text": "Evidence"}]
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_reports(temp_dir, topic, vendors)
            
            # Check CSV content
            with open(artifacts["matrix"], 'r', encoding='utf-8') as f:
                csv_content = f.read()
                assert "Vendor A" in csv_content
                assert "0.85" in csv_content
                assert "Free tier" in csv_content
            
            # Check JSON content
            with open(artifacts["vendors_json"], 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                assert json_data[0]["name"] == "Vendor A"
                assert json_data[0]["score"] == 0.85
            
            # Check Markdown content
            with open(artifacts["brief_md"], 'r', encoding='utf-8') as f:
                md_content = f.read()
                assert "# Market Scan Brief" in md_content
                assert "AI Tools" in md_content
                assert "Vendor A" in md_content
            
            # Check HTML content
            with open(artifacts["report_html"], 'r', encoding='utf-8') as f:
                html_content = f.read()
                assert "<!doctype html>" in html_content
                assert "Vendor A" in html_content
                assert "AI Tools" in html_content


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_vendors_list(self):
        """Test handling of empty vendors list"""
        topic = "Empty Topic"
        vendors = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_reports(temp_dir, topic, vendors)
            
            # Should still generate all artifacts
            assert "matrix" in artifacts
            assert "vendors_json" in artifacts
            assert "brief_md" in artifacts
            assert "report_html" in artifacts
            
            # Check that files are created but may be empty or minimal
            for artifact_type, path in artifacts.items():
                assert os.path.exists(path)

    def test_vendors_with_none_values(self):
        """Test handling of vendors with None values"""
        topic = "Test Topic"
        vendors = [
            {
                "name": "Vendor A",
                "score": None,
                "pricing": None,
                "key_features": None,
                "integrations": None,
                "evidence": None
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_reports(temp_dir, topic, vendors)
            
            # Should handle None values gracefully
            assert "matrix" in artifacts
            assert "vendors_json" in artifacts
            
            # Check CSV handles None values
            with open(artifacts["matrix"], 'r', encoding='utf-8') as f:
                csv_content = f.read()
                assert "Vendor A" in csv_content

    def test_special_characters_in_topic(self):
        """Test handling of special characters in topic"""
        topic = "AI Tools with émojis 🚀 & symbols <>&\"'"
        vendors = [{"name": "Vendor A", "score": 0.8, "evidence": []}]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = write_reports(temp_dir, topic, vendors)
            
            # Should handle special characters in all formats
            with open(artifacts["brief_md"], 'r', encoding='utf-8') as f:
                md_content = f.read()
                assert "émojis 🚀" in md_content
            
            with open(artifacts["report_html"], 'r', encoding='utf-8') as f:
                html_content = f.read()
                # Should be HTML escaped
                assert "&amp; symbols" in html_content
                assert "&lt;&gt;&amp;" in html_content
