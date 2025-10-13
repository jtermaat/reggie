"""Unit tests for database repository functions."""

import pytest
from reggie.db.repository import CommentRepository
from reggie.models.agent import StatisticsBreakdownItem


class TestStatisticsSorting:
    """Tests for statistics breakdown sorting."""

    def test_sort_breakdown_sentiment_fixed_order(self):
        """Test that sentiment breakdown uses fixed order: for, against, mixed, unclear."""
        # Create unsorted breakdown items
        breakdown = [
            StatisticsBreakdownItem(value="unclear", count=10, percentage=10.0),
            StatisticsBreakdownItem(value="for", count=50, percentage=50.0),
            StatisticsBreakdownItem(value="mixed", count=15, percentage=15.0),
            StatisticsBreakdownItem(value="against", count=25, percentage=25.0),
        ]

        # Sort using the repository method
        sorted_breakdown = CommentRepository._sort_breakdown_for_visualization(
            breakdown, "sentiment"
        )

        # Verify order
        assert len(sorted_breakdown) == 4
        assert sorted_breakdown[0].value == "for"
        assert sorted_breakdown[1].value == "against"
        assert sorted_breakdown[2].value == "mixed"
        assert sorted_breakdown[3].value == "unclear"

    def test_sort_breakdown_category_alphabetical(self):
        """Test that category breakdown is sorted alphabetically."""
        breakdown = [
            StatisticsBreakdownItem(value="Physicians & Surgeons", count=100, percentage=40.0),
            StatisticsBreakdownItem(value="Advocacy Organizations", count=80, percentage=32.0),
            StatisticsBreakdownItem(value="Patients & Caregivers", count=70, percentage=28.0),
        ]

        sorted_breakdown = CommentRepository._sort_breakdown_for_visualization(
            breakdown, "category"
        )

        assert len(sorted_breakdown) == 3
        assert sorted_breakdown[0].value == "Advocacy Organizations"
        assert sorted_breakdown[1].value == "Patients & Caregivers"
        assert sorted_breakdown[2].value == "Physicians & Surgeons"

    def test_sort_breakdown_topic_alphabetical(self):
        """Test that topic breakdown is sorted alphabetically."""
        breakdown = [
            StatisticsBreakdownItem(value="Workforce Concerns", count=50, percentage=25.0),
            StatisticsBreakdownItem(value="Access to Care", count=100, percentage=50.0),
            StatisticsBreakdownItem(value="Quality & Safety", count=50, percentage=25.0),
        ]

        sorted_breakdown = CommentRepository._sort_breakdown_for_visualization(
            breakdown, "topic"
        )

        assert len(sorted_breakdown) == 3
        assert sorted_breakdown[0].value == "Access to Care"
        assert sorted_breakdown[1].value == "Quality & Safety"
        assert sorted_breakdown[2].value == "Workforce Concerns"

    def test_sort_breakdown_case_insensitive(self):
        """Test that alphabetical sorting is case-insensitive."""
        breakdown = [
            StatisticsBreakdownItem(value="zebra", count=10, percentage=10.0),
            StatisticsBreakdownItem(value="Apple", count=30, percentage=30.0),
            StatisticsBreakdownItem(value="banana", count=60, percentage=60.0),
        ]

        sorted_breakdown = CommentRepository._sort_breakdown_for_visualization(
            breakdown, "category"
        )

        assert len(sorted_breakdown) == 3
        assert sorted_breakdown[0].value == "Apple"
        assert sorted_breakdown[1].value == "banana"
        assert sorted_breakdown[2].value == "zebra"

    def test_sort_breakdown_sentiment_case_insensitive(self):
        """Test that sentiment sorting handles different cases."""
        breakdown = [
            StatisticsBreakdownItem(value="Mixed", count=10, percentage=10.0),
            StatisticsBreakdownItem(value="FOR", count=50, percentage=50.0),
            StatisticsBreakdownItem(value="Against", count=25, percentage=25.0),
            StatisticsBreakdownItem(value="unclear", count=15, percentage=15.0),
        ]

        sorted_breakdown = CommentRepository._sort_breakdown_for_visualization(
            breakdown, "sentiment"
        )

        assert len(sorted_breakdown) == 4
        assert sorted_breakdown[0].value == "FOR"
        assert sorted_breakdown[1].value == "Against"
        assert sorted_breakdown[2].value == "Mixed"
        assert sorted_breakdown[3].value == "unclear"

    def test_sort_breakdown_unknown_sentiment_goes_last(self):
        """Test that unknown sentiment values are placed at the end."""
        breakdown = [
            StatisticsBreakdownItem(value="unknown", count=5, percentage=5.0),
            StatisticsBreakdownItem(value="for", count=50, percentage=50.0),
            StatisticsBreakdownItem(value="weird_value", count=10, percentage=10.0),
            StatisticsBreakdownItem(value="against", count=35, percentage=35.0),
        ]

        sorted_breakdown = CommentRepository._sort_breakdown_for_visualization(
            breakdown, "sentiment"
        )

        assert len(sorted_breakdown) == 4
        assert sorted_breakdown[0].value == "for"
        assert sorted_breakdown[1].value == "against"
        # Unknown values should be at the end (order between them doesn't matter)
        assert sorted_breakdown[2].value in ["unknown", "weird_value"]
        assert sorted_breakdown[3].value in ["unknown", "weird_value"]

    def test_sort_breakdown_empty_list(self):
        """Test sorting an empty breakdown list."""
        breakdown = []
        sorted_breakdown = CommentRepository._sort_breakdown_for_visualization(
            breakdown, "sentiment"
        )
        assert sorted_breakdown == []

    def test_sort_breakdown_single_item(self):
        """Test sorting a single-item breakdown list."""
        breakdown = [
            StatisticsBreakdownItem(value="for", count=100, percentage=100.0),
        ]
        sorted_breakdown = CommentRepository._sort_breakdown_for_visualization(
            breakdown, "sentiment"
        )
        assert len(sorted_breakdown) == 1
        assert sorted_breakdown[0].value == "for"
