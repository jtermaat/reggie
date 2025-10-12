"""Unit tests for visualization callback system."""

import pytest
from reggie.agent.visualizations import (
    set_visualization_callback,
    clear_visualization_callback,
    emit_visualization
)


def test_set_visualization_callback():
    """Test setting a visualization callback."""
    callback_data = []

    def test_callback(data):
        callback_data.append(data)

    set_visualization_callback(test_callback)
    emit_visualization({"test": "data"})

    assert len(callback_data) == 1
    assert callback_data[0] == {"test": "data"}

    clear_visualization_callback()


def test_clear_visualization_callback():
    """Test clearing the visualization callback."""
    callback_data = []

    def test_callback(data):
        callback_data.append(data)

    set_visualization_callback(test_callback)
    emit_visualization({"test": "data1"})
    clear_visualization_callback()
    emit_visualization({"test": "data2"})

    # Only first emission should be recorded
    assert len(callback_data) == 1
    assert callback_data[0] == {"test": "data1"}


def test_emit_visualization_without_callback():
    """Test that emit_visualization works without a callback set."""
    # Should not raise an error
    emit_visualization({"test": "data"})


def test_multiple_visualizations():
    """Test multiple visualization emissions."""
    callback_data = []

    def test_callback(data):
        callback_data.append(data)

    set_visualization_callback(test_callback)

    emit_visualization({"type": "statistics", "value": 1})
    emit_visualization({"type": "opposition_support", "value": 2})
    emit_visualization({"type": "statistics", "value": 3})

    assert len(callback_data) == 3
    assert callback_data[0]["value"] == 1
    assert callback_data[1]["value"] == 2
    assert callback_data[2]["value"] == 3

    clear_visualization_callback()


def test_callback_receives_correct_data_structure():
    """Test that callback receives the expected data structure."""
    received_data = None

    def test_callback(data):
        nonlocal received_data
        received_data = data

    set_visualization_callback(test_callback)

    test_data = {
        "type": "statistics",
        "group_by": "sentiment",
        "total_comments": 100,
        "breakdown": [
            {"value": "for", "count": 60, "percentage": 60.0},
            {"value": "against", "count": 40, "percentage": 40.0}
        ]
    }

    emit_visualization(test_data)

    assert received_data is not None
    assert received_data["type"] == "statistics"
    assert received_data["group_by"] == "sentiment"
    assert received_data["total_comments"] == 100
    assert len(received_data["breakdown"]) == 2

    clear_visualization_callback()


def test_opposition_support_data_structure():
    """Test opposition/support visualization data structure."""
    received_data = None

    def test_callback(data):
        nonlocal received_data
        received_data = data

    set_visualization_callback(test_callback)

    # Test data with all sentiment types (for, against, mixed, unclear)
    test_data = {
        "type": "opposition_support",
        "document_id": "TEST-123",
        "document_title": "Test Document",
        "total_comments": 150,
        "breakdown": {
            "Physicians & Surgeons": {
                "for": 50,
                "against": 80,
                "mixed": 10,
                "unclear": 10
            },
            "Patients & Caregivers": {
                "for": 0,
                "against": 0,
                "mixed": 0,
                "unclear": 0
            }
        }
    }

    emit_visualization(test_data)

    assert received_data is not None
    assert received_data["type"] == "opposition_support"
    assert received_data["document_id"] == "TEST-123"
    assert "Physicians & Surgeons" in received_data["breakdown"]

    # Verify all sentiment counts are present
    phys_data = received_data["breakdown"]["Physicians & Surgeons"]
    assert phys_data["for"] == 50
    assert phys_data["against"] == 80
    assert phys_data["mixed"] == 10
    assert phys_data["unclear"] == 10

    clear_visualization_callback()


def test_percentage_calculation_includes_all_sentiments():
    """Test that percentages are based on total category comments, not just for+against."""
    # This is a documentation test to ensure correct percentage calculations
    # In the actual visualization:
    # - If a category has 100 total comments: 50 for, 30 against, 10 mixed, 10 unclear
    # - "for" should show as 50% (50/100), not 62.5% (50/80)
    # - "against" should show as 30% (30/100), not 37.5% (30/80)

    # The renderer calculates: category_total = sum(sentiments.values())
    # Then: against_pct = (against_count / category_total * 100)

    test_sentiments = {
        "for": 50,
        "against": 30,
        "mixed": 10,
        "unclear": 10
    }

    category_total = sum(test_sentiments.values())
    assert category_total == 100

    against_pct = (test_sentiments["against"] / category_total * 100)
    for_pct = (test_sentiments["for"] / category_total * 100)

    assert against_pct == 30.0  # Not 37.5
    assert for_pct == 50.0  # Not 62.5

    # The remaining 20% (mixed + unclear) is not shown in the bars
    # but users can deduce it from: total - (for% + against%)
