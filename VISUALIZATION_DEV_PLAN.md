# Visualization Development Plan

This document outlines the ordered tasks required to implement visualization features for Reggie. The implementation is divided into four phases, with each phase building on the previous one.

---

## Overview

### Features to Implement
1. **Auto-visualization during agent conversations**: When the agent calls `get_statistics()`, automatically display a bar chart of the results
2. **`reggie visualize` command**: New CLI command that displays opposition/support breakdown by category using centered horizontal bars

### Design Principles
- Maintain separation of concerns (visualization logic separate from business logic)
- Use callback mechanism (similar to existing status callbacks)
- Leverage Rich library for rendering
- Percentage-based bar scaling for visual clarity

---

## Phase 1: Core Infrastructure

### Task 1.1: Create Visualization Callback System
**File**: `reggie/agent/visualizations.py`

Create a new module similar to `status.py` that provides:
- Global visualization callback registration
- `set_visualization_callback(callback)` function
- `clear_visualization_callback()` function
- `emit_visualization(data)` function

**Implementation Notes**:
- Follow the same pattern as `reggie/agent/status.py`
- Callback should accept structured data (dict or Pydantic model)
- No rendering logic in this file - pure callback mechanism

**Dependencies**: None

---

### Task 1.2: Create Renderer Module Structure
**Files**:
- `reggie/agent/renderers/__init__.py`
- `reggie/agent/renderers/statistics_renderer.py`

Create the directory structure and base renderer file.

**Implementation Notes**:
- `statistics_renderer.py` will contain all Rich-based rendering logic
- Keep rendering completely separate from data/business logic
- Export main rendering functions from `__init__.py` for easy imports

**Dependencies**: Task 1.1

---

### Task 1.3: Define Visualization Data Models
**File**: `reggie/models/visualization.py`

Create Pydantic models for visualization data:
- `VisualizationData` (base model with type field)
- `StatisticsVisualizationData` (for single-dimension stats)
- `OppositionSupportVisualizationData` (for sentiment-by-category data)

**Implementation Notes**:
- Include all necessary metadata (group_by, filters, totals, etc.)
- Use discriminated unions for type safety
- These models are passed through the callback system

**Dependencies**: None

---

## Phase 2: Auto-Visualization for Agent Tool Calls

### Task 2.1: Implement Single-Dimension Bar Chart Renderer
**File**: `reggie/agent/renderers/statistics_renderer.py`

Create function: `render_single_dimension_chart(data: StatisticsVisualizationData) -> None`

**Features**:
- Render horizontal bar charts for sentiment/category/topic breakdowns
- Show count and percentage for each item
- Use appropriate colors:
  - Sentiment "for": green
  - Sentiment "against": red
  - Sentiment "mixed": yellow
  - Sentiment "unclear": dim/gray
  - Categories/Topics: default colors
- Scale bars based on percentages (longest bar = 100%)
- Include title showing what's being displayed (e.g., "Breakdown by Sentiment")

**Implementation Notes**:
- Use `rich.console.Console` and `rich.text.Text` for rendering
- Consider using `rich.table.Table` or manual rendering with Text objects
- Add proper spacing and alignment
- Include filter information if filters were applied

**Dependencies**: Tasks 1.2, 1.3

---

### Task 2.2: Integrate Visualization Emission into get_statistics()
**File**: `reggie/agent/tools.py`

Modify the `get_statistics()` function to:
1. After getting results from repository
2. Before formatting the text return value
3. Call `emit_visualization()` with structured data

**Implementation Notes**:
- Import visualization models and emit function
- Create `StatisticsVisualizationData` object with all relevant info
- Ensure this doesn't break existing functionality
- The text return value is still needed for the agent to read

**Dependencies**: Tasks 1.1, 1.3

---

### Task 2.3: Hook Up Visualization Callback in CLI
**File**: `reggie/cli/main.py`

In the `discuss()` command:
1. Import the renderer and callback setter
2. Set up visualization callback alongside status callback
3. Clear it when session ends

**Implementation Notes**:
- Add import for `set_visualization_callback` and renderer
- Set callback after status callback setup (around line 339)
- Clear callback in the finally block (around line 430)
- The callback should simply delegate to the renderer

**Dependencies**: Tasks 2.1, 2.2

---

### Task 2.4: Manual Testing - Agent Auto-Visualization
**Manual Testing Checkpoint**

Test the auto-visualization feature with various agent queries:
- "How many comments support vs oppose?"
- "Break down comments by category"
- "What topics are discussed?"
- "How many physicians support this?" (with filtering)

**Verify**:
- Visualizations appear after tool calls
- Bars scale correctly
- Colors are appropriate
- Doesn't break existing functionality
- Agent can still read the text response

**Dependencies**: Task 2.3

---

## Phase 3: `reggie visualize` Command

### Task 3.1: Add Repository Method for Sentiment-by-Category Cross-Tabulation
**File**: `reggie/db/repository.py`

Create new method in `CommentRepository`:
```python
async def get_sentiment_by_category(
    document_id: str,
    conn
) -> Dict[str, Dict[str, int]]
```

**Returns**: Nested dict structure like:
```python
{
    "Physicians & Surgeons": {"for": 45, "against": 12, "mixed": 3, "unclear": 0},
    "Patient Advocates": {"for": 23, "against": 38, "mixed": 5, "unclear": 2},
    ...
}
```

**Implementation Notes**:
- Use SQL GROUP BY with category and sentiment
- Filter out categories with 0 total comments
- Order by total comment count (descending) for better visualization
- Consider using a proper Pydantic model instead of dict if preferred

**Dependencies**: Task 1.3

---

### Task 3.2: Create Centered Opposition/Support Bar Renderer
**File**: `reggie/agent/renderers/statistics_renderer.py`

Create function: `render_opposition_support_chart(data: OppositionSupportVisualizationData) -> None`

**Features**:
- Centered horizontal bars with opposition (left) and support (right)
- Red bars for "against" (◄━━━━━), green bars for "for" (━━━━━►)
- Show counts and percentages on each side
- Scale bars based on percentage within category
- Each row shows one category
- Proper alignment with category names

**Visual Format**:
```
Opposition/Support Breakdown by Category
Document: {title}
Total comments: {count}

Physicians & Surgeons       ◄━━━━━━━━━ 45 (78%) | 12 (22%) ━━►
Patient Advocates           ◄━━━━ 23 (38%) | 38 (62%) ━━━━━━━━━►
Hospitals & Systems         ◄━━━━━━━━━━━ 67 (93%) | 5 (7%) ►
```

**Implementation Notes**:
- Calculate bar widths based on terminal width
- Use percentage within each category for bar length
- Consider only "for" and "against" for the centered bars
- Show "mixed" and "unclear" separately below (optional) or omit
- Use `rich.console.Console.width` to determine available space
- Add proper padding and alignment with Text objects

**Dependencies**: Tasks 1.3, 3.1

---

### Task 3.3: Add `visualize` CLI Command
**File**: `reggie/cli/main.py`

Create new command:
```python
@cli.command()
@click.argument("document_id")
def visualize(document_id: str):
    """Display opposition/support visualization for a document."""
```

**Implementation**:
1. Verify document exists and has processed comments
2. Call repository method to get sentiment-by-category data
3. Get document metadata (title, total comments)
4. Create visualization data model
5. Call renderer to display

**Implementation Notes**:
- Follow similar pattern to `list()` and `discuss()` commands
- Include document title in visualization
- Handle errors gracefully (document not found, no data, etc.)
- Use async properly with `asyncio.run()`

**Dependencies**: Tasks 3.1, 3.2

---

### Task 3.4: Manual Testing - Visualize Command
**Manual Testing Checkpoint**

Test the new `reggie visualize` command:
- Run with different documents
- Verify bars scale correctly to percentages
- Check alignment and formatting
- Test with edge cases (document with few comments, one-sided sentiment, etc.)

**Verify**:
- Visual clarity and readability
- Correct data representation
- Handles missing/empty categories
- Terminal width adaptability

**Dependencies**: Task 3.3

---

## Phase 4: Polish, Edge Cases & Testing

### Task 4.1: Refine Bar Scaling Algorithm
**File**: `reggie/agent/renderers/statistics_renderer.py`

Optimize the bar width calculation:
- Determine optimal maximum bar width based on terminal size
- Ensure bars never overflow terminal width
- Handle very long category names (truncate if needed)
- Make sure percentages display clearly

**Implementation Notes**:
- Test with various terminal widths
- Consider minimum bar width for very small percentages
- Add constants for bar characters and styling

**Dependencies**: Tasks 2.1, 3.2

---

### Task 4.2: Handle Edge Cases
**Files**: Multiple renderer and CLI files

Add robust handling for:
- No comments in dataset
- All comments in single sentiment category
- Very long category/topic names
- Zero percentages (should still show label)
- Empty filters (no matching comments)

**Implementation Notes**:
- Add helpful messages when no data available
- Graceful degradation
- Clear error messages to users

**Dependencies**: Tasks 2.1, 2.3, 3.2, 3.3

---

### Task 4.3: Add Color Scheme Configuration
**File**: `reggie/agent/renderers/statistics_renderer.py`

Create color constants and styling utilities:
- Define color scheme at top of file
- Support future theming if needed
- Ensure colors work on both light and dark terminals

**Implementation Notes**:
- Use Rich's color tags consistently
- Consider accessibility (colorblind-friendly)
- Add visual separators and styling for clarity

**Dependencies**: Tasks 2.1, 3.2

---

### Task 4.4: Write Unit Tests
**File**: `tests/unit/test_visualizations.py`

Create unit tests for:
- Visualization callback registration/clearing
- Data model validation
- Repository cross-tabulation method
- Bar scaling calculations (percentage to width)

**Implementation Notes**:
- Mock Rich console output for renderer tests
- Test with various data scenarios
- Use pytest fixtures for common test data

**Dependencies**: All previous tasks

---

### Task 4.5: Write Integration Tests
**File**: `tests/integration/test_visualization_integration.py`

Create integration tests for:
- Full flow from tool call to visualization
- CLI visualize command with real database
- Callback system working end-to-end

**Implementation Notes**:
- Use test database with sample data
- Capture console output for verification
- Test both visualization types

**Dependencies**: Task 4.4

---

### Task 4.6: Update Documentation
**Files**:
- `README.md`
- Docstrings in new modules

Update documentation:
- Add `reggie visualize` to command documentation in README
- Document the auto-visualization behavior
- Add docstrings to all new functions and classes
- Include examples and screenshots if possible

**Implementation Notes**:
- Keep documentation concise
- Show example commands
- Explain visualization features

**Dependencies**: All previous tasks

---

## Implementation Order Summary

1. **Phase 1** (Infrastructure): Tasks 1.1 → 1.2 → 1.3
2. **Phase 2** (Auto-viz): Tasks 2.1 → 2.2 → 2.3 → 2.4 (test)
3. **Phase 3** (Visualize command): Tasks 3.1 → 3.2 → 3.3 → 3.4 (test)
4. **Phase 4** (Polish): Tasks 4.1 → 4.2 → 4.3 → 4.4 → 4.5 → 4.6

---

## Notes

- Each task should be completed and tested before moving to the next
- Manual testing checkpoints (2.4, 3.4) are critical - don't skip them
- If any task reveals architectural issues, pause and reassess
- Keep commits small and focused on individual tasks
- Maintain backward compatibility throughout

---

## Estimated Effort

- **Phase 1**: 1-2 hours (foundational setup)
- **Phase 2**: 2-3 hours (first renderer + integration)
- **Phase 3**: 2-3 hours (cross-tab + centered bars)
- **Phase 4**: 2-3 hours (polish + testing)

**Total**: 7-11 hours of focused development time
