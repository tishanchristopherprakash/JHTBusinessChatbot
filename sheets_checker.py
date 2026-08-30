import os
import streamlit as st
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def _build_service():
    api_key = os.environ.get("API_KEY", "")
    return build("sheets", "v4", developerKey=api_key)


@st.cache_resource(ttl=300)
def load_sheet_data(sheet_id: str) -> dict | None:
    try:
        service = _build_service()
        response = (
            service.spreadsheets()
            .get(spreadsheetId=sheet_id, includeGridData=True)
            .execute()
        )
        sheet = response["sheets"][0]
        return {
            "merges": sheet.get("merges", []),
            "grid_data": sheet["data"][0]["rowData"],
        }
    except Exception:
        return None


def _cell_value(row_data: dict, col: int) -> str:
    values = row_data.get("values", [])
    if col >= len(values):
        return ""
    return values[col].get("formattedValue", "") or ""


def _is_white(bg: dict) -> bool:
    if not bg:
        return True
    return (
        bg.get("red", 1.0) >= 0.99
        and bg.get("green", 1.0) >= 0.99
        and bg.get("blue", 1.0) >= 0.99
    )


def _parse_date_from_label(label: str) -> datetime | None:
    """Parse labels like 'Monday, 1/6' or '2/6' or '1/6/2026'."""
    s = label.strip()
    # Strip day-of-week prefix, e.g. "Monday, "
    if ", " in s:
        s = s.split(", ", 1)[1]
    for fmt in ("%d/%m/%Y", "%d/%m", "%d-%m-%Y", "%d-%m"):
        try:
            parsed = datetime.strptime(s, fmt)
            if parsed.year == 1900:
                parsed = parsed.replace(year=datetime.now().year)
            return parsed
        except ValueError:
            continue
    return None


def find_date_rows(grid_data: list) -> list[tuple[str, int]]:
    """
    Scan col 0 of every row for date labels.
    Returns [(date_label, row_index), ...] sorted by row index.
    """
    results = []
    for i, row in enumerate(grid_data):
        val = _cell_value(row, 0)
        if val and _parse_date_from_label(val) is not None:
            results.append((val, i))
    return results


def _normalise_time(t: str) -> str:
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I%p"):
        try:
            return datetime.strptime(t.strip(), fmt).strftime("%-H:%M")
        except ValueError:
            continue
    return t.strip()


def _is_purple(bg: dict) -> bool:
    if not bg:
        return False
    r = bg.get("red", 0.0)
    g = bg.get("green", 0.0)
    b = bg.get("blue", 0.0)
    # Purple: notable red and blue, low green, not white
    return r > 0.3 and b > 0.3 and g < 0.5 and not (r > 0.95 and g > 0.95 and b > 0.95)


def _is_slot_taken_in_rows(grid_data: list, col: int, booking_rows: range) -> bool:
    for row_idx in booking_rows:
        if row_idx >= len(grid_data):
            break
        cells = grid_data[row_idx].get("values", [])
        if col >= len(cells):
            continue
        bg = cells[col].get("effectiveFormat", {}).get("backgroundColor", {})
        if _is_purple(bg):
            print(f"[DEBUG] Row {row_idx} col {col} is purple (R={bg.get('red',0):.2f} G={bg.get('green',0):.2f} B={bg.get('blue',0):.2f}) — skipping", flush=True)
            continue  # purple rows are not customer bookings — ignore
        if not _is_white(bg):
            return True
    return False


def _resolve_date_block(grid_data: list, date_str: str):
    """
    Returns (time_slots, booking_rows, error_str) for a given ISO date string.
    On failure, time_slots and booking_rows are None and error_str is set.
    """
    date_rows = find_date_rows(grid_data)
    if not date_rows:
        return None, None, "error:no_dates_found"
    try:
        requested_dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError:
        return None, None, "error:invalid_date"

    target_row_idx: int | None = None
    for label, row_idx in date_rows:
        label_dt = _parse_date_from_label(label)
        if label_dt and label_dt.month == requested_dt.month and label_dt.day == requested_dt.day:
            target_row_idx = row_idx
            break
    if target_row_idx is None:
        return None, None, "error:date_not_found"

    next_date_row = len(grid_data)
    for _, row_idx in date_rows:
        if row_idx > target_row_idx:
            next_date_row = row_idx
            break

    booking_rows = range(target_row_idx + 2, next_date_row)
    time_row_idx = target_row_idx + 1
    if time_row_idx >= len(grid_data):
        return None, None, "error:no_time_row"

    time_slots: dict[int, str] = {}
    for i, cell in enumerate(grid_data[time_row_idx].get("values", [])):
        val = cell.get("formattedValue", "") or ""
        if val.strip():
            time_slots[i] = val.strip()

    return time_slots, booking_rows, None


def find_available_slots(date_str: str, duration_hours: float) -> list[str] | str:
    """
    Return a list of available start times (as friendly strings like '9:00am')
    for the given date and duration, or an error string on failure.
    """
    sheet_id = os.environ.get("SHEET_ID", "")
    if not sheet_id:
        return "error:no_sheet_id"
    sheet_data = load_sheet_data(sheet_id)
    if sheet_data is None:
        return "error:load_failed"

    grid_data = sheet_data["grid_data"]
    time_slots, booking_rows, err = _resolve_date_block(grid_data, date_str)
    if err:
        return err

    num_slots = max(1, int(duration_hours * 2))
    sorted_cols = sorted(time_slots.keys())
    available: list[str] = []

    for i, start_col in enumerate(sorted_cols):
        required_cols = sorted_cols[i: i + num_slots]
        if len(required_cols) < num_slots:
            break
        if not any(_is_slot_taken_in_rows(grid_data, col, booking_rows) for col in required_cols):
            raw = time_slots[start_col]
            try:
                friendly = datetime.strptime(raw, "%H:%M").strftime("%-I:%M%p").lower()
            except ValueError:
                friendly = raw
            available.append(friendly)

    return available


def check_availability(date_str: str, start_time_str: str, duration_hours: float) -> str:
    sheet_id = os.environ.get("SHEET_ID", "")
    if not sheet_id:
        return "error:no_sheet_id"
    sheet_data = load_sheet_data(sheet_id)
    if sheet_data is None:
        return "error:load_failed"

    grid_data = sheet_data["grid_data"]
    time_slots, booking_rows, err = _resolve_date_block(grid_data, date_str)
    if err:
        return err

    target_norm = _normalise_time(start_time_str)
    start_col: int | None = None
    for col, label in time_slots.items():
        if _normalise_time(label) == target_norm:
            start_col = col
            break
    if start_col is None:
        return "error:time_not_found"

    sorted_cols = sorted(time_slots.keys())
    try:
        start_idx = sorted_cols.index(start_col)
    except ValueError:
        return "error:time_not_found"

    num_slots = max(1, int(duration_hours * 2))
    required_cols = sorted_cols[start_idx: start_idx + num_slots]
    if len(required_cols) < num_slots:
        return "error:duration_exceeds_schedule"

    for col in required_cols:
        if _is_slot_taken_in_rows(grid_data, col, booking_rows):
            return "taken"

    return "available"
