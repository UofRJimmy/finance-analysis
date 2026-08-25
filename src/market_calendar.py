from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


US_MARKET_TIMEZONE = "America/New_York"


def is_us_market_session(day: date | None = None) -> bool:
    """Return whether NYSE/Nasdaq have a regular session on this date."""
    day = day or datetime.now(ZoneInfo(US_MARKET_TIMEZONE)).date()
    try:
        import pandas_market_calendars as mcal

        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=day.isoformat(), end_date=day.isoformat())
        return not schedule.empty
    except Exception:
        return _fallback_market_session(day)


def market_closed_reason(day: date | None = None) -> str:
    day = day or datetime.now(ZoneInfo(US_MARKET_TIMEZONE)).date()
    if is_us_market_session(day):
        return ""
    if day.weekday() >= 5:
        return "周末休市"
    holiday = _market_holidays(day.year).get(day)
    return f"{holiday}休市" if holiday else "美股休市"


def _fallback_market_session(day: date) -> bool:
    if day.weekday() >= 5:
        return False
    return day not in _market_holidays(day.year)


def _market_holidays(year: int) -> dict[date, str]:
    holidays = {
        _observed(date(year, 1, 1)): "New Year's Day",
        _nth_weekday(year, 1, 0, 3): "Martin Luther King Jr. Day",
        _nth_weekday(year, 2, 0, 3): "Presidents' Day",
        _good_friday(year): "Good Friday",
        _last_weekday(year, 5, 0): "Memorial Day",
        _observed(date(year, 6, 19)): "Juneteenth",
        _observed(date(year, 7, 4)): "Independence Day",
        _nth_weekday(year, 9, 0, 1): "Labor Day",
        _nth_weekday(year, 11, 3, 4): "Thanksgiving Day",
        _observed(date(year, 12, 25)): "Christmas Day",
    }
    previous_new_year = _observed(date(year + 1, 1, 1))
    if previous_new_year.year == year:
        holidays[previous_new_year] = "New Year's Day"
    return holidays


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    day = date(year, month, 1)
    offset = (weekday - day.weekday()) % 7
    return day + timedelta(days=offset + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    day = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    offset = (day.weekday() - weekday) % 7
    return day - timedelta(days=offset)


def _good_friday(year: int) -> date:
    return _easter_sunday(year) - timedelta(days=2)


def _easter_sunday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)
