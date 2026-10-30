"""Mock company systems behind the department tools.

Reference data (employees, rooms, holidays, ...) comes from data/company_data.json and never changes.
Everything the agents create (tickets, leave requests, room bookings, expense claims) is appended to
data/runtime_store.json, so it survives restarts and appears on the "My Tickets & Requests" page.
"""

import json
import os
import re
import threading
from calendar import monthrange
from copy import deepcopy
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REFERENCE_FILE = DATA_DIR / "company_data.json"
KNOWLEDGE_BASE_FILE = DATA_DIR / "knowledge_base.json"
STORE_FILE = DATA_DIR / "runtime_store.json"

_EMPTY_STORE: dict[str, Any] = {
    "counters": {},
    "tickets": [],
    "leave_requests": [],
    "room_bookings": [],
    "expense_claims": [],
}
_store_lock = threading.RLock()  # sync tools run in worker threads, sometimes in parallel


@lru_cache(maxsize=1)
def reference() -> dict[str, Any]:
    return json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def knowledge_base() -> list[dict[str, Any]]:
    return json.loads(KNOWLEDGE_BASE_FILE.read_text(encoding="utf-8"))


def list_employees() -> list[dict[str, Any]]:
    return reference()["employees"]


def get_employee(employee_id: str) -> dict[str, Any]:
    for employee in list_employees():
        if employee["id"] == employee_id:
            return employee
    raise KeyError(f"Unknown employee ID: {employee_id}")


# ---- Runtime store --------------------------------------------------------------------------------


def _read_store() -> dict[str, Any]:
    if not STORE_FILE.exists():
        return deepcopy(_EMPTY_STORE)
    data = json.loads(STORE_FILE.read_text(encoding="utf-8"))
    for key, default in _EMPTY_STORE.items():
        data.setdefault(key, deepcopy(default))
    return data


def _write_store(data: dict[str, Any]) -> None:
    tmp = STORE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STORE_FILE)


def insert(collection: str, prefix: str, record: dict[str, Any]) -> dict[str, Any]:
    """Append a record with the next sequential ID for `prefix` (e.g. IT-1001) and return it."""
    with _store_lock:
        data = _read_store()
        number = data["counters"].get(prefix, 1000) + 1
        data["counters"][prefix] = number
        saved = {"id": f"{prefix}-{number}", "created_at": datetime.now().isoformat(timespec="seconds"), **record}
        data[collection].append(saved)
        _write_store(data)
    return saved


def records(collection: str, **filters: Any) -> list[dict[str, Any]]:
    with _store_lock:
        rows = _read_store()[collection]
    return [row for row in rows if all(row.get(key) == value for key, value in filters.items())]


# ---- Calendar ---------------------------------------------------------------------------------------


def holidays() -> dict[date, str]:
    return {date.fromisoformat(h["date"]): h["name"] for h in reference()["holidays"]}


def is_working_day(day: date) -> bool:
    return day.weekday() < 5 and day not in holidays()


def working_days(start: date, end: date) -> int:
    return sum(is_working_day(start + timedelta(days=offset)) for offset in range((end - start).days + 1))


def last_working_day(year: int, month: int) -> date:
    day = date(year, month, monthrange(year, month)[1])
    while not is_working_day(day):
        day -= timedelta(days=1)
    return day


# ---- HR -------------------------------------------------------------------------------------------


def leave_balance(employee: dict[str, Any]) -> dict[str, Any]:
    used = dict(employee["leave_taken"])
    for request in records("leave_requests", employee_id=employee["id"]):
        used[request["leave_type"]] = used.get(request["leave_type"], 0) + request["working_days"]
    entitlement = employee["leave_entitlement"]
    return {
        "leave_year": f"January-December {date.today().year}",
        "entitlement": entitlement,
        "used_or_pending": used,
        "balance": {kind: entitlement[kind] - used.get(kind, 0) for kind in entitlement},
    }


# ---- Accounts -------------------------------------------------------------------------------------


def payslip(employee: dict[str, Any], month: str) -> dict[str, Any]:
    try:
        year, month_number = (int(part) for part in month.strip().split("-"))
        first_day = date(year, month_number, 1)
    except ValueError:
        return {"error": "Use the YYYY-MM format, for example 2026-09."}
    if first_day < date.fromisoformat(employee["joined"]).replace(day=1):
        return {"error": f"{employee['name']} joined on {employee['joined']}, so there is no payslip for {month}."}
    pay_date = last_working_day(year, month_number)
    today = date.today()
    if pay_date > today:
        return {"error": f"The {first_day:%B %Y} payslip will be published on the pay date, {pay_date.isoformat()}."}
    if (today.year - year) * 12 + today.month - month_number > 12:
        return {"error": "Only the last 12 payslips are available here; older ones are in SAP SuccessFactors."}
    salary = employee["salary"]
    earnings = {key: salary[key] for key in ("basic", "hra", "special_allowance")}
    deductions = {key: salary[key] for key in ("pf", "professional_tax", "income_tax")}
    gross, total_deductions = sum(earnings.values()), sum(deductions.values())
    return {
        "month": f"{first_day:%B %Y}",
        "pay_date": pay_date.isoformat(),
        "currency": "INR",
        "earnings": earnings,
        "gross_pay": gross,
        "deductions": deductions,
        "total_deductions": total_deductions,
        "net_pay": gross - total_deductions,
        "credited_to": f"bank account ending {employee['bank_last4']}",
    }


def expense_claims(employee_id: str) -> list[dict[str, Any]]:
    seeded = [claim for claim in reference()["expense_claims"] if claim["employee_id"] == employee_id]
    return seeded + records("expense_claims", employee_id=employee_id)


# ---- Admin ----------------------------------------------------------------------------------------


def _minutes(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


def find_room(room_id: str) -> dict[str, Any] | None:
    return next((room for room in reference()["meeting_rooms"] if room["id"] == room_id), None)


def room_bookings_on(room_id: str, day: date) -> list[dict[str, str]]:
    """Recurring meetings plus confirmed bookings for one room on one day."""
    busy = [
        {"start": r["start"], "end": r["end"], "title": r["title"]}
        for r in reference()["recurring_bookings"]
        if r["room_id"] == room_id and day.weekday() in r["weekdays"]
    ]
    busy += [
        {"start": b["start_time"], "end": b["end_time"], "title": b["purpose"]}
        for b in records("room_bookings", room_id=room_id, date=day.isoformat())
    ]
    return busy


def is_room_free(room_id: str, day: date, start: str, end: str) -> bool:
    start_min, end_min = _minutes(start), _minutes(end)
    return all(
        end_min <= _minutes(b["start"]) or start_min >= _minutes(b["end"]) for b in room_bookings_on(room_id, day)
    )


def available_rooms(location: str, day: date, start: str, end: str, attendees: int) -> list[dict[str, Any]]:
    rooms = [r for r in reference()["meeting_rooms"] if r["location"] == location and r["capacity"] >= attendees]
    return sorted((r for r in rooms if is_room_free(r["id"], day, start, end)), key=lambda r: r["capacity"])


# ---- Knowledge base ---------------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "can", "do", "for", "from", "how", "i", "in", "is", "it", "me", "my",
    "of", "on", "or", "the", "to", "what", "when", "with", "you", "your",
}  # fmt: skip


def _terms(text: str) -> set[str]:
    words = (word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS)
    return {word[:-1] if len(word) > 4 and word.endswith("s") else word for word in words}


def search_knowledge_base(query: str, department: str, limit: int = 3) -> list[dict[str, Any]]:
    """Keyword search over the department's articles plus general ones (title > tags > body)."""
    terms = _terms(query)
    scored = []
    for article in knowledge_base():
        if article["department"] not in (department, "General"):
            continue
        score = (
            3 * len(terms & _terms(article["title"]))
            + 2 * len(terms & _terms(" ".join(article["tags"])))
            + len(terms & _terms(article["content"]))
        )
        if score:
            scored.append((score, article))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{"id": a["id"], "title": a["title"], "content": a["content"]} for _, a in scored[:limit]]
