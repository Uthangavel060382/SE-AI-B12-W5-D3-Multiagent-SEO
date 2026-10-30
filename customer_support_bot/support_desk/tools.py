"""Function tools for the four department agents.

Every tool is created for one signed-in employee: the employee record is captured in a closure, so the model
never passes an employee ID and can't read or change anyone else's data.
"""

import json
from datetime import date, datetime
from typing import Annotated, Any, Callable, Literal

from autogen_core.tools import FunctionTool

from . import data_store as db
from .departments import DEPARTMENTS

Priority = Literal["low", "medium", "high", "urgent"]
RESPONSE_TARGETS = {
    "urgent": "within 1 hour",
    "high": "within 4 working hours",
    "medium": "within 1 working day",
    "low": "within 3 working days",
}


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def _error(message: str) -> str:
    return _json({"error": message})


def _parse_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def _ticket(employee: dict, department: str, summary: str, details: str, priority: str, status: str) -> dict:
    return db.insert(
        "tickets",
        DEPARTMENTS[department].ticket_prefix,
        {
            "employee_id": employee["id"],
            "department": department,
            "summary": summary,
            "details": details,
            "priority": priority,
            "status": status,
        },
    )


# ---- Shared by every department ---------------------------------------------------------------------


def _shared_tools(employee: dict, department: str) -> list[FunctionTool]:
    title = DEPARTMENTS[department].title

    def search_knowledge_base(query: Annotated[str, "What to look up, e.g. 'VPN keeps disconnecting'"]) -> str:
        articles = db.search_knowledge_base(query, department)
        if not articles:
            return _json({"results": [], "note": "No matching article. Use your memory, or raise a ticket."})
        return _json({"results": articles})

    def create_ticket(
        summary: Annotated[str, "One-line summary of the issue"],
        details: Annotated[str, "Full description, including what the employee has already tried"],
        priority: Annotated[Priority, "urgent = security incident or nobody can work"],
    ) -> str:
        ticket = _ticket(employee, department, summary, details, priority, "Open")
        return _json({"ticket_id": ticket["id"], "status": "Open", "response_target": RESPONSE_TARGETS[priority]})

    def list_my_tickets() -> str:
        fields = ("id", "department", "summary", "priority", "status", "created_at")
        tickets = db.records("tickets", employee_id=employee["id"])
        return _json({"tickets": [{key: t[key] for key in fields} for t in tickets]})

    return [
        FunctionTool(search_knowledge_base, description=f"Search the {title} knowledge base for guides and policies."),
        FunctionTool(create_ticket, description=f"Raise a {title} ticket for an issue that can't be resolved in chat."),
        FunctionTool(list_my_tickets, description="List every help-desk ticket the employee has raised, with status."),
    ]


# ---- IT ---------------------------------------------------------------------------------------------


def _it_tools(employee: dict) -> list[FunctionTool]:
    def check_system_status(
        system: Annotated[
            Literal["all", "email", "vpn", "wifi", "sap", "jira", "sharepoint"], "System to check, or 'all'"
        ],
    ) -> str:
        systems = db.reference()["systems"]
        return _json({"systems": [s for s in systems if system == "all" or s["key"] == system]})

    def reset_password(
        account: Annotated[Literal["windows", "email", "vpn", "sap", "jira"], "Account whose password to reset"],
    ) -> str:
        ticket = _ticket(
            employee, "IT", f"Password reset - {account}", "Self-service reset link issued from chat.", "medium",
            "Resolved - reset link sent",
        )  # fmt: skip
        return _json(
            {
                "result": "Self-service reset link sent. The employee sets a new password; no password is shared in chat.",
                "sent_to": f"mobile ending {employee['phone_last4']} and {employee['personal_email_masked']}",
                "link_valid_minutes": 30,
                "ticket_id": ticket["id"],
            }
        )

    def request_software(
        software_name: Annotated[str, "Name of the software"],
        business_justification: Annotated[str, "Why it is needed for work"],
    ) -> str:
        wanted = software_name.strip().lower()
        if not wanted:
            return _error("Tell me which software is needed.")
        catalog = db.reference()["software_catalog"]

        def find(names: list[str]) -> str | None:
            return next((name for name in names if wanted in name.lower() or name.lower() in wanted), None)

        if name := find(catalog["pre_approved"]):
            status = "Approved - installs automatically from Company Portal within 4 hours"
        elif name := find(catalog["needs_approval"]):
            status = f"Pending licence approval from {employee['manager']}"
        else:
            name, status = software_name.strip(), "Pending security review (not in the software catalogue)"
        ticket = _ticket(employee, "IT", f"Software request - {name}", business_justification, "low", status)
        return _json({"ticket_id": ticket["id"], "software": name, "status": status})

    def list_my_assets() -> str:
        return _json({"assets": employee["assets"]})

    return [
        FunctionTool(check_system_status, description="Check live status and known incidents for company IT systems."),
        FunctionTool(reset_password, description="Send the employee a self-service password reset link for one account."),
        FunctionTool(request_software, description="Request a software install; checks the approved catalogue."),
        FunctionTool(list_my_assets, description="List the laptop and other IT assets assigned to the employee."),
    ]


# ---- HR ---------------------------------------------------------------------------------------------


def _hr_tools(employee: dict) -> list[FunctionTool]:
    def get_leave_balance() -> str:
        return _json(db.leave_balance(employee))

    def apply_leave(
        leave_type: Annotated[Literal["annual", "sick", "casual"], "Type of leave"],
        start_date: Annotated[str, "First day of leave, YYYY-MM-DD"],
        end_date: Annotated[str, "Last day of leave, YYYY-MM-DD"],
        reason: Annotated[str, "Short reason for the leave"],
    ) -> str:
        try:
            start, end = _parse_date(start_date), _parse_date(end_date)
        except ValueError:
            return _error("Dates must use the YYYY-MM-DD format.")
        if end < start:
            return _error("end_date is before start_date.")
        if start < date.today() and leave_type != "sick":
            return _error("Only sick leave can be applied for past dates.")
        days = db.working_days(start, end)
        if days == 0:
            return _error("Those dates are weekends or company holidays, so no leave is needed.")
        available = db.leave_balance(employee)["balance"][leave_type]
        if days > available:
            return _error(f"Not enough {leave_type} leave: {days} working day(s) requested, {available} available.")
        for existing in db.records("leave_requests", employee_id=employee["id"]):
            if _parse_date(existing["start_date"]) <= end and _parse_date(existing["end_date"]) >= start:
                return _error(f"These dates overlap leave request {existing['id']}.")
        request = db.insert(
            "leave_requests",
            "LV",
            {
                "employee_id": employee["id"],
                "leave_type": leave_type,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "working_days": days,
                "reason": reason,
                "status": f"Pending approval from {employee['manager']}",
            },
        )
        return _json(
            {
                "request_id": request["id"],
                "working_days": days,
                "status": request["status"],
                "balance_after_approval": available - days,
            }
        )

    def get_holiday_calendar() -> str:
        upcoming = [
            {"date": day.isoformat(), "weekday": f"{day:%A}", "name": name}
            for day, name in sorted(db.holidays().items())
            if day >= date.today()
        ]
        return _json({"upcoming_holidays": upcoming[:8]})

    def get_my_profile() -> str:
        fields = ("id", "name", "email", "department", "designation", "manager", "location", "joined")
        return _json({key: employee[key] for key in fields})

    return [
        FunctionTool(get_leave_balance, description="Get the employee's leave entitlement, usage and balance."),
        FunctionTool(apply_leave, description="Apply for leave; validates dates and balance, then sends it for approval."),
        FunctionTool(get_holiday_calendar, description="List the upcoming company holidays."),
        FunctionTool(get_my_profile, description="Get the employee's HR profile: role, manager, location, joining date."),
    ]


# ---- Admin ------------------------------------------------------------------------------------------


def _check_slot(booking_date: str, start_time: str, end_time: str) -> tuple[date, str, str] | str:
    """Validate a room-booking slot. Returns (day, start, end), or a JSON error string."""
    try:
        day = _parse_date(booking_date)
        start = datetime.strptime(start_time.strip(), "%H:%M")
        end = datetime.strptime(end_time.strip(), "%H:%M")
    except ValueError:
        return _error("Use YYYY-MM-DD for the date and 24-hour HH:MM for times.")
    now = datetime.now()
    if day < now.date() or (day == now.date() and start.time() <= now.time()):
        return _error("That slot is in the past.")
    if not db.is_working_day(day):
        return _error(f"{day:%A %d %B %Y} is not a working day; the office is closed.")
    if end <= start:
        return _error("end_time must be after start_time.")
    if start.hour < 8 or end > datetime.strptime("20:00", "%H:%M"):
        return _error("Rooms can only be booked between 08:00 and 20:00.")
    if (end - start).total_seconds() > 3 * 3600:
        return _error("Bookings are limited to 3 hours.")
    return day, f"{start:%H:%M}", f"{end:%H:%M}"


def _admin_tools(employee: dict) -> list[FunctionTool]:
    def find_available_rooms(
        booking_date: Annotated[str, "YYYY-MM-DD"],
        start_time: Annotated[str, "24-hour HH:MM, e.g. 15:00"],
        end_time: Annotated[str, "24-hour HH:MM"],
        attendees: Annotated[int, "Number of people"],
    ) -> str:
        slot = _check_slot(booking_date, start_time, end_time)
        if isinstance(slot, str):
            return slot
        rooms = db.available_rooms(employee["location"], *slot, attendees)
        return _json(
            {
                "office": employee["location"],
                "date": slot[0].isoformat(),
                "time": f"{slot[1]}-{slot[2]}",
                "available_rooms": rooms,
                "note": "" if rooms else "No room fits that slot. Suggest another time.",
            }
        )

    def book_meeting_room(
        room_id: Annotated[str, "Room ID from find_available_rooms, e.g. RM-102"],
        booking_date: Annotated[str, "YYYY-MM-DD"],
        start_time: Annotated[str, "24-hour HH:MM"],
        end_time: Annotated[str, "24-hour HH:MM"],
        purpose: Annotated[str, "Meeting title"],
    ) -> str:
        slot = _check_slot(booking_date, start_time, end_time)
        if isinstance(slot, str):
            return slot
        room = db.find_room(room_id.strip().upper())
        if room is None:
            return _error(f"Unknown room ID {room_id}.")
        day, start, end = slot
        if not db.is_room_free(room["id"], day, start, end):
            return _error(f"{room['name']} is already booked then. Use find_available_rooms to pick another room.")
        booking = db.insert(
            "room_bookings",
            "BK",
            {
                "employee_id": employee["id"],
                "room_id": room["id"],
                "room_name": room["name"],
                "date": day.isoformat(),
                "start_time": start,
                "end_time": end,
                "purpose": purpose,
                "status": "Confirmed",
            },
        )
        return _json(
            {
                "booking_id": booking["id"],
                "room": f"{room['name']} ({room['id']}), floor {room['floor']}, {room['location']}",
                "date": day.isoformat(),
                "time": f"{start}-{end}",
                "status": f"Confirmed - calendar invite sent to {employee['email']}",
            }
        )

    def request_access_card(
        reason: Annotated[Literal["lost", "damaged", "new_joiner", "access_change"], "Why a card is needed"],
        details: Annotated[str, "Extra details, e.g. where it was lost or which floors need access"],
    ) -> str:
        lost = reason == "lost"
        ticket = _ticket(
            employee, "Admin", f"Access card - {reason.replace('_', ' ')}", details, "high" if lost else "medium",
            "In progress - ready for collection in 24 hours",
        )  # fmt: skip
        next_steps = ["The lost card has been deactivated."] if lost else []
        next_steps += [
            "Collect the new card from the Admin desk (ground floor) after 24 hours, with photo ID.",
            "Use a temporary visitor badge from reception until then.",
        ]
        return _json({"ticket_id": ticket["id"], "replacement_fee_inr": 500 if lost else 0, "next_steps": next_steps})

    def request_office_supplies(
        item: Annotated[str, "Item, e.g. 'A4 notebook'"],
        quantity: Annotated[int, "How many (1-20)"],
    ) -> str:
        if not 1 <= quantity <= 20:
            return _error("Quantity must be between 1 and 20.")
        ticket = _ticket(
            employee, "Admin", f"Supplies - {quantity} x {item}",
            f"Deliver to {employee['name']}'s desk, {employee['location']} office.", "low",
            "Approved - delivered to your desk within 2 working days",
        )  # fmt: skip
        return _json({"ticket_id": ticket["id"], "status": ticket["status"]})

    return [
        FunctionTool(find_available_rooms, description="Find free meeting rooms in the employee's office for a slot."),
        FunctionTool(book_meeting_room, description="Book a meeting room for a slot after checking availability."),
        FunctionTool(request_access_card, description="Request a new or replacement ID/access card."),
        FunctionTool(request_office_supplies, description="Order stationery or other office supplies to the desk."),
    ]


# ---- Accounts ---------------------------------------------------------------------------------------


def _accounts_tools(employee: dict) -> list[FunctionTool]:
    limits = db.reference()["expense_limits_inr"]

    def get_payslip(month: Annotated[str, "Pay month as YYYY-MM, e.g. 2026-09"]) -> str:
        return _json(db.payslip(employee, month))

    def list_expense_claims() -> str:
        return _json({"claims": db.expense_claims(employee["id"])})

    def submit_expense_claim(
        category: Annotated[Literal["travel", "meals", "internet", "training", "other"], "Expense category"],
        amount_inr: Annotated[float, "Amount in INR"],
        expense_date: Annotated[str, "Date of the expense, YYYY-MM-DD"],
        description: Annotated[str, "What the expense was for"],
    ) -> str:
        try:
            spent_on = _parse_date(expense_date)
        except ValueError:
            return _error("expense_date must use the YYYY-MM-DD format.")
        age = (date.today() - spent_on).days
        if age < 0:
            return _error("The expense date is in the future.")
        if age > 30:
            return _error(f"Claims must be filed within 30 days; this expense is {age} days old. Raise a ticket.")
        if amount_inr <= 0:
            return _error("The amount must be positive.")
        if amount_inr > limits[category]:
            return _error(
                f"The {category} limit is INR {limits[category]:,} per claim. "
                f"Split the claim or get pre-approval from {employee['manager']}."
            )
        claim = db.insert(
            "expense_claims",
            "EXP",
            {
                "employee_id": employee["id"],
                "category": category,
                "amount_inr": round(amount_inr, 2),
                "date": spent_on.isoformat(),
                "description": description,
                "status": f"Submitted - pending approval from {employee['manager']}",
            },
        )
        return _json(
            {
                "claim_id": claim["id"],
                "status": claim["status"],
                "next_step": "Upload the receipt in the Expenses portal within 3 days.",
                "payout": "Claims approved by the 25th are paid with that month's salary.",
            }
        )

    return [
        FunctionTool(get_payslip, description="Get the employee's payslip breakdown for one month."),
        FunctionTool(list_expense_claims, description="List the employee's expense claims and their status."),
        FunctionTool(submit_expense_claim, description="Submit an expense claim; checks the 30-day rule and limits."),
    ]


_DEPARTMENT_TOOLS: dict[str, Callable[[dict], list[FunctionTool]]] = {
    "IT": _it_tools,
    "HR": _hr_tools,
    "Admin": _admin_tools,
    "Accounts": _accounts_tools,
}


def build_tools(department: str, employee: dict) -> list[FunctionTool]:
    """All tools for one department agent, bound to the signed-in employee."""
    return _shared_tools(employee, department) + _DEPARTMENT_TOOLS[department](employee)
