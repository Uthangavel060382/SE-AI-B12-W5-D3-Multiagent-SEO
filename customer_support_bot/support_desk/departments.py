"""The help-desk roster: the Manager plus the four department agents it can assign work to."""

from dataclasses import dataclass

MANAGER_NAME = "Manager"
MANAGER_SUMMARY = "Triages every request and assigns it"


@dataclass(frozen=True)
class Department:
    key: str  # value used in RoutingDecision.departments
    agent_name: str  # AutoGen agent name (must be a valid Python identifier)
    title: str
    emoji: str
    summary: str  # short blurb for the UI
    scope: str  # what the department handles; used in prompts and agent descriptions
    ticket_prefix: str


DEPARTMENTS: dict[str, Department] = {
    d.key: d
    for d in (
        Department(
            "IT",
            "IT_Support",
            "IT Support",
            "💻",
            "Passwords, VPN, laptops, software",
            "passwords and account lockouts, VPN, Wi-Fi and network problems, email, laptops and other hardware, "
            "software installs and licences, system outages, and security incidents such as phishing",
            "IT",
        ),
        Department(
            "HR",
            "HR_Support",
            "HR",
            "🤝",
            "Leave, holidays, policies, benefits",
            "leave balances and leave applications, the holiday calendar, HR policies such as work from home, "
            "benefits and insurance, onboarding, and the employee profile",
            "HR",
        ),
        Department(
            "Admin",
            "Admin_Support",
            "Admin & Facilities",
            "🏢",
            "Meeting rooms, ID cards, supplies",
            "meeting room availability and bookings, ID and access cards, office supplies, seating, parking, "
            "and other facilities requests",
            "ADM",
        ),
        Department(
            "Accounts",
            "Accounts_Support",
            "Accounts & Finance",
            "💰",
            "Payslips, expense claims, tax",
            "payslips and salary breakdowns, expense claims and reimbursements, tax deductions, "
            "and other finance questions",
            "FIN",
        ),
    )
}

AGENT_TITLE = {MANAGER_NAME: "Manager", **{d.agent_name: d.title for d in DEPARTMENTS.values()}}
AGENT_EMOJI = {MANAGER_NAME: "👔", **{d.agent_name: d.emoji for d in DEPARTMENTS.values()}}
