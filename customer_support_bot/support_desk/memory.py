"""Long-term context for the agents, stored in AutoGen ListMemory and injected before every reply."""

from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType

from .departments import MANAGER_NAME

# Standing policies each agent should always keep in mind (detailed guides live in the knowledge base).
POLICIES: dict[str, list[str]] = {
    MANAGER_NAME: [
        "Response targets: urgent within 1 hour, high within 4 working hours, medium within 1 working day, "
        "low within 3 working days.",
        "Security incidents (phishing clicks, lost or stolen laptops, suspected malware) go to IT before anything else.",
        "Salary bank-account changes belong to Accounts; personal-detail changes belong to HR.",
    ],
    "IT": [
        "Never ask for or reveal passwords or MFA codes. Resets only happen through the reset_password tool, "
        "which sends a self-service link.",
        "Check system status before troubleshooting VPN, email, Wi-Fi, SAP, Jira or SharePoint problems. "
        "If there is a known incident, share it and its workaround instead of troubleshooting.",
        "Phishing clicks and lost or stolen laptops are 'urgent' tickets.",
        "Hardware swaps happen at the IT counter, 2nd floor, 10:00-18:00 on working days.",
    ],
    "HR": [
        "Annual leave should be applied at least 3 working days ahead; sick leave can be applied for past dates.",
        "The leave year runs January-December, and up to 5 unused annual-leave days carry forward.",
        "All leave is approved by the reporting manager. HR can't approve leave in chat.",
        "For harassment or grievance reports, respond with empathy, raise a 'high' priority ticket, "
        "and mention the confidential ethics line ethics@acme.example.",
    ],
    "Admin": [
        "Rooms can be booked 08:00-20:00 on working days, for up to 3 hours per booking.",
        "Always check availability with find_available_rooms before booking, and pick the smallest room that fits.",
        "Replacing a lost ID card costs INR 500 (deducted from salary); damaged cards are replaced free.",
    ],
    "Accounts": [
        "Never reveal another employee's salary. Payslip figures come only from the get_payslip tool.",
        "Expense claims must be filed within 30 days with receipts. Per-claim limits (INR): travel 25,000, "
        "meals 1,500, internet 1,000, training 20,000, other 5,000.",
        "Claims approved by the 25th are paid with that month's salary; later approvals are paid the next month.",
        "Salary is paid on the last working day of the month.",
    ],
}


def build_memory(employee: dict, owner: str) -> ListMemory:
    """The signed-in employee's profile plus the owner's standing policies.

    `owner` is MANAGER_NAME or a department key such as "IT".
    """
    profile = (
        f"Signed-in employee: {employee['name']} ({employee['id']}), {employee['designation']} in "
        f"{employee['department']}, based in the {employee['location']} office. Reports to {employee['manager']}. "
        f"Work email {employee['email']}. Joined on {employee['joined']}."
    )
    contents = [MemoryContent(content=profile, mime_type=MemoryMimeType.TEXT, metadata={"kind": "employee_profile"})]
    contents += [
        MemoryContent(content=policy, mime_type=MemoryMimeType.TEXT, metadata={"kind": "policy"})
        for policy in POLICIES[owner]
    ]
    return ListMemory(name=f"{owner.lower()}_memory", memory_contents=contents)
