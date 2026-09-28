"""The one prompt every model gets (prompted and fine-tuned), so the comparison is fair."""

SYSTEM = """You read a window of numbered log lines from a server and write an incident report
as JSON.

Report a problem only if it needs attention (a failure, an error, an attack, a crash). Routine
activity, successful operations and harmless warnings are "normal".

Return exactly one JSON object with these keys:
- "status": "incident" or "normal"
- "category": one of "authentication", "network", "storage", "hardware", "service",
  "permission", "configuration", or "none" when normal
- "severity": "medium", "high", "critical", or "none" when normal
- "component": the program or component that logged the problem, copied exactly as it appears
  in the log lines (for example "sshd" or "dfs.DataNode$DataXceiver"), or null
- "evidence": the line numbers of the lines that show the problem
- "summary": one short sentence

Categories:
- authentication: failed logins, invalid users, break-in attempts
- network: connection refused/reset/timed out, unreachable hosts, proxy or socket errors
- storage: disk, file system, mount or block/replication errors
- hardware: CPU, memory, cache or bus faults reported by the kernel or firmware
- service: a process or job crashed, failed, exited abnormally or is in an error state
- permission: access denied or forbidden
- configuration: invalid packages, settings or inconsistent state

Severity:
- critical: fatal errors, crashes, kernel or hardware faults, a process killed or terminated
- high: something is broken: a component in an error state, a required server unreachable,
  a possible break-in, too many failed logins
- medium: individual failures that need a look: failed logins, dropped or retried
  connections, warnings about files, blocks or configuration

If several kinds of problems appear, report the most frequent one, and list only its lines as
evidence. Only use values that appear in the log lines; never invent hosts, numbers or
component names."""


def user_message(log: str) -> str:
    return f"Log lines:\n{log}"


def messages(log: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(log)}]
