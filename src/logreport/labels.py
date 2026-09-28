"""Line-level labels.

Every Loghub line has a template id (EventId). I went through the templates of
each system and marked the ones that describe a problem with a category and a
severity; everything not listed here is normal. BGL comes with expert anomaly
labels, which are used instead of my own judgement. Thunderbird's 2k sample is
labelled all-normal by the experts, so it stays all-normal here too.

Only problems worth a report are labelled (severity medium and up). Things like
"session opened" or harmless warnings stay normal on purpose.
"""

from __future__ import annotations

import re

CATEGORIES = ["authentication", "network", "storage", "hardware", "service", "permission",
              "configuration"]
SEVERITIES = ["medium", "high", "critical"]

M, H, C = "medium", "high", "critical"
AUTH, NET, STOR, HW, SVC, PERM, CONF = CATEGORIES

RULES: dict[str, dict[str, tuple[str, str]]] = {
    "Linux": {
        # sshd / klogind / gdm authentication failures
        "E16": (AUTH, M), "E17": (AUTH, M), "E18": (AUTH, M), "E19": (AUTH, M),
        "E27": (AUTH, M), "E61": (AUTH, M), "E13": (AUTH, M), "E14": (AUTH, M),
        "E15": (AUTH, M), "E31": (AUTH, M),
        "E8": (SVC, H),  # logrotate: ALERT exited abnormally
        "E47": (NET, M),  # ftpd getpeername: Transport endpoint is not connected
        "E116": (NET, M),  # xinetd: Connection reset by peer
    },
    "OpenSSH": {
        "E20": (AUTH, M), "E19": (AUTH, M), "E9": (AUTH, M), "E21": (AUTH, M),
        "E10": (AUTH, M), "E13": (AUTH, M), "E12": (AUTH, M), "E8": (AUTH, M),
        "E14": (AUTH, M), "E15": (AUTH, M), "E16": (AUTH, M), "E17": (AUTH, M),
        "E7": (AUTH, M), "E6": (AUTH, M),
        "E27": (AUTH, H),  # POSSIBLE BREAK-IN ATTEMPT!
        "E5": (AUTH, H), "E4": (AUTH, H),  # Too many authentication failures
        "E18": (AUTH, H),  # PAM ignoring max retries
        "E11": (NET, M),  # fatal: Write failed: Connection reset by peer
        "E3": (NET, M),  # Did not receive identification string (port scan)
    },
    "Apache": {
        "E3": (SVC, H),  # mod_jk child workerEnv in error state
        "E5": (SVC, H),  # jk2_init() Can't find child in scoreboard
        "E6": (SVC, H),  # mod_jk child init failed
        "E4": (PERM, M),  # Directory index forbidden by rule
    },
    "HDFS": {
        "E3": (STOR, M),  # Got exception while serving block
    },
    "Hadoop": {
        "E10": (NET, M),  # Address change detected
        "E91": (NET, M),  # Retrying connect to server
        "E38": (NET, H),  # ERROR IN CONTACTING RM
        "E35": (NET, H),  # NoRouteToHostException
        "E101": (NET, C),  # task exited: NoRouteToHostException
        "E44": (STOR, M),  # Failed to renew lease
        "E33": (STOR, M), "E30": (STOR, M),  # DFSOutputStream / DataStreamer exception
        "E39": (STOR, H),  # Error Recovery for block, bad datanode
        "E99": (SVC, M),  # Task cleanup failed
        "E15": (SVC, M), "E16": (SVC, M), "E18": (SVC, M),  # attempt -> FAILED transitions
        "E2": (SVC, M), "E1": (SVC, M), "E28": (SVC, M),
        "E40": (SVC, H),  # Error writing History Event
        "E108": (SVC, H),  # thread threw an Exception
    },
    "Spark": {},
    "Zookeeper": {
        "E11": (NET, M),  # Connection broken for id
        "E24": (NET, M), "E25": (NET, M), "E42": (NET, M),  # send/recv worker torn down with it
        "E1": (NET, M),  # follower GOODBYE
        "E6": (NET, M),  # caught end of stream exception
        "E5": (NET, H),  # Cannot open channel to election address
        "E49": (SVC, H), "E50": (SVC, H),  # unexpected exception
        "E14": (SVC, H),  # ZooKeeperServer not running
    },
    "OpenStack": {
        # "Unknown base file" (E42) is routine image-cache housekeeping, not an incident
        "E43": (CONF, M),  # instance count differs between DB and hypervisor
    },
    "Windows": {
        "E21": (CONF, M),  # Failed to internally open package (CBS_E_INVALID_PACKAGE)
        "E19": (STOR, M),  # Failed to create backup log cab
    },
    "Proxifier": {
        "E4": (NET, H), "E3": (NET, H),  # could not connect to proxy / could not resolve
        "E5": (NET, M), "E6": (NET, M),  # proxy could not reach target / closed unexpectedly
    },
    "Thunderbird": {},
}

# BGL expert label -> category. All of these lines are FATAL.
BGL_CATEGORY = {
    "KERNDTLB": HW, "KERNSTOR": HW, "KERNMNTF": STOR, "APPOUT": STOR,
    "KERNTERM": SVC, "KERNRTSP": SVC, "APPCHILD": SVC,
    "APPREAD": NET, "APPRES": NET, "APPSEV": NET, "APPTO": NET, "KERNREC": NET,
}


def line_label(system: str, row: dict) -> tuple[str, str] | None:
    """(category, severity) if this line is a problem, else None."""
    if system == "BGL":
        if row["Label"] == "-":
            return None
        return BGL_CATEGORY.get(row["Label"], HW), C
    return RULES[system].get(row["EventId"])


def component(system: str, row: dict) -> str | None:
    """The program/component that wrote the line, exactly as it appears in the raw text.

    It has to be a substring of the raw line: the point of the task is to copy it,
    not to invent it.
    """
    raw = row["raw"]
    if system == "OpenSSH":
        cand = "sshd"
    elif system == "Apache":
        # "[date] [error] mod_jk child ..." -> "mod_jk"
        m = re.match(r"\[[^\]]+\] \[\w+\] (\[client [^\]]+\] )?(\S+)", raw)
        cand = m.group(2) if m else None
    elif system == "Zookeeper":
        cand = row["Component"].rsplit(":", 1)[-1]
    elif system == "Proxifier":
        cand = row["Program"]
    else:
        cand = row.get("Component") or None
    if cand and cand in raw:
        return cand
    return None
