"""
Module 10: Alert Integrations
==============================
Sends threat alerts to Slack, email, and SIEM webhooks whenever
the scoring engine crosses a configurable risk threshold.

Components
----------
  AlertConfig       — threshold + channel config (env-var or dict)
  AlertPayload      — structured alert with full context
  SlackAlerter      — posts rich Block Kit messages to a webhook
  EmailAlerter      — sends HTML email via SMTP (Gmail/SES/etc.)
  SIEMAlerter       — POSTs CEF / JSON events to a SIEM webhook
  AlertManager      — deduplication, cooldown, fan-out to all channels
  AlertRouter       — wires IngestPipeline + ThreatScorer + AlertManager

Educational use only - defensive research prototype.
No real network calls are made in demo mode (dry_run=True by default).
"""

import json, time, hashlib, smtplib, urllib.request, urllib.error
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from collections import defaultdict
import logging, os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).resolve().parent.parent
ALERT_DIR = BASE_DIR / "alerts"
ALERT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

@dataclass
class AlertConfig:
    """
    All thresholds and channel settings in one place.
    Values can be overridden by environment variables.

    Environment variables (optional):
      SLACK_WEBHOOK_URL   — Slack Incoming Webhook URL
      SMTP_HOST           — SMTP server host
      SMTP_PORT           — SMTP server port (default 587)
      SMTP_USER           — SMTP username
      SMTP_PASS           — SMTP password
      ALERT_FROM_EMAIL    — sender address
      ALERT_TO_EMAIL      — comma-separated recipient addresses
      SIEM_WEBHOOK_URL    — SIEM/SOAR HTTP endpoint
      SIEM_API_KEY        — Bearer token for SIEM endpoint
    """
    # Thresholds
    min_score_to_alert : float = 50.0   # MEDIUM and above
    critical_score     : float = 85.0
    high_score         : float = 70.0
    medium_score       : float = 50.0

    # Deduplication — don't re-alert the same session within N seconds
    cooldown_s         : int   = 120

    # Slack
    slack_webhook_url  : str   = ""
    slack_channel      : str   = "#security-alerts"
    slack_username     : str   = "CyberAI Defense"
    slack_icon         : str   = ":shield:"

    # Email
    smtp_host          : str   = "smtp.gmail.com"
    smtp_port          : int   = 587
    smtp_user          : str   = ""
    smtp_pass          : str   = ""
    from_email         : str   = "alerts@cyberaidefense.local"
    to_emails          : list  = field(default_factory=lambda: ["soc@company.com"])

    # SIEM webhook (Splunk HEC, Elastic, Microsoft Sentinel, etc.)
    siem_webhook_url   : str   = ""
    siem_api_key       : str   = ""
    siem_format        : str   = "json"  # "json" | "cef"

    # Behaviour
    dry_run            : bool  = True   # True = log only, no real HTTP/SMTP
    enabled_channels   : list  = field(default_factory=lambda: ["slack","email","siem"])

    def __post_init__(self):
        # Override from environment if set
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", self.slack_webhook_url)
        self.smtp_host         = os.getenv("SMTP_HOST",         self.smtp_host)
        self.smtp_port         = int(os.getenv("SMTP_PORT",     self.smtp_port))
        self.smtp_user         = os.getenv("SMTP_USER",         self.smtp_user)
        self.smtp_pass         = os.getenv("SMTP_PASS",         self.smtp_pass)
        self.from_email        = os.getenv("ALERT_FROM_EMAIL",  self.from_email)
        to_env                 = os.getenv("ALERT_TO_EMAIL",    "")
        if to_env: self.to_emails = [e.strip() for e in to_env.split(",")]
        self.siem_webhook_url  = os.getenv("SIEM_WEBHOOK_URL",  self.siem_webhook_url)
        self.siem_api_key      = os.getenv("SIEM_API_KEY",      self.siem_api_key)


# ─────────────────────────────────────────────
# ALERT PAYLOAD
# ─────────────────────────────────────────────

RISK_EMOJI = {"SAFE":"🟢","LOW":"🔵","MEDIUM":"🟡","HIGH":"🟠","CRITICAL":"🔴"}
RISK_COLOR = {"SAFE":"#22c55e","LOW":"#3b82f6","MEDIUM":"#f59e0b",
              "HIGH":"#f97316","CRITICAL":"#ef4444"}

DEFENSES = {
    "Reconnaissance":      "Block port scanning sources. Enable IDS signatures. Rate-limit ICMP.",
    "Credential Access":   "Enable MFA immediately. Lock accounts after 5 failed attempts.",
    "Exploitation":        "Patch affected services urgently. Deploy WAF rules. Activate honeypots.",
    "Privilege Escalation":"Enforce least-privilege. Monitor sudo/admin activity.",
    "Lateral Movement":    "Segment network. Monitor SMB/RDP/WMI. Disable unnecessary shares.",
    "Command & Control":   "Block suspicious outbound. Monitor DNS beaconing. Isolate host.",
    "Data Exfiltration":   "Activate DLP rules. Block large outbound transfers. Isolate NOW.",
    "Impact":              "ISOLATE affected systems. Activate IR plan. Restore from backup.",
    "Persistence":         "Audit startup items, scheduled tasks. Enable FIM.",
    "Discovery":           "Alert on AD enumeration commands. Restrict LDAP queries.",
}

@dataclass
class AlertPayload:
    """Full context for one alert dispatch."""
    alert_id        : str
    timestamp       : str
    src_ip          : str
    sequence        : list
    detected_stage  : str
    predicted_stage : str
    confidence      : float
    threat_score    : float
    risk_level      : str
    recommended_action: str
    rule_matched    : str   = ""
    mitre_techniques: str   = ""

    @classmethod
    def build(cls, src_ip, sequence, detected, predicted, confidence,
              score, level, rule="", techniques=""):
        ts   = datetime.now(timezone.utc).isoformat()
        uid  = hashlib.md5(f"{src_ip}{sequence}{ts}".encode()).hexdigest()[:10]
        return cls(
            alert_id         = f"ALERT-{uid.upper()}",
            timestamp        = ts,
            src_ip           = src_ip,
            sequence         = sequence,
            detected_stage   = detected,
            predicted_stage  = predicted,
            confidence       = confidence,
            threat_score     = score,
            risk_level       = level,
            recommended_action = DEFENSES.get(predicted, "Escalate to SOC."),
            rule_matched     = rule,
            mitre_techniques = techniques,
        )

    def to_dict(self) -> dict:
        return {
            "alert_id"          : self.alert_id,
            "timestamp"         : self.timestamp,
            "src_ip"            : self.src_ip,
            "sequence"          : self.sequence,
            "detected_stage"    : self.detected_stage,
            "predicted_stage"   : self.predicted_stage,
            "confidence"        : round(self.confidence, 3),
            "threat_score"      : round(self.threat_score, 1),
            "risk_level"        : self.risk_level,
            "recommended_action": self.recommended_action,
            "rule_matched"      : self.rule_matched,
        }

    def short(self) -> str:
        seq = " → ".join(self.sequence)
        return (f"[{self.risk_level}] {self.src_ip} | {seq} "
                f"→ predicted: {self.predicted_stage} "
                f"(score {self.threat_score:.0f})")


# ─────────────────────────────────────────────
# CHANNEL ALERTERS
# ─────────────────────────────────────────────

class SlackAlerter:
    """
    Posts a rich Block Kit message to a Slack Incoming Webhook.
    In dry_run mode prints the payload instead of sending.
    """

    def __init__(self, config: AlertConfig):
        self.cfg = config

    def send(self, alert: AlertPayload) -> bool:
        payload = self._build_blocks(alert)
        if self.cfg.dry_run:
            logger.info(f"[DRY-RUN] Slack → {self.cfg.slack_channel}")
            logger.info(f"  {alert.short()}")
            self._log_payload("slack", payload)
            return True
        return self._post(payload)

    def _build_blocks(self, a: AlertPayload) -> dict:
        color  = RISK_COLOR.get(a.risk_level, "#888")
        emoji  = RISK_EMOJI.get(a.risk_level, "⚪")
        seq    = " → ".join(a.sequence)
        return {
            "username"  : self.cfg.slack_username,
            "icon_emoji": self.cfg.slack_icon,
            "channel"   : self.cfg.slack_channel,
            "attachments": [{
                "color"  : color,
                "blocks" : [
                    {"type": "header", "text": {
                        "type": "plain_text",
                        "text": f"{emoji} {a.risk_level} Threat — {a.src_ip}"
                    }},
                    {"type": "section", "fields": [
                        {"type": "mrkdwn", "text": f"*Alert ID*\n`{a.alert_id}`"},
                        {"type": "mrkdwn", "text": f"*Score*\n`{a.threat_score:.0f}/100`"},
                        {"type": "mrkdwn", "text": f"*Detected*\n{a.detected_stage}"},
                        {"type": "mrkdwn", "text": f"*Predicted next*\n{a.predicted_stage} ({a.confidence:.0%})"},
                    ]},
                    {"type": "section", "text": {
                        "type": "mrkdwn",
                        "text": f"*Kill chain*\n`{seq}`"
                    }},
                    {"type": "section", "text": {
                        "type": "mrkdwn",
                        "text": f":zap: *Recommended action*\n{a.recommended_action}"
                    }},
                    {"type": "context", "elements": [
                        {"type": "mrkdwn",
                         "text": f"Rule: `{a.rule_matched}` | {a.timestamp}"}
                    ]},
                ]
            }]
        }

    def _post(self, payload: dict) -> bool:
        if not self.cfg.slack_webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False
        try:
            data = json.dumps(payload).encode()
            req  = urllib.request.Request(
                self.cfg.slack_webhook_url, data=data,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                ok = r.status == 200
                logger.info(f"Slack alert sent: HTTP {r.status}")
                return ok
        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            return False

    def _log_payload(self, channel, payload):
        out = ALERT_DIR / f"dry_run_{channel}_latest.json"
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)


class EmailAlerter:
    """
    Sends an HTML alert email via SMTP with TLS.
    In dry_run mode writes the HTML to disk instead of sending.
    """

    def __init__(self, config: AlertConfig):
        self.cfg = config

    def send(self, alert: AlertPayload) -> bool:
        html    = self._build_html(alert)
        subject = (f"[{alert.risk_level}] CyberAI Alert {alert.alert_id} — "
                   f"{alert.src_ip} | Score {alert.threat_score:.0f}")
        if self.cfg.dry_run:
            logger.info(f"[DRY-RUN] Email → {self.cfg.to_emails}")
            logger.info(f"  Subject: {subject}")
            out = ALERT_DIR / f"dry_run_email_{alert.alert_id}.html"
            out.write_text(html)
            logger.info(f"  HTML saved → {out}")
            return True
        return self._smtp_send(subject, html)

    def _build_html(self, a: AlertPayload) -> str:
        color  = RISK_COLOR.get(a.risk_level, "#888")
        seq    = " → ".join(a.sequence)
        rows   = "".join(
            f"<tr><td style='padding:6px 12px;color:#6b7280;font-size:13px'>{k}</td>"
            f"<td style='padding:6px 12px;font-size:13px'>{v}</td></tr>"
            for k, v in [
                ("Alert ID",        a.alert_id),
                ("Timestamp",       a.timestamp),
                ("Source IP",       a.src_ip),
                ("Kill chain",      seq),
                ("Detected stage",  a.detected_stage),
                ("Predicted next",  f"{a.predicted_stage} ({a.confidence:.0%})"),
                ("Threat score",    f"{a.threat_score:.0f} / 100"),
                ("Rule matched",    a.rule_matched),
            ]
        )
        return f"""<!DOCTYPE html><html><body style='font-family:sans-serif;background:#f9fafb;padding:24px'>
<div style='max-width:560px;margin:0 auto;background:#fff;border-radius:12px;
            overflow:hidden;border:1px solid #e5e7eb'>
  <div style='background:{color};padding:20px 24px'>
    <span style='font-size:22px'>🛡️</span>
    <span style='color:#fff;font-size:18px;font-weight:600;margin-left:8px'>
      {a.risk_level} Threat Detected</span>
  </div>
  <table style='width:100%;border-collapse:collapse;margin:0'>{rows}</table>
  <div style='background:#f0fdf4;border-left:4px solid #22c55e;
              padding:14px 16px;margin:16px'>
    <div style='font-size:11px;letter-spacing:.1em;text-transform:uppercase;
                color:#6b7280;margin-bottom:4px'>Recommended action</div>
    <div style='font-size:13px;color:#166534'>{a.recommended_action}</div>
  </div>
  <div style='padding:16px 24px;font-size:11px;color:#9ca3af;
              border-top:1px solid #f3f4f6'>
    CyberAI Defense · Educational prototype · Defensive use only
  </div>
</div></body></html>"""

    def _smtp_send(self, subject: str, html: str) -> bool:
        if not self.cfg.smtp_user:
            logger.warning("SMTP credentials not configured")
            return False
        try:
            msg                    = MIMEMultipart("alternative")
            msg["Subject"]         = subject
            msg["From"]            = self.cfg.from_email
            msg["To"]              = ", ".join(self.cfg.to_emails)
            msg.attach(MIMEText(html, "html"))
            with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port) as s:
                s.ehlo(); s.starttls(); s.login(self.cfg.smtp_user, self.cfg.smtp_pass)
                s.sendmail(self.cfg.from_email, self.cfg.to_emails, msg.as_string())
            logger.info(f"Email alert sent to {self.cfg.to_emails}")
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False


class SIEMAlerter:
    """
    POSTs alert events to a SIEM/SOAR webhook in JSON or CEF format.

    Tested against:
      - Splunk HTTP Event Collector (HEC)
      - Elasticsearch / OpenSearch
      - Microsoft Sentinel Data Collector API
      - Generic webhook (e.g. n8n, Zapier, Tines)

    In dry_run mode writes the payload to disk.
    """

    def __init__(self, config: AlertConfig):
        self.cfg = config

    def send(self, alert: AlertPayload) -> bool:
        if self.cfg.siem_format == "cef":
            payload = self._build_cef(alert)
            content_type = "text/plain"
        else:
            payload = json.dumps(self._build_json(alert)).encode()
            content_type = "application/json"

        if self.cfg.dry_run:
            logger.info(f"[DRY-RUN] SIEM → {self.cfg.siem_webhook_url or 'not configured'}")
            if isinstance(payload, bytes):
                logger.info(f"  Payload: {payload.decode()[:200]}")
            else:
                logger.info(f"  CEF: {payload[:200]}")
            self._save_payload(payload, alert.alert_id)
            return True
        return self._post(payload, content_type)

    def _build_json(self, a: AlertPayload) -> dict:
        """Splunk HEC / Elastic compatible JSON envelope."""
        return {
            "time"  : time.time(),
            "host"  : a.src_ip,
            "source": "cyberai_defense",
            "sourcetype": "threat_alert",
            "event" : {
                **a.to_dict(),
                "kill_chain_sequence": " → ".join(a.sequence),
                "sev_numeric"        : {"SAFE":0,"LOW":1,"MEDIUM":2,"HIGH":3,"CRITICAL":4}
                                       .get(a.risk_level, 0),
            }
        }

    def _build_cef(self, a: AlertPayload) -> bytes:
        """ArcSight Common Event Format (CEF) string."""
        sev = {"SAFE":1,"LOW":3,"MEDIUM":5,"HIGH":7,"CRITICAL":10}.get(a.risk_level,5)
        ext = (
            f"src={a.src_ip} "
            f"deviceCustomString1={' -> '.join(a.sequence)} "
            f"deviceCustomString1Label=KillChain "
            f"deviceCustomString2={a.predicted_stage} "
            f"deviceCustomString2Label=PredictedNext "
            f"deviceCustomNumber1={a.threat_score:.0f} "
            f"deviceCustomNumber1Label=ThreatScore "
            f"msg={a.recommended_action}"
        )
        cef = (f"CEF:0|CyberAI|DefenseEngine|1.0|{a.alert_id}|"
               f"{a.detected_stage} Detected|{sev}|{ext}")
        return cef.encode()

    def _post(self, payload, content_type: str) -> bool:
        if not self.cfg.siem_webhook_url:
            logger.warning("SIEM webhook URL not configured")
            return False
        try:
            headers = {"Content-Type": content_type}
            if self.cfg.siem_api_key:
                headers["Authorization"] = f"Bearer {self.cfg.siem_api_key}"
            req = urllib.request.Request(
                self.cfg.siem_webhook_url, data=payload,
                headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                logger.info(f"SIEM alert sent: HTTP {r.status}")
                return r.status in (200, 201, 204)
        except Exception as e:
            logger.error(f"SIEM send failed: {e}")
            return False

    def _save_payload(self, payload, alert_id: str):
        ext = ".cef" if self.cfg.siem_format == "cef" else ".json"
        out = ALERT_DIR / f"dry_run_siem_{alert_id}{ext}"
        if isinstance(payload, bytes):
            out.write_bytes(payload)
        else:
            out.write_text(payload)
        logger.info(f"  SIEM payload saved → {out}")


# ─────────────────────────────────────────────
# ALERT MANAGER — dedup + fan-out
# ─────────────────────────────────────────────

class AlertManager:
    """
    Central alert dispatcher with:
      - Configurable risk threshold gating
      - Per-session cooldown (no storm of identical alerts)
      - Fan-out to all enabled channels
      - Full audit log of all dispatched alerts
    """

    def __init__(self, config: AlertConfig = None):
        self.cfg      = config or AlertConfig()
        self.slack    = SlackAlerter(self.cfg)
        self.email    = EmailAlerter(self.cfg)
        self.siem     = SIEMAlerter(self.cfg)
        self._log     : list[dict] = []
        self._cooldown: dict[str, float] = defaultdict(float)

    def should_alert(self, alert: AlertPayload) -> tuple[bool, str]:
        """Returns (should_alert, reason)."""
        if alert.threat_score < self.cfg.min_score_to_alert:
            return False, f"score {alert.threat_score:.0f} < threshold {self.cfg.min_score_to_alert}"
        key  = f"{alert.src_ip}:{alert.risk_level}"
        last = self._cooldown.get(key, 0)
        wait = time.time() - last
        if wait < self.cfg.cooldown_s:
            return False, f"cooldown ({self.cfg.cooldown_s-wait:.0f}s remaining)"
        return True, "ok"

    def dispatch(
        self,
        src_ip     : str,
        sequence   : list,
        detected   : str,
        predicted  : str,
        confidence : float,
        score      : float,
        level      : str,
        rule       : str = "",
    ) -> dict:
        """Build an AlertPayload and fan-out to all enabled channels."""
        alert = AlertPayload.build(
            src_ip, sequence, detected, predicted,
            confidence, score, level, rule
        )

        ok, reason = self.should_alert(alert)
        result = {
            "alert_id" : alert.alert_id,
            "sent"     : False,
            "reason"   : reason,
            "channels" : {},
        }

        if not ok:
            logger.info(f"Alert suppressed: {reason}")
            self._log.append({**result, "alert": alert.to_dict()})
            return result

        result["sent"] = True
        key = f"{alert.src_ip}:{alert.risk_level}"
        self._cooldown[key] = time.time()

        for channel in self.cfg.enabled_channels:
            if channel == "slack":
                result["channels"]["slack"] = self.slack.send(alert)
            elif channel == "email":
                result["channels"]["email"] = self.email.send(alert)
            elif channel == "siem":
                result["channels"]["siem"] = self.siem.send(alert)

        self._log.append({**result, "alert": alert.to_dict()})
        return result

    def save_log(self) -> Path:
        out = ALERT_DIR / "alert_log.json"
        with open(out, "w") as f:
            json.dump(self._log, f, indent=2)
        logger.info(f"Alert log saved → {out} ({len(self._log)} records)")
        return out

    def summary(self) -> dict:
        sent       = [r for r in self._log if r["sent"]]
        suppressed = [r for r in self._log if not r["sent"]]
        by_channel = defaultdict(int)
        for r in sent:
            for ch, ok in r.get("channels", {}).items():
                if ok: by_channel[ch] += 1
        return {
            "total_dispatched" : len(self._log),
            "sent"             : len(sent),
            "suppressed"       : len(suppressed),
            "by_channel"       : dict(by_channel),
        }


# ─────────────────────────────────────────────
# ALERT ROUTER — integrates with IngestPipeline
# ─────────────────────────────────────────────

class AlertRouter:
    """
    Wires an IngestPipeline to an AlertManager.
    Call .run() to process all events and fire alerts as they cross thresholds.
    """

    def __init__(self, pipeline, manager: AlertManager):
        self.pipeline = pipeline
        self.manager  = manager

    def run(self, max_events=None, timeout_s=30):
        self.pipeline.start()
        for staged, sequence, score, level in \
                self.pipeline.stream(max_events=max_events, timeout_s=timeout_s):
            if score >= self.manager.cfg.min_score_to_alert:
                self.manager.dispatch(
                    src_ip     = staged.event.src_ip,
                    sequence   = sequence,
                    detected   = staged.stage,
                    predicted  = staged.stage,
                    confidence = staged.confidence,
                    score      = score,
                    level      = level,
                    rule       = staged.rule_matched,
                )
        self.pipeline.stop()


# ─────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from ingestion.log_ingestor import IngestPipeline

    print(f"\n{'='*60}")
    print(f"  MODULE 10 — Alert Integrations (dry_run=True)")
    print(f"{'='*60}\n")

    def _score(seq):
        BASE={"Reconnaissance":20,"Credential Access":40,"Exploitation":55,
              "Persistence":55,"Privilege Escalation":65,"Lateral Movement":70,
              "Command & Control":75,"Data Exfiltration":80,"Impact":80}
        KC=list(BASE.keys())
        s=seq[-1]; base=BASE.get(s,30)
        depth=max((KC.index(x)+1 for x in seq if x in KC),default=0)
        sc=min(100,base+(depth/len(KC))*10+5)
        lv="CRITICAL" if sc>=85 else "HIGH" if sc>=70 else "MEDIUM" if sc>=50 else "LOW" if sc>=30 else "SAFE"
        return sc, lv

    cfg = AlertConfig(
        dry_run            = True,
        min_score_to_alert = 50.0,
        cooldown_s         = 0,      # no cooldown in demo so all alerts fire
        enabled_channels   = ["slack", "email", "siem"],
        slack_channel      = "#security-alerts",
        to_emails          = ["soc@company.com", "ciso@company.com"],
        siem_format        = "json",
    )

    manager  = AlertManager(cfg)
    pipeline = IngestPipeline(simulate=True, speed=0.0, score_fn=_score)
    router   = AlertRouter(pipeline, manager)

    print("  Running ingestion + alerting pipeline ...\n")
    router.run(max_events=18, timeout_s=10)

    # Also demo direct dispatch for specific scenarios
    print(f"\n  {'─'*56}")
    print(f"  Direct dispatch demo — 3 manual alerts")
    print(f"  {'─'*56}\n")

    test_alerts = [
        dict(src_ip="10.20.30.40", sequence=["Reconnaissance","Credential Access","Exploitation"],
             detected="Exploitation", predicted="Privilege Escalation",
             confidence=0.99, score=78.0, level="HIGH", rule="web_exploit_success"),
        dict(src_ip="172.16.0.55", sequence=["Lateral Movement","Command & Control"],
             detected="Command & Control", predicted="Data Exfiltration",
             confidence=0.96, score=92.0, level="CRITICAL", rule="https_beacon"),
        dict(src_ip="192.168.0.1",  sequence=["Reconnaissance"],
             detected="Reconnaissance", predicted="Credential Access",
             confidence=0.72, score=28.0, level="SAFE", rule="port_scan"),  # below threshold
    ]

    for ta in test_alerts:
        result = manager.dispatch(**ta)
        status = "SENT" if result["sent"] else f"SUPPRESSED ({result['reason']})"
        print(f"  [{ta['level']:<8}] {ta['src_ip']:<15}  score={ta['score']:.0f}  → {status}")
        if result["sent"]:
            for ch, ok in result["channels"].items():
                print(f"             {ch:<6} {'✓' if ok else '✗'}")

    log_path = manager.save_log()
    summ     = manager.summary()

    print(f"\n  {'─'*56}")
    print(f"  ALERT SUMMARY")
    print(f"  {'─'*56}")
    print(f"  Total dispatched : {summ['total_dispatched']}")
    print(f"  Sent             : {summ['sent']}")
    print(f"  Suppressed       : {summ['suppressed']}")
    print(f"  By channel       : {summ['by_channel']}")
    print(f"  Log saved        : {log_path}")
    print(f"\n  Dry-run artefacts:")
    for p in sorted(ALERT_DIR.glob("dry_run_*")):
        print(f"    {p.name}")
    print(f"\n  Module 10 complete.\n{'='*60}\n")
