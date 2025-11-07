"""
Chrono Pro — Offline Web App (Streamlit) — PRO edition
- Single-file app with advanced AI weight controls + styling (dark/light)
- Offline (no external API calls)
- Save as chrono_pro_web_pro.py and run: streamlit run chrono_pro_web_pro.py
"""

import streamlit as st
import json, os, math, random, datetime, csv, io, re
from dateutil.relativedelta import relativedelta
from typing import Optional
# ====== Mock google module to avoid errors ======
import sys, types
sys.modules['google'] = types.SimpleNamespace()
# ==============================================
# ====== Fake pyarrow to avoid errors if not installed ======
try:
    import pyarrow
except ModuleNotFoundError:
    import types
    pyarrow = types.SimpleNamespace()
# ==========================================================
# -------------------------
# Files / Defaults
# -------------------------
DATA_FILE = "tasks_pro.json"
WEEKLY_REPORT_FILE = "weekly_report.txt"

DEFAULT_SETTINGS = {
    "daily_work_hours": 4,
    "pomodoro_work_min": 25,
    "pomodoro_short_min": 5,
    "pomodoro_long_min": 15,
    "split_threshold_hours": 3.0,
    "work_hours_per_day_for_duration_conv": 8
}

# -------------------------
# Persona lines (Chrono) — Part 2
# -------------------------
AI_NAME = "Chrono"
_CHRONO_LINES = {
    "greeting": ["Ah — you again. Let's pretend today you'll finish something.",
                 "Chrono online. I'll judge your to-do list and your life choices."],
    "add_task": ["Adding '{name}'. Cute.", "Task '{name}' received. Motivational lie incoming."],
    "complete_task": ["You actually finished '{name}'? Small miracles.", "Task '{name}' completed. Hall of Unexpected Wins updated."],
    "rescore": ["Recalculating priorities. Because chaos wasn't good enough."],
    "recommend": ["I'd suggest '{name}' next. Short, sweet, mildly soul-healing."]
}
def chrono_speak(category: str, **ctx):
    lines = _CHRONO_LINES.get(category, ["..."])
    try:
        return random.choice(lines).format(**ctx)
    except Exception:
        return random.choice(lines)

# -------------------------
# Date/time helpers
# -------------------------
def now_iso():
    return datetime.datetime.now().isoformat()

def today_date():
    return datetime.date.today()

def parse_date(s: Optional[str]):
    if not s: return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.datetime.strptime(s.strip(), fmt).date()
        except Exception:
            continue
    try:
        return datetime.date.fromisoformat(s.strip())
    except Exception:
        return None

def parse_iso_datetime(s: Optional[str]):
    if not s: return None
    try:
        return datetime.datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None

def days_until(d: Optional[datetime.date]):
    if not d: return None
    return (d - today_date()).days

# Flexible duration/deadline parsing (supports hours/days/weeks/months/years and absolute date YYYY-MM-DD)
_DUR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|d|day|days|w|wk|week|weeks|m|month|months|y|yr|year|years)?\s*$", re.I)
def parse_deadline_input(text: str) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return None
    # direct date
    dt = parse_date(text)
    if dt:
        return dt.isoformat()
    m = _DUR_RE.match(text)
    if m:
        val = float(m.group(1))
        unit = (m.group(2) or "d").lower()
        base = datetime.datetime.now()
        if unit.startswith("h"):
            new = base + datetime.timedelta(hours=val)
        elif unit.startswith("d"):
            new = base + datetime.timedelta(days=val)
        elif unit.startswith("w"):
            new = base + datetime.timedelta(weeks=val)
        elif unit.startswith("m") and unit in ("m","month","months"):
            new = base + relativedelta(months=int(val))
        elif unit.startswith("y"):
            new = base + relativedelta(years=int(val))
        else:
            new = base + datetime.timedelta(days=val)
        return new.date().isoformat()
    return None

def parse_duration_to_hours(text: str, settings: dict) -> float:
    text = (text or "").strip()
    if text == "":
        return 2.0
    m = _DUR_RE.match(text)
    if m:
        val = float(m.group(1))
        unit = (m.group(2) or "h").lower()
        hours_per_day = settings.get("work_hours_per_day_for_duration_conv", 8)
        if unit.startswith("h"):
            return round(val, 2)
        elif unit.startswith("d"):
            return round(val * hours_per_day, 2)
        elif unit.startswith("w"):
            return round(val * hours_per_day * 5, 2)
        elif unit.startswith("m"):
            return round(val * hours_per_day * 21, 2)
        elif unit.startswith("y"):
            return round(val * hours_per_day * 250, 2)
    try:
        return round(float(text), 2)
    except:
        return 2.0

# -------------------------
# Storage & undo
# -------------------------
def load_data():
    if not os.path.exists(DATA_FILE):
        return {"tasks": [], "meta": {"points":0,"achievements":[], "undo_stack":[] , "ai_model": None}, "settings": DEFAULT_SETTINGS.copy()}
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
            s = d.setdefault("settings", {})
            for k,v in DEFAULT_SETTINGS.items(): s.setdefault(k, v)
            d.setdefault("meta", {"points":0,"achievements":[], "undo_stack":[], "ai_model": None})
            d.setdefault("tasks", [])
            return d
    except Exception as e:
        st.warning(f"[Chrono] Could not read data file: {e}")
        return {"tasks": [], "meta": {"points":0,"achievements":[], "undo_stack":[], "ai_model": None}, "settings": DEFAULT_SETTINGS.copy()}

def save_data(d):
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"[Chrono] Couldn't save data: {e}")

def snapshot_state(data):
    snap = {"tasks": data.get("tasks", []), "meta": {k:v for k,v in data.get("meta", {}).items() if k!="undo_stack"}}
    return json.dumps(snap, ensure_ascii=False)

def push_undo(data):
    stack = data.setdefault("meta", {}).setdefault("undo_stack", [])
    stack.append(snapshot_state(data))
    if len(stack) > 60: stack.pop(0)
    save_data(data)

def pop_undo(data):
    stack = data.get("meta", {}).get("undo_stack", [])
    if not stack: return False
    last = stack.pop()
    snap = json.loads(last)
    data["tasks"] = snap.get("tasks", [])
    for k,v in snap.get("meta", {}).items(): data["meta"][k] = v
    save_data(data)
    return True

# -------------------------
# Priority & AI (weights editable)
# -------------------------
def init_ai_model_if_missing(data):
    model = data.setdefault("meta", {}).get("ai_model")
    if not model:
        model = {
            "w_importance": 0.7,
            "w_urgency": 0.6,
            "w_duration_penalty": 0.15,
            "w_deadline_boost": 0.25,
            "w_postpone_boost": 0.04,
            "short_task_bias": 0.05,
            "learning_rate": 0.03,
            "sarcasm_level": 0.5,
            "history": []
        }
        data["meta"]["ai_model"] = model
        save_data(data)
    return model

def ai_model_adjustment(data, task):
    model = init_ai_model_if_missing(data)
    mult = 1.0
    imp = float(task.get("importance",5))/10.0
    urg = float(task.get("urgency",5))/10.0
    duration = max(0.1, float(task.get("duration",2)))
    try:
        duration_factor = 1.0 - (math.log(duration + 1.0) * model.get("w_duration_penalty",0.15) / 2.0)
    except:
        duration_factor = 1.0
    deadline = parse_date(task.get("deadline")) if task.get("deadline") else None
    days_left = days_until(deadline) if deadline else None
    deadline_boost = 1.0
    if days_left is not None:
        if days_left < 0:
            deadline_boost = 1.6 + abs(days_left) * (model.get("w_deadline_boost",0.25) / 5.0)
        elif days_left == 0:
            deadline_boost = 1.4
        elif days_left <= 3:
            deadline_boost = 1.2
        elif days_left <= 7:
            deadline_boost = 1.1
    postpone_boost = 1.0 + min(int(task.get("times_postponed",0)), 15) * model.get("w_postpone_boost",0.04)
    short_bonus = 1.0 + (model.get("short_task_bias",0.05) if duration <= 1.0 else 0.0)
    mult *= (1 + (imp * model.get("w_importance",0.7) + urg * model.get("w_urgency",0.6)) / 10.0)
    mult *= duration_factor
    mult *= deadline_boost
    mult *= postpone_boost
    mult *= short_bonus
    mult = max(0.35, min(mult, 3.5))
    return mult

def calculate_priority(task, data):
    importance = float(task.get("importance",5))
    urgency = float(task.get("urgency",5))
    duration = max(0.1, float(task.get("duration",2)))
    times_postponed = int(task.get("times_postponed",0))
    deadline = parse_date(task.get("deadline")) if task.get("deadline") else None
    days_left = days_until(deadline) if deadline else None

    base = importance * 0.7 + urgency * 0.6
    try:
        time_factor = 1.0 / (math.log(duration + 1.5))
    except:
        time_factor = 1.0
    score = base * time_factor

    if days_left is not None:
        if days_left < 0:
            score *= (1.6 + abs(days_left) * 0.05)
        elif days_left == 0:
            score *= 1.4
        elif days_left <= 3:
            score *= 1.2
        elif days_left <= 7:
            score *= 1.1

    if duration > 8:
        score *= 0.9

    score *= (1 + min(times_postponed,15)*0.04)

    subtasks = task.get("subtasks", [])
    if subtasks:
        total = len(subtasks)
        done = sum(1 for s in subtasks if s.get("completed"))
        if total > 0:
            ratio_done = done / total
            score *= (1 - 0.25 * ratio_done)

    score += random.uniform(0, 0.05)

    try:
        ai_adj = ai_model_adjustment(data, task)
    except:
        ai_adj = 1.0

    return round(score * ai_adj, 3)

def rescore_all(data):
    for t in data.get("tasks", []):
        t["score"] = calculate_priority(t, data)
    save_data(data)

def update_ai_model_on_completion(data, task):
    model = init_ai_model_if_missing(data)
    lr = model.get("learning_rate", 0.03)
    duration = float(task.get("duration",0))
    had_deadline = bool(task.get("deadline"))
    postponed = int(task.get("times_postponed",0))
    if duration <= 1.0:
        model["short_task_bias"] = min(0.8, model.get("short_task_bias",0.05) + lr * 0.5)
    else:
        model["short_task_bias"] = max(0.0, model.get("short_task_bias",0.05) - lr * 0.1)
    if had_deadline:
        model["w_deadline_boost"] = min(1.0, model.get("w_deadline_boost",0.25) + lr * 0.2)
    else:
        model["w_deadline_boost"] = max(0.05, model.get("w_deadline_boost",0.25) - lr * 0.02)
    if postponed > 0:
        model["w_postpone_boost"] = min(0.2, model.get("w_postpone_boost",0.04) + lr * 0.1)
    else:
        model["w_postpone_boost"] = max(0.02, model.get("w_postpone_boost",0.04) - lr * 0.005)
    hist = model.setdefault("history", [])
    hist.append({"task": task.get("name"), "duration": duration, "deadline": had_deadline, "postponed": postponed, "at": now_iso()})
    if len(hist) > 500: hist.pop(0)
    save_data(data)

# -------------------------
# Core CRUD + helpers (Part 1)
# -------------------------
def add_task(data, task):
    push_undo(data)
    task.setdefault("created_at", now_iso())
    task.setdefault("completed", False)
    task.setdefault("completed_at", None)
    task.setdefault("times_postponed", 0)
    task.setdefault("subtasks", [])
    task.setdefault("notes","")
    task.setdefault("history",[])
    task["score"] = calculate_priority(task, data)
    data.setdefault("tasks", []).append(task)
    save_data(data)
    return task

def get_task_by_name(data, name):
    for t in data.get("tasks", []):
        if t.get("name","").lower() == name.lower():
            return t
    return None

def mark_task_complete(data, task):
    push_undo(data)
    task["completed"] = True
    task["completed_at"] = now_iso()
    task.setdefault("history", []).append({"action":"completed","at":now_iso()})
    pts = max(5, int(round(task.get("score",0)*3)))
    # award points (simple)
    meta = data.setdefault("meta", {})
    meta["points"] = meta.get("points",0) + pts
    meta["level"] = int(math.sqrt(max(0, meta["points"])//10 + 1))
    try:
        update_ai_model_on_completion(data, task)
    except:
        pass
    if task.get("recurring"):
        schedule_next_recurring(data, task)
    save_data(data)
    return pts

def delete_task(data, task):
    push_undo(data)
    try:
        data["tasks"].remove(task)
        save_data(data)
        return True
    except:
        return False

def schedule_next_recurring(data, task):
    freq = task.get("recurring")
    if not freq: return
    dl = parse_date(task.get("deadline")) if task.get("deadline") else None
    next_dl = None
    if dl:
        if freq == "daily":
            next_dl = dl + datetime.timedelta(days=1)
        elif freq == "weekly":
            next_dl = dl + datetime.timedelta(weeks=1)
        elif freq == "monthly":
            next_dl = dl + relativedelta(months=1)
    new_task = {k:v for k,v in task.items() if k not in ("completed","completed_at","history")}
    new_task["created_at"] = now_iso()
    new_task["completed"] = False
    new_task["times_postponed"] = 0
    if next_dl:
        new_task["deadline"] = next_dl.isoformat()
    push_undo(data)
    data["tasks"].append(new_task)
    save_data(data)

def suggest_split(task, threshold):
    duration = float(task.get("duration",0))
    if duration <= threshold: return None
    parts = int(math.ceil(duration/threshold))
    chunks = []; remaining = duration
    for i in range(parts):
        d = round(min(threshold, remaining), 2)
        chunks.append({"name": f"{task['name']} — part {i+1}", "duration": d})
        remaining -= d
    return chunks

def eisenhower_matrix(data):
    q1=[]; q2=[]; q3=[]; q4=[]
    for t in data.get("tasks", []):
        imp = t.get("importance",5) >= 6
        urg = t.get("urgency",5) >= 6
        if imp and urg: q1.append(t)
        elif imp and not urg: q2.append(t)
        elif urg and not imp: q3.append(t)
        else: q4.append(t)
    return {"Q1":q1,"Q2":q2,"Q3":q3,"Q4":q4}

def suggest_schedule_simple(data, days):
    daily_hours = data.get("settings", {}).get("daily_work_hours", 4)
    tasks = sorted([t for t in data.get("tasks", []) if not t.get("completed")], key=lambda x: x.get("score",0), reverse=True)
    schedule = { (today_date() + datetime.timedelta(days=i)).isoformat(): [] for i in range(days) }
    capacity = {d: daily_hours for d in schedule}
    for t in tasks:
        dur = float(t.get("duration",0))
        assigned = False
        for d in schedule:
            if capacity[d] >= dur:
                schedule[d].append(t["name"]); capacity[d] -= dur; assigned = True; break
        if not assigned:
            if dur > daily_hours:
                first = list(schedule.keys())[0]
                schedule[first].append(f"{t['name']} (~{daily_hours}h part)")
                capacity[first] = 0
    return schedule

def generate_weekly_report(data, filename=WEEKLY_REPORT_FILE):
    now = datetime.datetime.now()
    week_ago = now - datetime.timedelta(days=7)
    tasks = data.get("tasks", [])
    completed_recent = [t for t in tasks if t.get("completed_at") and parse_iso_datetime(t.get("completed_at")) and parse_iso_datetime(t.get("completed_at")) >= week_ago]
    total_completed = len(completed_recent)
    total_tasks = len(tasks)
    xp = data.get("meta", {}).get("points",0)
    top_tags={}
    for t in completed_recent:
        for tg in t.get("tags", []): top_tags[tg] = top_tags.get(tg,0)+1
    top_sorted = sorted(top_tags.items(), key=lambda x:x[1], reverse=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Weekly Report — {now.date().isoformat()}\n")
        f.write("="*40 + "\n")
        f.write(f"Total tasks in system: {total_tasks}\n")
        f.write(f"Tasks completed in last 7 days: {total_completed}\n")
        f.write(f"Total XP: {xp}\n\n")
        f.write("Completed tasks (last 7 days):\n")
        for t in completed_recent:
            f.write(f" - {t['name']} (completed at {t.get('completed_at')})\n")
        f.write("\nTop tags this week:\n")
        for tag,cnt in top_sorted[:10]:
            f.write(f" - {tag}: {cnt}\n")
        f.write("\nGenerated at: " + now_iso() + "\n")
    return filename

# -------------------------
# Export / Import
# -------------------------
def export_json(data): return json.dumps(data, indent=2, ensure_ascii=False)
def import_json_string(data, s):
    try:
        inc = json.loads(s)
        data.setdefault("tasks", []).extend(inc.get("tasks", []))
        data.setdefault("meta", {}).update(inc.get("meta", {}))
        data.setdefault("settings", {}).update(inc.get("settings", {}))
        save_data(data); return True, None
    except Exception as e:
        return False, str(e)

def export_csv_bytes(data):
    out = io.StringIO(); writer = csv.writer(out)
    writer.writerow(["name","importance","urgency","duration","deadline","tags","recurring","completed"])
    for t in data.get("tasks", []):
        writer.writerow([t.get("name",""), t.get("importance",""), t.get("urgency",""), t.get("duration",""), t.get("deadline",""), "|".join(t.get("tags",[])), t.get("recurring",""), t.get("completed",False)])
    return out.getvalue().encode("utf-8")

def import_csv_bytes(data, b):
    try:
        s = b.decode("utf-8"); rdr = csv.DictReader(io.StringIO(s))
        for r in rdr:
            tt = {"name": r.get("name",""), "importance": float(r.get("importance") or 5), "urgency": float(r.get("urgency") or 5), "duration": float(r.get("duration") or 2), "deadline": r.get("deadline") or None, "tags": (r.get("tags") or "").split("|") if r.get("tags") else [], "recurring": r.get("recurring") or None, "created_at": now_iso(), "completed": r.get("completed","False").lower() in ("true","1","yes")}
            data.setdefault("tasks", []).append(tt)
        save_data(data); return True, None
    except Exception as e:
        return False, str(e)

# -------------------------
# Recommendation (simple) — Part 2
# -------------------------
def recommend_next_task(data):
    rescore_all(data)
    candidates = [t for t in data.get("tasks", []) if not t.get("completed")]
    if not candidates:
        return None, ["No pending tasks."]
    avg_postponed = sum(t.get("times_postponed",0) for t in data.get("tasks",[]))/max(1,len(data.get("tasks",[])))
    completed = [t for t in data.get("tasks",[]) if t.get("completed")]
    avg_completed_duration = sum(t.get("duration",0) for t in completed)/max(1,len(completed)) if completed else 0
    weight_short_pref = min(0.6, avg_postponed / 10.0)
    best=None; best_metric=-1
    for t in candidates:
        base = t.get("score",0)
        dur = float(t.get("duration",0))
        short_bonus = (1.0/(1.0+dur)) * weight_short_pref
        m = base + short_bonus
        if dur <= 1.0: m += 0.05
        if m > best_metric: best_metric=m; best=t
    insights=[]
    if avg_postponed > 1.5: insights.append("You postpone tasks often — try splitting them.")
    if avg_completed_duration > 2.5: insights.append("You finish medium-long tasks — prioritize meaningful ones.")
    if not insights: insights.append("Good balance — keep the momentum.")
    return best, insights

# -------------------------
# Streamlit UI + styling (dark/light) + AI advanced controls
# -------------------------
st.set_page_config(page_title="Chrono Pro — Offline (PRO)", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for modern look (offline friendly) ---
BASE_CSS = """
:root {
  --accent: #00C2A8;
  --accent-2: #6EE7B7;
  --bg: #0f1724;
  --card: #0b1220;
  --muted: #9ca3af;
  --glass: rgba(255,255,255,0.03);
}
body { background-color: var(--bg); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; }
.stApp { color: #e6eef6; }
.stButton>button { border-radius: 8px; background-color: var(--accent); color: #072024; font-weight:600; padding:8px 12px; }
.stTextInput>div>input, .stNumberInput>div>input, textarea { background-color: #07121a; color: #e6eef6; border-radius:6px; }
.css-1d391kg { background-color: transparent; } /* sidebar background tweak fallback */
.streamlit-expanderHeader { color: var(--accent-2); font-weight:600; }
"""

# allow switching dark/light
if "theme_mode" not in st.session_state: st.session_state["theme_mode"] = "dark"
left_bar, right_bar = st.columns([1,3])
with left_bar:
    st.markdown("<h2 style='margin:0 0 8px 0; color:var(--accent)'>Chrono Pro</h2>", unsafe_allow_html=True)
with right_bar:
    mode = st.selectbox("Theme", options=["dark","light"], index=0 if st.session_state["theme_mode"]=="dark" else 1)
    st.session_state["theme_mode"] = mode

if st.session_state["theme_mode"] == "dark":
    st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)
else:
    # light theme CSS
    LIGHT_CSS = BASE_CSS.replace("var(--bg)", "#f6f8fb").replace("var(--card)","#ffffff").replace("#e6eef6","#07121a").replace("#07121a","#f6f8fb")
    st.markdown(f"<style>{LIGHT_CSS}</style>", unsafe_allow_html=True)

st.title("Chrono Pro — Offline (PRO)")
st.write("Personal productivity + lightweight adaptive AI. Everything offline.")

# init data
if "data" not in st.session_state:
    st.session_state["data"] = load_data()
    init_ai_model_if_missing(st.session_state["data"])
    rescore_all(st.session_state["data"])
    st.session_state["log"] = []

data = st.session_state["data"]

def log_msg(s):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["log"].insert(0, f"{ts} — {s}")
    if len(st.session_state["log"]) > 300: st.session_state["log"] = st.session_state["log"][:300]

# Sidebar: Quick actions & AI advanced settings
with st.sidebar:
    st.header("Quick Actions")
    if st.button("Rescore all"):
        rescore_all(data); save_data(data)
        log_msg(chrono_speak("rescore"))
        st.experimental_rerun()
    if st.button("Recommend next"):
        best, insights = recommend_next_task(data)
        if best:
            log_msg(chrono_speak("recommend", name=best["name"]))
            st.success(f"Recommend: {best['name']}")
            for ins in insights: st.write("- " + ins)
        else:
            st.info("No pending tasks.")
    if st.button("Generate weekly report"):
        fname = generate_weekly_report(data)
        log_msg(chrono_speak("rescore")); st.success(f"Saved: {fname}")

    st.markdown("---")
    st.header("Export / Import")
    if st.button("Export JSON"):
        payload = export_json(data)
        st.download_button("Download JSON", payload, file_name="chrono_data.json", mime="application/json")
    if st.button("Export CSV"):
        b = export_csv_bytes(data)
        st.download_button("Download CSV", b, file_name="chrono_tasks.csv", mime="text/csv")

    upj = st.file_uploader("Import JSON", type=["json"])
    if upj:
        ok, err = import_json_string(data, upj.read().decode("utf-8"))
        if ok: st.success("Imported JSON"); rescore_all(data)
        else: st.error(f"Error: {err}")
    upc = st.file_uploader("Import CSV", type=["csv"])
    if upc:
        ok, err = import_csv_bytes(data, upc.read())
        if ok: st.success("Imported CSV"); rescore_all(data)
        else: st.error(f"CSV error: {err}")

    st.markdown("---")
    st.header("AI Advanced Settings")
    model = init_ai_model_if_missing(data)
    # sliders for weights
    w_imp = st.slider("Importance weight", 0.0, 2.0, float(model.get("w_importance",0.7)), step=0.01)
    w_urg = st.slider("Urgency weight", 0.0, 2.0, float(model.get("w_urgency",0.6)), step=0.01)
    w_dpen = st.slider("Duration penalty", 0.0, 1.0, float(model.get("w_duration_penalty",0.15)), step=0.005)
    w_dead = st.slider("Deadline boost factor", 0.0, 1.5, float(model.get("w_deadline_boost",0.25)), step=0.01)
    w_post = st.slider("Postpone boost (per postpone)", 0.0, 0.2, float(model.get("w_postpone_boost",0.04)), step=0.001)
    short_bias = st.slider("Short-task bias", 0.0, 0.8, float(model.get("short_task_bias",0.05)), step=0.01)
    lr = st.slider("Learning rate", 0.0, 0.2, float(model.get("learning_rate",0.03)), step=0.001)
    sarcasm = st.slider("Sarcasm level (0 quiet — 1 savage)", 0.0, 1.0, float(model.get("sarcasm_level",0.5)), step=0.01)
    if st.button("Save AI settings"):
        model["w_importance"]=w_imp; model["w_urgency"]=w_urg; model["w_duration_penalty"]=w_dpen
        model["w_deadline_boost"]=w_dead; model["w_postpone_boost"]=w_post; model["short_task_bias"]=short_bias
        model["learning_rate"]=lr; model["sarcasm_level"]=sarcasm
        data["meta"]["ai_model"] = model
        save_data(data)
        rescore_all(data)
        st.success("AI settings saved & priorities rescored.")
    st.markdown("---")
    st.header("Settings")
    s = data.setdefault("settings", DEFAULT_SETTINGS.copy())
    s["daily_work_hours"] = st.number_input("Daily work hours", value=s.get("daily_work_hours",4), min_value=1, max_value=24, step=1)
    s["split_threshold_hours"] = st.number_input("Split threshold (hours)", value=s.get("split_threshold_hours",3.0), min_value=1.0, max_value=40.0, step=0.5)
    s["work_hours_per_day_for_duration_conv"] = st.number_input("Hours per day (conv)", value=s.get("work_hours_per_day_for_duration_conv",8), min_value=1, max_value=24, step=1)
    save_data(data)

# Main layout: left (task form/list), right (AI persona + analytics)
left, right = st.columns([2,1])

with left:
    st.subheader("Add Task — Part 1 (Task Manager)")
    with st.form("add_task_form", clear_on_submit=False):
        name = st.text_input("Task name")
        importance = st.slider("Importance", 1, 10, 5)
        urgency = st.slider("Urgency", 1, 10, 5)
        duration_raw = st.text_input("Estimated duration (e.g. '2h', '3d', '1w')", value="2h")
        duration = parse_duration_to_hours(duration_raw, data.get("settings", {}))
        deadline_raw = st.text_input("Deadline (YYYY-MM-DD or '3 days', '2 months', '1 year')", value="")
        deadline = parse_deadline_input(deadline_raw) if deadline_raw.strip() else None
        tags_raw = st.text_input("Tags (space-separated)", value="")
        tags = [t.strip() for t in tags_raw.split() if t.strip()]
        recurring = st.selectbox("Recurring", options=["", "daily", "weekly", "monthly"])
        notes = st.text_area("Notes (optional)", value="")
        col1, col2 = st.columns(2)
        with col1:
            add_clicked = st.form_submit_button("Add Task")
        with col2:
            clear_clicked = st.form_submit_button("Clear")

    if add_clicked:
        if not name.strip():
            st.error("Task name required.")
        else:
            t = {"name": name.strip(), "importance": importance, "urgency": urgency, "duration": duration, "deadline": deadline, "tags": tags, "recurring": recurring or None, "notes": notes, "created_at": now_iso(), "completed": False, "times_postponed": 0, "subtasks": [], "history": []}
            add_task(data, t)
            log_msg(chrono_speak("add_task", name=name))
            st.success(f"Added '{name}' ({duration}h).")
            rescore_all(data)

    st.markdown("---")
    st.subheader("Tasks (sorted by score) — Part 1 & Part 2")
    view_all = st.checkbox("Show completed tasks", value=False)
    tasks_list = sorted(data.get("tasks", []), key=lambda x: x.get("score",0), reverse=True)
    if not view_all: tasks_list = [t for t in tasks_list if not t.get("completed")]
    for idx, t in enumerate(tasks_list):
        cols = st.columns([4,1,1,1,1])
        with cols[0]:
            st.markdown(f"{t['name']}**  {'✅' if t.get('completed') else '⏳'}")
            st.write(f"Score: {t.get('score')}, Dur: {t.get('duration')}h, Tags: {', '.join(t.get('tags',[])) or '—'}")
            if t.get("deadline"):
                du = days_until(parse_date(t.get("deadline")))
                st.write(f"Deadline: {t.get('deadline')} ({du} days)")
        with cols[1]:
            if st.button("Complete", key=f"c_{idx}"):
                if not t.get("completed"):
                    pts = mark_task_complete(data, t)
                    log_msg(chrono_speak("complete_task", name=t["name"]))
                    st.success(f"Completed '{t['name']}' (+{pts} XP)."); st.experimental_rerun()
        with cols[2]:
            if st.button("Edit", key=f"e_{idx}"):
                st.session_state["edit_name"] = t["name"]; st.experimental_rerun()
        with cols[3]:
            if st.button("Split", key=f"s_{idx}"):
                threshold = data.get("settings", {}).get("split_threshold_hours", 3.0)
                suggested = suggest_split(t, threshold)
                if not suggested:
                    st.warning("Not long enough to split.")
                else:
                    push_undo(data)
                    t["subtasks"] = [{"name":p["name"], "duration":p["duration"], "completed":False} for p in suggested]
                    t.setdefault("history",[]).append({"action":"split","at":now_iso()})
                    save_data(data); rescore_all(data); log_msg(chrono_speak("add_task", name=t["name"])); st.success("Subtasks added."); st.experimental_rerun()
        with cols[4]:
            if st.button("Delete", key=f"d_{idx}"):
                ok = delete_task(data, t)
                if ok: log_msg(chrono_speak("rescore")); st.experimental_rerun()

    # Edit flow (inline)
    if "edit_name" in st.session_state:
        name_to_edit = st.session_state.pop("edit_name")
        task = get_task_by_name(data, name_to_edit)
        if task:
            st.markdown("---"); st.subheader(f"Edit: {task['name']}")
            with st.form("edit_form"):
                new_name = st.text_input("Name", value=task["name"])
                new_imp = st.slider("Importance", 1, 10, int(task.get("importance",5)))
                new_urg = st.slider("Urgency", 1, 10, int(task.get("urgency",5)))
                new_dur_raw = st.text_input("Duration (e.g. 3h, 2d)", value=f"{task.get('duration')}h")
                new_dur = parse_duration_to_hours(new_dur_raw, data.get("settings",{}))
                new_dl_raw = st.text_input("Deadline", value=task.get("deadline") or "")
                new_dl = parse_deadline_input(new_dl_raw) if new_dl_raw.strip() else None
                new_tags = st.text_input("Tags", value=" ".join(task.get("tags",[])))
                new_rec = st.selectbox("Recurring", options=["","daily","weekly","monthly"], index=0 if not task.get("recurring") else (["","daily","weekly","monthly"].index(task.get("recurring"))))
                save_btn = st.form_submit_button("Save")
                cancel_btn = st.form_submit_button("Cancel")
            if save_btn:
                push_undo(data)
                task["name"] = new_name
                task["importance"] = new_imp
                task["urgency"] = new_urg
                task["duration"] = new_dur
                task["deadline"] = new_dl
                task["tags"] = [x.strip() for x in new_tags.split() if x.strip()]
                task["recurring"] = new_rec or None
                task.setdefault("history",[]).append({"action":"edited","at":now_iso()})
                task["score"] = calculate_priority(task, data)
                save_data(data); rescore_all(data)
                log_msg(f"Edited: {task['name']}"); st.experimental_rerun()
            if cancel_btn: st.experimental_rerun()

with right:
    st.subheader("Chrono (Part 2 Persona & AI)")
    st.markdown("Chrono messages (latest top):")
    for msg in st.session_state["log"][:60]:
        st.write(msg)
    st.markdown("---")
    st.subheader("AI Quick Tools")
    if st.button("Run quick recommend (Part 2)"):
        best, insights = recommend_next_task(data)
        if best:
            log_msg(chrono_speak("recommend", name=best["name"]))
            st.success(f"Recommend: {best['name']}")
            for ins in insights: st.write("- " + ins)
        else:
            st.info("No pending tasks.")
    st.markdown("---")
    st.subheader("Analytics")
    tasks = data.get("tasks", [])
    total = len(tasks)
    completed = sum(1 for t in tasks if t.get("completed"))
    avg_score = round(sum(t.get("score",0) for t in tasks)/max(1,total),3) if total>0 else 0
    st.write(f"Total tasks: {total} • Completed: {completed} • Avg score: {avg_score}")
    st.markdown("---")
    st.subheader("Model Inspect")
    if st.button("Show AI model details"):
        st.json(data.get("meta",{}).get("ai_model",{}))

# Bottom actions
st.markdown("---")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("Undo"):
        ok = pop_undo(data)
        if ok: log_msg(chrono_speak("rescore")); st.experimental_rerun()
        else: st.info("No undo available.")
with col2:
    days_plan = st.number_input("Plan days (schedule suggestion)", value=3, min_value=1, max_value=30)
    if st.button("Suggest schedule"):
        sched = suggest_schedule_simple(data, days_plan)
        for d, items in sched.items():
            st.write(f"{d}")
            for it in items: st.write("- " + it)
with col3:
    if st.button("Save & Refresh"):
        save_data(data); rescore_all(data); st.success("Saved & rescored")

st.caption("Chrono Pro — Offline — Part1 (tasks & scheduling) & Part2 (AI analysis). AI weights adjustable in sidebar.")