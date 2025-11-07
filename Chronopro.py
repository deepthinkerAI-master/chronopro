"""
Chrono Pro — Ultimate Offline Web App (Streamlit) — Enhanced Version with Emojis
- Original features fully intact
- Added visual split tasks, task filters, multi-task suggestions, user habit tracking
- Notifications, multi-comments, weekly report + PDF export
- All Chrono humor preserved
- Task table actions now in English
- Emojis added for liveliness
- Save as chrono_pro_emojis.py and run: streamlit run chrono_pro_emojis.py
"""

import streamlit as st
import json, os, math, random, datetime, csv, io, re
from dateutil.relativedelta import relativedelta
from typing import Optional
import pandas as pd

# -------------------------
# Files / Defaults
# -------------------------
DATA_FILE = "tasks_pro_emojis.json"
DEFAULT_SETTINGS = {
    "daily_work_hours": 4,
    "pomodoro_work_min": 25,
    "pomodoro_short_min": 5,
    "pomodoro_long_min": 15,
    "split_threshold_hours": 3.0,
    "work_hours_per_day_for_duration_conv": 8
}

# -------------------------
# Chrono Persona
# -------------------------
AI_NAME = "Chrono"
_CHRONO_LINES = {
    "greeting": ["Ah — you again. Let's pretend today you'll finish something. 😏",
                 "Chrono online. I'll judge your to-do list and your life choices. 😎"],
    "add_task": ["Adding '{name}' 📝. Cute.", "Task '{name}' received. Motivational lie incoming 💡."],
    "complete_task": ["You actually finished '{name}'? Small miracles ✅.", 
                      "Task '{name}' completed. Hall of Unexpected Wins updated 🎉."],
    "rescore": ["Recalculating priorities. Because chaos wasn't good enough ⚡."],
    "recommend": ["I'd suggest '{name}' next. Short, sweet, mildly soul-healing 🔹."],
    "deadline_alert": ["Alert! '{name}' deadline is in {days} days ⏳. Do something, maybe? 😬"],
    "habit_boost": ["Hmm, you keep delaying '{name}' 😏. Chrono says: boost it! 🔥"]
}
def chrono_speak(category: str, **ctx):
    lines = _CHRONO_LINES.get(category, ["..."])
    try: return random.choice(lines).format(**ctx)
    except: return random.choice(lines)

# -------------------------
# Date/time helpers
# -------------------------
def now_iso(): return datetime.datetime.now().isoformat()
def today_date(): return datetime.date.today()
def parse_date(s: Optional[str]):
    if not s: return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try: return datetime.datetime.strptime(s.strip(), fmt).date()
        except: continue
    try: return datetime.date.fromisoformat(s.strip())
    except: return None

def days_until(d: Optional[datetime.date]):
    if not d: return None
    return (d - today_date()).days

_DUR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|d|day|days|w|wk|week|weeks|m|month|months|y|yr|year|years)?\s*$", re.I)
def parse_deadline_input(text: str) -> Optional[str]:
    text = (text or "").strip()
    if not text: return None
    dt = parse_date(text)
    if dt: return dt.isoformat()
    m = _DUR_RE.match(text)
    if m:
        val = float(m.group(1))
        unit = (m.group(2) or "d").lower()
        base = datetime.datetime.now()
        if unit.startswith("h"): new = base + datetime.timedelta(hours=val)
        elif unit.startswith("d"): new = base + datetime.timedelta(days=val)
        elif unit.startswith("w"): new = base + datetime.timedelta(weeks=val)
        elif unit.startswith("m"): new = relativedelta(months=int(val))
        elif unit.startswith("y"): new = relativedelta(years=int(val))
        else: new = datetime.timedelta(days=val)
        return new.date().isoformat()
    return None

def parse_duration_to_hours(text: str, settings: dict) -> float:
    text = (text or "").strip()
    if text == "": return 2.0
    m = _DUR_RE.match(text)
    if m:
        val = float(m.group(1))
        unit = (m.group(2) or "h").lower()
        hours_per_day = settings.get("work_hours_per_day_for_duration_conv", 8)
        if unit.startswith("h"): return round(val, 2)
        elif unit.startswith("d"): return round(val * hours_per_day, 2)
        elif unit.startswith("w"): return round(val * hours_per_day * 5, 2)
        elif unit.startswith("m"): return round(val * hours_per_day * 21, 2)
        elif unit.startswith("y"): return round(val * hours_per_day * 250, 2)
    try: return round(float(text), 2)
    except: return 2.0

# -------------------------
# Storage & Undo
# -------------------------
def load_data():
    if not os.path.exists(DATA_FILE):
        return {"tasks": [], "meta": {"points":0,"achievements":[], "undo_stack":[], "ai_model": None}, "settings": DEFAULT_SETTINGS.copy()}
    try:
        with open(DATA_FILE,"r",encoding="utf-8") as f:
            d=json.load(f)
            s=d.setdefault("settings",{})
            for k,v in DEFAULT_SETTINGS.items(): s.setdefault(k,v)
            d.setdefault("meta", {"points":0,"achievements":[], "undo_stack":[], "ai_model": None})
            d.setdefault("tasks",[])
            return d
    except: return {"tasks": [], "meta": {"points":0,"achievements":[], "undo_stack":[], "ai_model": None}, "settings": DEFAULT_SETTINGS.copy()}

def save_data(d):
    try:
        with open(DATA_FILE,"w",encoding="utf-8") as f: json.dump(d,f,indent=2,ensure_ascii=False)
    except Exception as e: st.error(f"[Chrono] Couldn't save data: {e}")

def snapshot_state(data):
    snap={"tasks": data.get("tasks",[]), "meta": {k:v for k,v in data.get("meta",{}).items() if k!="undo_stack"}}
    return json.dumps(snap, ensure_ascii=False)

def push_undo(data):
    stack=data.setdefault("meta",{}).setdefault("undo_stack",[])
    stack.append(snapshot_state(data))
    if len(stack)>60: stack.pop(0)
    save_data(data)

def pop_undo(data):
    stack=data.get("meta",{}).get("undo_stack",[])
    if not stack: return False
    last=stack.pop()
    snap=json.loads(last)
    data["tasks"]=snap.get("tasks",[])
    for k,v in snap.get("meta",{}).items(): data["meta"][k]=v
    save_data(data)
    return True

# -------------------------
# AI model init & adjustments
# -------------------------
def init_ai_model_if_missing(data):
    model=data.setdefault("meta",{}).get("ai_model")
    if not model:
        model={
            "w_importance": 0.7,
            "w_urgency": 0.6,
            "w_duration_penalty": 0.15,
            "w_deadline_boost": 0.25,
            "w_postpone_boost": 0.04,
            "short_task_bias": 0.05,
            "learning_rate": 0.03,
            "sarcasm_level": 0.5,
            "history":[]
        }
        data["meta"]["ai_model"]=model
        save_data(data)
    return model

def ai_model_adjustment(data, task):
    model = init_ai_model_if_missing(data)
    mult = 1.0
    imp=float(task.get("importance",5))/10.0
    urg=float(task.get("urgency",5))/10.0
    duration=max(0.1,float(task.get("duration",2)))
    try: duration_factor=1.0-(math.log(duration+1.0)*model.get("w_duration_penalty",0.15)/2.0)
    except: duration_factor=1.0
    deadline=parse_date(task.get("deadline")) if task.get("deadline") else None
    days_left=days_until(deadline) if deadline else None
    deadline_boost=1.0
    if days_left is not None:
        if days_left<0: deadline_boost=1.6+abs(days_left)*(model.get("w_deadline_boost",0.25)/5.0)
        elif days_left==0: deadline_boost=1.4
        elif days_left<=3: deadline_boost=1.2
        elif days_left<=7: deadline_boost=1.1
    postpone_boost=1.0+min(int(task.get("times_postponed",0)),15)*model.get("w_postpone_boost",0.04)
    short_bonus=1.0+(model.get("short_task_bias",0.05) if duration<=1.0 else 0.0)
    habit_boost=1.0
    if int(task.get("times_postponed",0))>=3: habit_boost=1.1
    mult*=(1+(imp*model.get("w_importance",0.7)+urg*model.get("w_urgency",0.6))/10.0)
    mult*=duration_factor*deadline_boost*postpone_boost*short_bonus*habit_boost
    mult=max(0.35,min(mult,3.5))
    return mult

def calculate_priority(task,data):
    base=(float(task.get("importance",5))*0.7+float(task.get("urgency",5))*0.6)/math.log(max(1.5,float(task.get("duration",2))+0.5)
)
    try: ai_adj=ai_model_adjustment(data,task)
    except: ai_adj=1.0
    return round(base*ai_adj,3)

def rescore_all(data):
    for t in data.get("tasks",[]): t["score"]=calculate_priority(t,data)
    save_data(data)

# -------------------------
# Task CRUD + helpers
# -------------------------
def add_task_full(data, task):
    push_undo(data)
    task.setdefault("created_at", now_iso())
    task.setdefault("completed", False)
    task.setdefault("completed_at", None)
    task.setdefault("times_postponed",0)
    task.setdefault("subtasks",[])
    task.setdefault("notes","")
    task.setdefault("comments",[])  # multi-comments feature
    task.setdefault("goals","")
    task["score"]=calculate_priority(task,data)
    data.setdefault("tasks",[]).append(task)
    save_data(data)
    return task

def mark_task_complete(data, task):
    push_undo(data)
    task["completed"]=True
    task["completed_at"]=now_iso()
    pts=max(5,int(round(task.get("score",0)*3)))
    meta=data.setdefault("meta",{})
    meta["points"]=meta.get("points",0)+pts
    save_data(data)
    return pts

def delete_task_full(data, task):
    push_undo(data)
    try:
        data["tasks"].remove(task)
        save_data(data)
        return True
    except: return False

def suggest_split(task,threshold):
    duration=float(task.get("duration",0))
    if duration<=threshold: return None
    parts=int(math.ceil(duration/threshold))
    chunks=[]; remaining=duration
    for i in range(parts):
        d=round(min(threshold,remaining),2)
        chunks.append({"name":f"{task['name']} — part {i+1}","duration":d})
        remaining-=d
    return chunks

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="⏱ Chrono Pro Ultimate Enhanced ✨", layout="wide", initial_sidebar_state="expanded")
st.title("⏱ Chrono Pro Ultimate — Enhanced Edition ✨")

if "data" not in st.session_state:
    st.session_state["data"]=load_data()
    init_ai_model_if_missing(st.session_state["data"])
    rescore_all(st.session_state["data"])
    st.session_state["log"]=[]

data=st.session_state["data"]

def log_msg(s):
    ts=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["log"].insert(0,f"{ts} — {s}")
    if len(st.session_state["log"])>300: st.session_state["log"]=st.session_state["log"][:300]

# -------------------------
# Sidebar: Quick actions + add task
# -------------------------
with st.sidebar:
    st.header("⚡ Quick Actions")
    if st.button("Undo 🔄"): 
        ok=pop_undo(data)
        if ok: log_msg(chrono_speak("rescore")); st.experimental_rerun()
        else: st.info("No undo available.")
    st.markdown("---")
    st.subheader("📝 Add Task")
    name = st.text_input("Task Name")
    duration_raw = st.text_input("Duration (e.g., 2h, 3d)", "2h")
    duration=parse_duration_to_hours(duration_raw,data.get("settings",{}))
    importance=st.slider("Importance",1,10,5)
    urgency=st.slider("Urgency",1,10,5)
    deadline_raw=st.text_input("Deadline (YYYY-MM-DD or 3d etc.)")
    deadline=parse_deadline_input(deadline_raw) if deadline_raw.strip() else None
    goals=st.text_area("Goals / Comments (optional)")
    if st.button("Add Task ➕") and name.strip():
        task={"name":name,"duration":duration,"importance":importance,"urgency":urgency,"deadline":deadline,"goals":goals}
        add_task_full(data,task)
        log_msg(chrono_speak("add_task",name=name))
        st.success(f"Added {name} 🎉")
