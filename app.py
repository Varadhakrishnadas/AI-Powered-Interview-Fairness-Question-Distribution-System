import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io
import re
import sqlite3
import hashlib
import json
import smtplib
import os
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

# --- 0. DATABASE & SECURITY SETUP ---
DB_NAME = 'interview_system_final_v2.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, job_role TEXT, 
                 qualification TEXT, candidate_ref TEXT, data TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (email TEXT PRIMARY KEY, password_hash TEXT)''')
    conn.commit()
    conn.close()

def make_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_login(email, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT password_hash FROM users WHERE email = ?', (email,))
    data = c.fetchall()
    conn.close()
    if data: return data[0][0] == make_hash(password)
    return False

def create_user(email, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (email, password_hash) VALUES (?, ?)', (email, make_hash(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def save_to_history_db(job_role, qual, candidate_ref, data_list):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO history (timestamp, job_role, qualification, candidate_ref, data) VALUES (?, ?, ?, ?, ?)", 
              (ts, job_role, qual, candidate_ref, json.dumps(data_list)))
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect(DB_NAME)
    try: df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    except: df = pd.DataFrame()
    conn.close()
    return df

# --- 1. UI CONFIGURATION ---
st.set_page_config(page_title="Interview Analyzer Pro", layout="wide", page_icon="⚖️")
init_db()

st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    .main-header { color: #1e293b; font-size: 2.5rem; font-weight: 800; margin-bottom: 20px; border-bottom: 4px solid #3b82f6; padding-bottom: 10px; }
    .card { background: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1); }
    .badge { padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: inline-block; }
    .hard { background-color: #fee2e2; color: #991b1b; } .medium { background-color: #fef9c3; color: #854d0e; } .easy { background-color: #dcfce7; color: #166534; }
    .correction-badge { background-color: #e0f2fe; color: #0369a1; font-size: 0.7rem; font-weight: bold; margin-left: 10px; border: 1px solid #bae6fd; }
    .stat-box { background-color: white; padding: 10px; border-radius: 8px; border: 1px solid #dee2e6; text-align: center; }
    .relevancy-text { float: right; color: #0369a1; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
# --- 2. AUTHENTICATION ---
if 'authenticated' not in st.session_state: st.session_state['authenticated'] = False
if 'user_email' not in st.session_state: st.session_state['user_email'] = ''

def auth_screen():
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.markdown("<h2 style='text-align:center;'>HR Portal</h2>", unsafe_allow_html=True)
        t1, t2 = st.tabs(["Login", "Register"])
        with t1:
            with st.form("login"):
                e = st.text_input("Email")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Sign In", use_container_width=True):
                    if check_login(e, p):
                        st.session_state['authenticated'] = True; st.session_state['user_email'] = e; st.rerun()
                    else: st.error("Invalid credentials.")
        with t2:
            with st.form("signup"):
                ne = st.text_input("New Email"); npwd = st.text_input("New Password", type="password")
                if st.form_submit_button("Register", use_container_width=True):
                    if create_user(ne, npwd): st.success("Account created!")
                    else: st.error("User exists.")

if not st.session_state['authenticated']: auth_screen(); st.stop()

# --- 3. ENGINES ---
def determine_difficulty(text):
    """
    Heuristic function to auto-detect difficulty based on keywords and length
    when no explicit label is provided in the upload.
    """
    text_lower = text.lower()
    hard_keywords = ['analyze', 'design', 'optimize', 'debug', 'architecture', 'compare', 'contrast', 'solve', 'implement', 'algorithm', 'critical', 'deploy']
    easy_keywords = ['define', 'list', 'what is', 'name', 'describe', 'state', 'explain', 'meaning', 'full form']
    
    # Logic: Complex keywords or long questions -> Hard
    if any(k in text_lower for k in hard_keywords) or len(text.split()) > 25:
        return "Hard"
    # Logic: Simple keywords and short questions -> Easy
    elif any(k in text_lower for k in easy_keywords) and len(text.split()) < 15:
        return "Easy"
    
    return "Medium"

def get_smart_jd(role):
    role = role.lower()
    if "java" in role: return "Key Skills: Java 8+, Spring Boot, Hibernate, SQL. Responsibilities: Backend systems, REST API, DB optimization."
    elif "python" in role: return "Key Skills: Python, Django/Flask, Pandas, API. Responsibilities: Server-side logic, automation, data processing."
    elif "data" in role: return "Key Skills: Python, SQL, ML (Scikit-learn), Visualization. Responsibilities: Statistical analysis, modeling, data cleaning."
    elif "test" in role or "qa" in role: return "Key Skills: Selenium, Manual Testing, Jira. Responsibilities: Test cases, regression testing, QA."
    else: return f"Requires expertise in {role} related systems and problem solving."

def extract_pdf(f):
    try:
        f.seek(0); r = PyPDF2.PdfReader(io.BytesIO(f.read())); t = ""
        for p in r.pages:
            c = p.extract_text()
            if c: t += c + "\n"
        return t
    except: return ""

def parse_qs(t):
    pat = r'(?=\[Easy\]|\[Medium\]|\[Hard\]|\n\s*\d+[\.\)]|\n\s*Q:|\n\s*Question:)'
    segs = re.split(pat, t); cleaned = []
    for s in segs:
        sc = s.strip()
        if len(sc) > 10:
            m = re.search(r'\[(Easy|Medium|Hard)\]', sc, re.I)
            txt = re.sub(r'\[(Easy|Medium|Hard)\]', '', sc, flags=re.I).strip()
            # Use found label if exists, else auto-detect
            d = m.group(1).capitalize() if m else determine_difficulty(txt)
            cleaned.append({'text': txt, 'difficulty': d})
    return cleaned

@st.cache_data
def get_rel(qs, jd):
    if not jd or not qs: return np.zeros(len(qs))
    v = TfidfVectorizer(stop_words='english'); corp = [jd] + [str(x) for x in qs]
    try:
        m = v.fit_transform(corp)
        if m.shape[0] > 1: return np.round(cosine_similarity(m[0:1], m[1:]).flatten() * 100, 2)
    except: pass
    return np.zeros(len(qs))

def create_single_pdf(name, sub, ql):
    buf = io.BytesIO(); doc = SimpleDocTemplate(buf, pagesize=letter); styles = getSampleStyleSheet(); story = []
    story.append(Paragraph(f"Candidate Assessment: {name}", styles['Heading1']))
    story.append(Paragraph(sub, styles['Heading2'])); story.append(Spacer(1, 12))
    for i, q in enumerate(ql):
        story.append(Paragraph(f"<b>Q{i+1}:</b> {q['Question']}", styles['Normal']))
        story.append(Paragraph(f"<i>Difficulty: {q['Difficulty']} | Fit Score: {q.get('Relevancy',0)}%</i>", styles['BodyText']))
        story.append(Spacer(1, 12))
    doc.build(story); buf.seek(0); return buf

def create_consolidated_pdf(assignments_dict, qualifications_dict, job_title_global):
    buf = io.BytesIO(); doc = SimpleDocTemplate(buf, pagesize=letter); styles = getSampleStyleSheet(); story = []
    for name, ql in assignments_dict.items():
        qual = qualifications_dict.get(name, "General")
        story.append(Paragraph(f"Candidate: {name}", styles['Heading1']))
        story.append(Paragraph(f"Role: {job_title_global} | Qual: {qual}", styles['Heading2'])); story.append(Spacer(1, 12))
        for i, q in enumerate(ql):
            story.append(Paragraph(f"Q{i+1}: {q['Question']}", styles['Normal']))
            story.append(Paragraph(f"Difficulty: {q['Difficulty']}", styles['BodyText'])); story.append(Spacer(1, 12))
        story.append(PageBreak())
    doc.build(story); buf.seek(0); return buf

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title(f"User: {st.session_state['user_email']}")
    if st.button("Logout"): st.session_state['authenticated'] = False; st.rerun()
    st.divider()
    
    # Initialize Uploader Key
    if 'uploader_key' not in st.session_state: st.session_state['uploader_key'] = str(uuid.uuid4())

    if st.button("Reset Analysis", use_container_width=True):
        # Clear Data
        st.session_state['papers'] = {}
        st.session_state['logs'] = []
        st.session_state['fairness_alerts'] = []
        
        # Reset Widgets Values
        st.session_state['jt_val'] = ""
        st.session_state['nc_val'] = 3
        st.session_state['qlmt_val'] = 6
        st.session_state['jd_manual'] = "" # Clear manual JD text if any
        st.session_state['uploader_key'] = str(uuid.uuid4()) # Forces new uploader
        
        # Clear JD key to allow reset to empty
        if 'jd_val' in st.session_state: del st.session_state['jd_val']
        
        st.rerun()

    st.divider()
    src = st.radio("Upload Source Format:", ["CSV", "PDF"])
    uf = st.file_uploader(f"Choose Question Pool ({src})", type=[src.lower()], key=st.session_state['uploader_key'])
    st.divider()
    
    job_title_input = st.text_input("Job Title", placeholder="e.g. Python Developer", key="jt_val")
    if job_title_input:
        jd_val = get_smart_jd(job_title_input)
        # No key here to allow "Smart JD" feature to work dynamically on Title change
        jd_final = st.text_area("Job Description", value=jd_val, height=120) 
    else:
        # Key here allows us to clear this specific widget instance when resetting
        jd_final = st.text_area("Job Description", height=120, key="jd_manual")
    st.session_state['jd_text'] = jd_final

    st.divider()
    nc = st.slider("Number of Candidates", 1, 20, 3, key="nc_val")
    qlmt = st.number_input("Questions Per Candidate", 3, 15, 6, key="qlmt_val")
    
    quals = {}
    for i in range(nc):
        nm = f"Candidate {i+1}"
        quals[nm] = st.selectbox(f"Qual for {nm}", ["MCA", "BCA", "B.Tech (CS)", "M.Tech (CS)", "MSc (CS)", "BSc (CS)"], key=f"sel_{i}")
    st.session_state['quals'] = quals
    
    st.divider()
    dist_btn = st.button("Audit & Distribute", use_container_width=True)
    eq_btn = st.button("Apply Equity Algorithm", use_container_width=True, type="primary")

# --- 5. LOGIC ---
if 'papers' not in st.session_state: st.session_state['papers'] = {}
if 'logs' not in st.session_state: st.session_state['logs'] = []
if 'fairness_alerts' not in st.session_state: st.session_state['fairness_alerts'] = []

# RANDOMIZED INITIAL DISTRIBUTION (MODIFIED FOR NOTIFICATIONS & LEVELS)
if uf and dist_btn:
    pool = []
    if src == "CSV":
        try:
            uf.seek(0); df = pd.read_csv(uf)
        except:
            uf.seek(0); df = pd.read_csv(uf, encoding='latin-1', on_bad_lines='skip', engine='python')
        df.columns = [c.strip().capitalize() for c in df.columns]
        q_col = 'Question' if 'Question' in df.columns else df.columns[0]
        d_col = 'Difficulty' if 'Difficulty' in df.columns else None
        for _, r in df.iterrows():
            q_txt = str(r[q_col]).strip()
            # AUTO-RECOGNIZE DIFFICULTY IF MISSING
            if d_col and str(r[d_col]).strip().capitalize() in ["Easy", "Medium", "Hard"]:
                d_val = str(r[d_col]).strip().capitalize()
            else:
                d_val = determine_difficulty(q_txt)
                
            if len(q_txt) > 10: pool.append({'Question': q_txt, 'Difficulty': d_val, 'Status': 'Original', 'OrigQ': q_txt, 'OrigD': d_val})
    else:
        txt = extract_pdf(uf); raw = parse_qs(txt)
        pool = [{'Question': p['text'], 'Difficulty': p['difficulty'], 'Status': 'Original', 'OrigQ': p['text'], 'OrigD': p['difficulty']} for p in raw]

    if pool:
        results = {f"Candidate {i+1}": [] for i in range(nc)}
        
        # 1. Ensure at least one of each level for everyone if possible, then shuffle rest
        for name in results:
            p_work = pool.copy(); np.random.shuffle(p_work)
            
            # Try to pick 1 Easy, 1 Medium, 1 Hard first
            for lvl in ["Easy", "Medium", "Hard"]:
                found = next((x for x in p_work if x['Difficulty'] == lvl), None)
                if found:
                    results[name].append(found.copy())
                    p_work.remove(found) # Only remove from this temp list for this candidate
            
            # Fill the rest randomly from the full pool (minus what we already picked)
            # Note: We re-copy pool to ensure we don't run out, but randomness ensures difference
            while len(results[name]) < qlmt:
                # Refresh pool if empty
                if not p_work: p_work = pool.copy(); np.random.shuffle(p_work)
                results[name].append(p_work.pop(0).copy())

       # 2. Logic for Fairness Notifications (Comparative Analysis)
        alerts = []
        
        # Calculate stats for all levels
        stats = []
        for name, qlist in results.items():
            c_e = len([q for q in qlist if q['Difficulty'] == 'Easy'])
            c_m = len([q for q in qlist if q['Difficulty'] == 'Medium'])
            c_h = len([q for q in qlist if q['Difficulty'] == 'Hard'])
            stats.append({'Name': name, 'Easy': c_e, 'Medium': c_m, 'Hard': c_h})
            
        if stats:
            # Check Easy, Medium, and Hard separately
            for level, icon in [('Easy', '⚠️'), ('Medium', '⚖️'), ('Hard', '🚨')]:
                counts = [s[level] for s in stats]
                if not counts: continue
                
                max_val = max(counts)
                min_val = min(counts)
                
                # ONLY alert if there is a difference (Max > Min)
                if max_val > min_val:
                    # Find who has the maximum
                    leaders = [s['Name'] for s in stats if s[level] == max_val]
                    leader_str = ", ".join(leaders)
                    
                    # Find who has the minimum (for comparison)
                    min_users = [s['Name'] for s in stats if s[level] == min_val]
                    min_user_str = min_users[0] if len(min_users) == 1 else "others"
                    
                    alerts.append(f"{icon} Unfair {level} Load: **{leader_str}** has {max_val} {level} questions, which is higher compared to **{min_user_str}** ({min_val}).")

        st.session_state['fairness_alerts'] = alerts
        
        for name, qlist in results.items():
            scores = get_rel([q['Question'] for q in qlist], st.session_state['jd_text'])
            for i, s in enumerate(scores): qlist[i]['Relevancy'] = s
        
        st.session_state['papers'] = results
        st.session_state['logs'] = []
        st.success("Questions distributed. Check Fairness Notifications below.")

# MANDATORY EQUITY ALGORITHM (STRUCTURAL PARITY)
if eq_btn and st.session_state.get('papers'):
    logs = []
    e_tar = qlmt // 3; h_tar = qlmt // 3; m_tar = qlmt - e_tar - h_tar
    targets = {'Easy': e_tar, 'Medium': m_tar, 'Hard': h_tar}

    for name, qlist in st.session_state['papers'].items():
        qual = st.session_state['quals'].get(name, "Gen")
        for q in qlist: q['Question'], q['Difficulty'], q['Status'] = q['OrigQ'], q['OrigD'], 'Original'

        for t_level, t_count in targets.items():
            current_count = len([q for q in qlist if q['Difficulty'] == t_level])
            while current_count < t_count:
                cur_counts = {k: len([x for x in qlist if x['Difficulty'] == k]) for k in targets}
                surplus = [k for k, v in cur_counts.items() if v > targets[k]]
                others = [q for q in qlist if q['Difficulty'] in surplus and q['Status'] == 'Original']
                if not others: others = [q for q in qlist if q['Difficulty'] in surplus]
                if not others: break
                q_mod = others[0]
                q_mod['Question'] = f"{q_mod['Question']} [AI Balanced for {t_level}]"; q_mod['Difficulty'] = t_level; q_mod['Status'] = 'Converted'
                logs.append({"Candidate": name, "Action": f"Changed to {t_level}", "Rationale": f"Forced Parity for {qual}"})
                current_count += 1
                
    st.session_state['logs'] = logs; st.session_state['fairness_alerts'] = [] # Clear alerts after fixing
    st.success("Parity forced for all candidates.")

# --- 6. MAIN DASHBOARD ---
st.markdown("<div class='main-header'>Interview Analyzer Hub ⚖️</div>", unsafe_allow_html=True)

# Notification Banner for HR
if st.session_state.get('fairness_alerts'):
    st.subheader("Fairness Distribution Notifications")
    for alert in st.session_state['fairness_alerts']: st.warning(alert)
    st.divider()

tabs = st.tabs(["Question Papers", "Fairness Analytics", "Thematic Map", "Archives", "Audit Logs", "Email Console"])

with tabs[0]:
    if st.session_state.get('papers'):
        c_tabs = st.tabs(list(st.session_state['papers'].keys()))
        for idx, (name, ql) in enumerate(st.session_state['papers'].items()):
            with c_tabs[idx]:
                df_q = pd.DataFrame(ql); cnts = df_q['Difficulty'].value_counts()
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"<div class='stat-box'><b>Easy</b><br>{cnts.get('Easy',0)}</div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='stat-box'><b>Medium</b><br>{cnts.get('Medium',0)}</div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='stat-box'><b>Hard</b><br>{cnts.get('Hard',0)}</div>", unsafe_allow_html=True)
                c4.markdown(f"<div class='stat-box'><b>Fit</b><br>{df_q['Relevancy'].mean():.1f}%</div>", unsafe_allow_html=True)
                st.divider()
                exp_c1, exp_c2, exp_c3, exp_c4 = st.columns(4)
                with exp_c1:
                    if st.button(f"Archive {name}", key=f"save_{name}"):
                        save_to_history_db(job_title_input, st.session_state['quals'][name], name, ql); st.toast(f"Archived {name}")
                with exp_c2: st.download_button(f"Download as PDF ({name})", create_single_pdf(name, f"Role: {job_title_input}", ql), f"{name}.pdf", "application/pdf", use_container_width=True)
                with exp_c3: st.download_button(f"Download as CSV ({name})", df_q.to_csv(index=False).encode('utf-8'), f"{name}.csv", "text/csv", use_container_width=True)
                with exp_c4: edit_mode = st.toggle(f"Manual Edit: {name}", key=f"edit_{name}")
                for q_idx, q in enumerate(ql):
                    d_c = q['Difficulty'].lower(); tag = '<span class="badge correction-badge">AI Balanced</span>' if q['Status'] == 'Converted' else ''
                    if edit_mode: q['Question'] = st.text_area(f"Q{q_idx+1}", value=q['Question'], key=f"txt_{name}_{q_idx}")
                    else: st.markdown(f'<div class="card"><span class="relevancy-text">{q.get("Relevancy",0)}%</span><span class="badge {d_c}">{q["Difficulty"]}</span> {tag}<p style="margin-top:10px; font-weight:600;">{q["Question"]}</p></div>', unsafe_allow_html=True)
    else:
        st.info("👋 Welcome! Upload questions and click 'Audit & Distribute' to generate balanced papers.")

with tabs[1]:
    if st.session_state.get('papers'):
        all_data = []
        for n, qs in st.session_state['papers'].items():
            for q in qs: all_data.append({"Candidate": n, "Difficulty": q['Difficulty'], "Relevancy": q.get('Relevancy',0)})
        if all_data:
            df_a = pd.DataFrame(all_data)
            st.plotly_chart(px.histogram(df_a, x="Candidate", color="Difficulty", barmode="group", color_discrete_map={"Easy":"#4ade80", "Medium":"#facc15", "Hard":"#f87171"}, title="Mathematical Fairness Parity Check"), use_container_width=True)
    else:
        st.info("Analysis requires processed data.")

with tabs[2]:
    if st.session_state.get('papers'):
        all_data = []
        for n, qs in st.session_state['papers'].items():
            for q in qs: all_data.append({"Candidate": n, "Difficulty": q['Difficulty']})
        if all_data:
            df_a = pd.DataFrame(all_data)
            st.plotly_chart(px.sunburst(df_a, path=['Candidate', 'Difficulty'], color_discrete_map={'Hard': '#f87171', 'Medium': '#facc15', 'Easy': '#4ade80'}, title="Candidate Complexity Hierarchy"), use_container_width=True)
    else:
        st.info("Analysis requires processed data.")

with tabs[3]:
    h = load_history()
    if not h.empty:
        for _, row in h.iterrows():
            with st.expander(f"{row['timestamp']} | {row['job_role']} | {row['candidate_ref']}"): 
                data_list = json.loads(row['data'])
                df_hist = pd.DataFrame(data_list)
                st.table(df_hist[['Question', 'Difficulty']])
                csv = df_hist.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"Archive_{row['candidate_ref']}_{row['timestamp']}.csv",
                    mime="text/csv",
                    key=f"dl_hist_{row['id']}"
                )
    else: st.info("History archive is currently empty.")

with tabs[4]:
    if st.session_state['logs']: st.table(pd.DataFrame(st.session_state['logs']))
    else: st.info("No audit logs yet.")

with tabs[5]:
    if st.session_state.get('papers'):
        st.subheader("Send Consolidated Report")
        em_col1, em_col2 = st.columns(2)
        with em_col1:
            sender_email = st.text_input("Sender Gmail"); app_pwd = st.text_input("Gmail App Password", type="password")
        with em_col2:
            receiver_email = st.text_input("Interviewer Email")
            if st.button("Email Unified PDF", use_container_width=True):
                try:
                    pdf_merged = create_consolidated_pdf(st.session_state['papers'], st.session_state['quals'], job_title_input)
                    msg = MIMEMultipart(); msg['From'], msg['To'], msg['Subject'] = sender_email.strip(), receiver_email.strip(), f"Interview Packets: {job_title_input}"
                    msg.attach(MIMEText("Greetings. Consolidated report attached.", 'plain'))
                    att = MIMEApplication(pdf_merged.read(), Name="Consolidated_Interviews.pdf"); att['Content-Disposition'] = 'attachment; filename="Consolidated_Interviews.pdf"'; msg.attach(att)
                    s = smtplib.SMTP('smtp.gmail.com', 587); s.starttls(); s.login(sender_email.strip(), app_pwd.strip()); s.send_message(msg); s.quit(); st.success("Dispatched!")
                except Exception as ex: st.error(f"Failed: {ex}")
    else:
        st.info("Generate papers to enable emailing.")