from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, flash
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import docx
import io
import re
import spacy
import pandas as pd
import os
import math

app = Flask(__name__)
app.secret_key = "super_secret_key_for_sessions"  # change in production

# === SQLAlchemy setup (SQLite DB) ===
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///jobmatch.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# === DB Models ===
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(120), unique=True, index=True)
    password_hash = db.Column(db.String(255))


class Profile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), index=True)
    full_name = db.Column(db.String(120))
    email = db.Column(db.String(120))
    phone = db.Column(db.String(50))
    location = db.Column(db.String(120))
    experience_level = db.Column(db.String(50))
    skills_text = db.Column(db.Text)  # skills as comma-separated string


with app.app_context():
    db.create_all()


# === Load spaCy model ===
nlp = spacy.load("en_core_web_sm")

# === Path to your CSV dataset ===
JOBS_CSV_PATH = r"C:\Users\madha\OneDrive\Documents\resume Parser (new)\resume Parser\job_recommendation_dataset.csv"


# -------------------------------------------------------------------
# Load jobs from CSV
# -------------------------------------------------------------------
def load_jobs_from_csv(path):
    if not os.path.exists(path):
        print(f"[ERROR] CSV not found at {path}")
        return []

    df = pd.read_csv(path)

    jobs = []
    for idx, row in df.iterrows():
        skills_raw = str(row.get("Required Skills", "")).strip()

        skills_list = [
            s.strip().lower()
            for s in re.split(r"[;,/|]", skills_raw)
            if s.strip()
        ]

        job = {
            "id": int(idx),
            "title": str(row.get("Job Title", "")).strip(),
            "company": str(row.get("Company", "")).strip(),
            "location": str(row.get("Location", "")).strip(),
            "experience_level": str(row.get("Experience Level", "")).strip(),
            "salary": str(row.get("Salary", "")).strip(),
            "industry": str(row.get("Industry", "")).strip(),
            "skills": skills_list,
            "raw_required_skills": skills_raw,
            "description": str(row.get("Industry", "")).strip() or "No description provided.",
        }
        jobs.append(job)

    return jobs


try:
    JOBS = load_jobs_from_csv(JOBS_CSV_PATH)
    print(f"[INFO] Loaded {len(JOBS)} jobs from CSV.")
except Exception as e:
    print(f"[ERROR] Failed to load jobs CSV: {e}")
    JOBS = []

# -------------------------------------------------------------------
# Build TF-IDF model over job required skills
# -------------------------------------------------------------------
if JOBS:
    job_texts = [job.get("raw_required_skills", "") or "" for job in JOBS]
    vectorizer = TfidfVectorizer(stop_words="english")
    job_tfidf_matrix = vectorizer.fit_transform(job_texts)
else:
    vectorizer = None
    job_tfidf_matrix = None


# -------------------------------------------------------------------
# Simple login required decorator
# -------------------------------------------------------------------
def login_required(view_func):
    def wrapper(*args, **kwargs):
        if "user_email" not in session:
            return redirect(url_for("login_signup"))
        return view_func(*args, **kwargs)

    wrapper.__name__ = view_func.__name__
    return wrapper


# -------------------------------------------------------------------
# Helpers to sync Profile between DB and session
# -------------------------------------------------------------------
def profile_row_to_session_dict(profile_row: Profile | None):
    """Convert Profile DB row to the dict used in session['profile']."""
    if not profile_row:
        return None
    skills = []
    if profile_row.skills_text:
        skills = [
            s.strip().lower()
            for s in profile_row.skills_text.split(",")
            if s.strip()
        ]
    return {
        "full_name": profile_row.full_name,
        "email": profile_row.email,
        "phone": profile_row.phone,
        "location": profile_row.location,
        "experience_level": profile_row.experience_level,
        "skills": skills,
    }


def save_profile_for_user(user_id: int, profile_dict: dict):
    """Upsert Profile row in DB based on session profile dict."""
    if not user_id:
        return
    profile_row = Profile.query.filter_by(user_id=user_id).first()
    if not profile_row:
        profile_row = Profile(user_id=user_id)

    profile_row.full_name = profile_dict.get("full_name")
    profile_row.email = profile_dict.get("email")
    profile_row.phone = profile_dict.get("phone")
    profile_row.location = profile_dict.get("location")
    profile_row.experience_level = profile_dict.get("experience_level")
    skills_list = profile_dict.get("skills") or []
    profile_row.skills_text = ", ".join(skills_list)

    db.session.add(profile_row)
    db.session.commit()


# -------------------------------------------------------------------
# Text extraction helpers
# -------------------------------------------------------------------
def extract_text_from_pdf(file_storage):
    text = ""
    reader = PdfReader(file_storage.stream)
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text


def extract_text_from_docx(file_storage):
    file_bytes = file_storage.read()
    file_stream = io.BytesIO(file_bytes)
    document = docx.Document(file_stream)
    text = "\n".join([p.text for p in document.paragraphs])
    return text


def extract_text(file_storage):
    filename = file_storage.filename.lower()
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_storage)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file_storage)
    elif filename.endswith(".txt"):
        return file_storage.read().decode("utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


# -------------------------------------------------------------------
# Parsing helpers
# -------------------------------------------------------------------
def parse_email(text):
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(pattern, text)
    return match.group(0) if match else None


def parse_phone(text):
    pattern = r"(\+?\d[\d\s\-()]{8,}\d)"
    match = re.search(pattern, text)
    return match.group(0) if match else None


def detect_name_spacy(text):
    """
    Advanced name detection:
    1) Look at the first few lines (header area) and try to detect a full name.
    2) Ignore headings like 'RESUME', 'CURRICULUM VITAE', etc.
    3) Use a mix of heuristics + spaCy PERSON entities.
    """
    if not text:
        return None

    # Take first 15 non-empty lines as "header"
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    header_lines = lines[:15]

    banned_headers = {
        "resume",
        "curriculum vitae",
        "curriculum-vitae",
        "cv",
        "profile",
        "bio-data",
        "biodata",
    }

    # 1) Heuristic: look for a line that looks like a name
    for line in header_lines:
        lower = line.lower()
        if lower in banned_headers:
            continue
        # skip lines that look like email or phone
        if "@" in line or any(ch.isdigit() for ch in line):
            continue

        tokens = line.split()
        # Candidate if 2-4 words, most start with capital letter
        if 1 < len(tokens) <= 4:
            caps = sum(1 for t in tokens if t[0].isalpha() and t[0].isupper())
            if caps >= max(2, len(tokens) - 1):
                # run spaCy just on this line to confirm
                doc_line = nlp(line)
                persons = [ent.text.strip() for ent in doc_line.ents if ent.label_ == "PERSON"]
                if persons:
                    return persons[0]
                # even if spaCy missed, this is a strong candidate
                return line

    # 2) Fallback: run spaCy on first 1000 chars and pick first PERSON
    doc = nlp(text[:1000])
    persons = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
    if persons:
        return persons[0]

    # 3) Last fallback: first non-empty non-header line
    for line in header_lines:
        lower = line.lower()
        if lower in banned_headers:
            continue
        if "@" in line or any(ch.isdigit() for ch in line):
            continue
        return line

    return None



def extract_section(text, section_keywords, next_section_keywords):
    text_lower = text.lower()
    start_idx = -1

    for key in section_keywords:
        idx = text_lower.find(key.lower())
        if idx != -1:
            start_idx = idx
            break

    if start_idx == -1:
        return None

    end_candidates = []
    for key in next_section_keywords:
        idx = text_lower.find(key.lower(), start_idx + 1)
        if idx != -1:
            end_candidates.append(idx)

    if end_candidates:
        end_idx = min(end_candidates)
        return text[start_idx:end_idx].strip()
    return text[start_idx:].strip()


def extract_education(text):
    return extract_section(
        text,
        ["education", "academic background", "qualifications", "academic profile"],
        ["experience", "work experience", "projects", "skills", "certifications"],
    )


def extract_experience(text):
    return extract_section(
        text,
        ["experience", "work experience", "professional experience", "employment history"],
        ["education", "skills", "projects", "certifications"],
    )


def extract_projects(text):
    return extract_section(
        text,
        ["projects", "academic projects", "personal projects"],
        ["education", "experience", "skills", "certifications"],
    )


SKILLS_DB = [
    "python",
    "java",
    "c++",
    "c",
    "javascript",
    "typescript",
    "html",
    "css",
    "react",
    "node.js",
    "angular",
    "vue.js",
    "django",
    "flask",
    "spring boot",
    "express.js",
    "machine learning",
    "deep learning",
    "data analysis",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "git",
    "docker",
    "kubernetes",
    "linux",
    "aws",
    "azure",
    "gcp",
    "rest api",
    "data structures",
    "algorithms",
    "nlp",
    # Add more based on your dataset
]


def extract_skills(text):
    """
    Extract skills using SKILLS_DB + dedicated 'Skills' section.
    """
    text_lower = text.lower()
    found = set()

    # 1) Direct substring match in whole text
    for skill in SKILLS_DB:
        if skill in text_lower:
            found.add(skill)

    # 2) Focus on lines under Skills section for more candidates
    skills_section = extract_section(
        text,
        ["skills", "technical skills", "skills & abilities", "key skills"],
        ["experience", "work experience", "education", "projects", "certifications"],
    )
    if skills_section:
        lines = skills_section.splitlines()
        for l in lines:
            tokens = re.split(r"[,\-•;|/]", l)
            for tok in tokens:
                tok = tok.strip().lower()
                if 1 < len(tok) <= 50:
                    # if token itself is in DB
                    if tok in SKILLS_DB:
                        found.add(tok)
                    else:
                        # check if any known skill appears inside token
                        for skill in SKILLS_DB:
                            if skill in tok:
                                found.add(skill)

    return sorted(found)



def parse_resume(file_storage):
    """
    Advanced resume parsing:
    - extract name, email, phone
    - extract education, experience, projects sections
    - extract skills using SKILLS_DB and skills section
    """
    text = extract_text(file_storage)

    # basic contact info
    email = parse_email(text)
    phone = parse_phone(text)
    name = detect_name_spacy(text)

    education = extract_education(text)
    experience = extract_experience(text)
    projects = extract_projects(text)
    skills = extract_skills(text)

    parsed = {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills,
        "education": education,
        "experience": experience,
        "projects": projects,
        "raw_text": text,
    }
    return parsed



# -------------------------------------------------------------------
# Job recommendation (TF-IDF + cosine similarity)
# -------------------------------------------------------------------
def recommend_jobs_tfidf(candidate_skills, top_k=10):
    """
    Use TF-IDF + cosine similarity on the 'Required Skills' text.
    Also compute matching_skills from the job's structured skill list.
    """
    if not candidate_skills or vectorizer is None or job_tfidf_matrix is None:
        return []

    cand_set = set(s.lower() for s in candidate_skills)
    user_text = " ".join(candidate_skills)
    user_vec = vectorizer.transform([user_text])

    similarities = cosine_similarity(user_vec, job_tfidf_matrix).flatten()

    scored_jobs = []
    for idx, sim in enumerate(similarities):
        job = JOBS[idx]
        job_skills = set(job.get("skills", []))
        match = cand_set.intersection(job_skills)
        scored_jobs.append(
            {
                **job,
                "score": float(sim),
                "matching_skills": sorted(list(match)),
            }
        )

    scored_jobs.sort(key=lambda j: j["score"], reverse=True)
    return scored_jobs[:top_k]


# (old skill-overlap scoring retained, but now unused)
def score_jobs_by_skills(candidate_skills, jobs, top_k=10):
    cand_set = set(s.lower() for s in candidate_skills)
    scored = []

    for job in jobs:
        job_skills = set(job.get("skills", []))
        if not job_skills:
            score = 0.0
            match = set()
        else:
            match = cand_set.intersection(job_skills)
            score = len(match) / len(job_skills)

        scored.append(
            {
                **job,
                "score": round(score, 2),
                "matching_skills": sorted(list(match)),
            }
        )

    scored.sort(key=lambda j: j["score"], reverse=True)
    return scored[:top_k]


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route("/")
def root():
    if "user_email" in session:
        return redirect(url_for("home"))
    return redirect(url_for("login_signup"))


# Step 1: Login / Sign up
@app.route("/auth", methods=["GET", "POST"])
def login_signup():
    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "login":
            email = request.form.get("login_email", "").strip().lower()
            password = request.form.get("login_password", "").strip()

            user = User.query.filter_by(email=email).first()
            if not user or not check_password_hash(user.password_hash, password):
                flash("Invalid email or password", "danger")
            else:
                session["user_id"] = user.id
                session["user_email"] = user.email
                session["user_name"] = user.name

                # load profile from DB into session
                profile_row = Profile.query.filter_by(user_id=user.id).first()
                profile_dict = profile_row_to_session_dict(profile_row)
                if profile_dict:
                    session["profile"] = profile_dict

                return redirect(url_for("personal_info"))

        elif form_type == "signup":
            name = request.form.get("signup_name", "").strip()
            email = request.form.get("signup_email", "").strip().lower()
            password = request.form.get("signup_password", "").strip()

            if not name or not email or not password:
                flash("Please fill all fields.", "warning")
            else:
                existing = User.query.filter_by(email=email).first()
                if existing:
                    flash("Email already registered. Please login.", "warning")
                else:
                    hashed = generate_password_hash(password)
                    new_user = User(name=name, email=email, password_hash=hashed)
                    db.session.add(new_user)
                    db.session.commit()

                    session["user_id"] = new_user.id
                    session["user_email"] = new_user.email
                    session["user_name"] = new_user.name
                    session.pop("profile", None)

                    flash("Account created successfully!", "success")
                    return redirect(url_for("personal_info"))

    return render_template("login_signup.html", hide_nav = True)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_signup"))


# Step 2: Personal info (manual / auto-fill)
@app.route("/personal-info", methods=["GET", "POST"])
@login_required
def personal_info():
    profile = session.get("profile", {})

    if request.method == "POST":
        full_name = request.form.get("full_name")
        email = request.form.get("email")
        phone = request.form.get("phone")
        location = request.form.get("location")
        experience_level = request.form.get("experience_level")
        skills_text = request.form.get("skills", "")

        skills = [s.strip().lower() for s in re.split(r"[;,]", skills_text) if s.strip()]

        profile = {
            "full_name": full_name,
            "email": email,
            "phone": phone,
            "location": location,
            "experience_level": experience_level,
            "skills": skills,
        }

        session["profile"] = profile

        # also persist to DB
        user_id = session.get("user_id")
        save_profile_for_user(user_id, profile)

        flash("Profile saved successfully!", "success")
        return redirect(url_for("home"))

    return render_template("personal_info.html", profile=profile)


# API: parse resume to auto-fill personal info form (AJAX)
@app.route("/api/parse-resume", methods=["POST"])
@login_required
def api_parse_resume():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    resume_file = request.files["resume"]
    if resume_file.filename == "":
        return jsonify({"error": "Empty file name"}), 400

    try:
        parsed = parse_resume(resume_file)
        return jsonify({"parsed": parsed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Step 3: Home / Dashboard / Jobs / Recommendations / Profile

@app.route("/home")
@login_required
def home():
    profile = session.get("profile")
    return render_template("home.html", profile=profile, jobs_count=len(JOBS))


@app.route("/dashboard")
@login_required
def dashboard():
    profile = session.get("profile")
    total_jobs = len(JOBS)
    skills_count = len(profile.get("skills", [])) if profile else 0

    # profile completeness (name, email, location, experience, skills)
    completeness = 0
    if profile:
        fields = [
            bool(profile.get("full_name")),
            bool(profile.get("email")),
            bool(profile.get("location")),
            bool(profile.get("experience_level")),
            bool(profile.get("skills")),
        ]
        completeness = int(sum(fields) / len(fields) * 100)

    # jobs in same location
    jobs_in_location = 0
    if profile and profile.get("location"):
        loc_lower = profile["location"].lower()
        jobs_in_location = sum(
            1
            for job in JOBS
            if job["location"] and loc_lower in job["location"].lower()
        )

    # recommendations and matched jobs (TF-IDF based)
    top_recs = []
    matched_jobs_count = 0
    if profile and profile.get("skills"):
        all_scored = recommend_jobs_tfidf(profile["skills"], top_k=len(JOBS))
        matched_jobs_count = len(all_scored)
        top_recs = all_scored[:5]

    return render_template(
        "dashboard.html",
        profile=profile,
        total_jobs=total_jobs,
        skills_count=skills_count,
        jobs_in_location=jobs_in_location,
        matched_jobs_count=matched_jobs_count,
        top_recs=top_recs,
        completeness=completeness,
    )


@app.route("/jobs")
@login_required
def all_jobs():
    query = request.args.get("q", "").strip().lower()
    page = request.args.get("page", 1, type=int)
    per_page = 50  # jobs per page

    # Filter jobs by search query
    if query:
        filtered_jobs = []
        for job in JOBS:
            text = f"{job['title']} {job['company']} {job['location']} {job['industry']}".lower()
            if query in text:
                filtered_jobs.append(job)
    else:
        filtered_jobs = JOBS

    total_jobs = len(filtered_jobs)
    total_pages = max(1, math.ceil(total_jobs / per_page))

    # clamp page number
    if page < 1:
        page = 1
    if page > total_pages:
        page = total_pages

    start = (page - 1) * per_page
    end = start + per_page
    page_jobs = filtered_jobs[start:end]

    return render_template(
        "all_jobs.html",
        jobs=page_jobs,
        query=query,
        page=page,
        total_pages=total_pages,
        total_jobs=total_jobs,
        per_page=per_page,
    )


@app.route("/recommendations")
@login_required
def recommendations():
    profile = session.get("profile")

    if not profile or not profile.get("skills"):
        flash("Please fill your personal info and skills first.", "warning")
        return redirect(url_for("personal_info"))

    recs = recommend_jobs_tfidf(profile["skills"], top_k=10)
    return render_template(
        "recommendations.html", profile=profile, recs=recs, is_dashboard=False
    )


@app.route("/profile", methods=["GET"])
@login_required
def profile():
    profile_data = session.get("profile")
    return render_template("profile.html", profile=profile_data)



if __name__ == "__main__":
    app.run(debug=True)
