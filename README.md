# Job-Recommendation-System
This is a job recommendation system called JobMatch AI.I has a Flask backend in Python, a frontend in HTML/CSS/Bootstrap and an NLP/ML layer that uses TF-IDF + cosine similarity. Flow is: user logs in→fills profile / uploads resume→convert skills to vectors→compute similarity with all jobs→then shows ranked recommendations and stats on dashboard.

I use Flask for the web framework, SQLAlchemy with SQLite for storing users and profiles, spaCy plus regex for resume parsing, and scikit-learn’s TF-IDF + cosine similarity for job recommendations. First, I load the job dataset from CSV and build a TF-IDF matrix over the ‘Required Skills’ column. When a user signs up and uploads a resume, I extract name, email, phone, skills, education, etc. I store the profile in both the database and session. For recommendation, I convert the candidate skills into a TF-IDF vector, compute cosine similarity with each job vector, and sort jobs by similarity score. These are shown in the dashboard and recommendations page. The remaining routes handle login/signup, profile management, job listing with search and pagination, and a profile page. I also have an AJAX API for resume parsing so the form auto-fills. This completes an end-to-end job recommendation system using NLP and machine learning.”

User Interface Screenshots

Login -Page
image
Profile
image
Home Page
image
Dashboard
image
Job Match AI improves productivity in job search and provides relevant recommendations using NLP-based skill mapping and machine learning.

The proposed JobMatch AI successfully:

✔ Extracts candidate skills from resumes

✔ Computes semantic similarity using ML

✔ Delivers highly relevant job recommendations

✔ Provides a modern and interactive user experience

This demonstrates that AI improves recruitment efficiency and supports smart career decision-making.
