# Job-Recommendation-System
This is a job recommendation system called JobMatch AI.I has a Flask backend in Python, a frontend in HTML/CSS/Bootstrap and an NLP/ML layer that uses TF-IDF + cosine similarity. Flow is: user logs in→fills profile / uploads resume→convert skills to vectors→compute similarity with all jobs→then shows ranked recommendations and stats on dashboard.

I use Flask for the web framework, SQLAlchemy with SQLite for storing users and profiles, spaCy plus regex for resume parsing, and scikit-learn’s TF-IDF + cosine similarity for job recommendations. First, I load the job dataset from CSV and build a TF-IDF matrix over the ‘Required Skills’ column. When a user signs up and uploads a resume, I extract name, email, phone, skills, education, etc. I store the profile in both the database and session. For recommendation, I convert the candidate skills into a TF-IDF vector, compute cosine similarity with each job vector, and sort jobs by similarity score. These are shown in the dashboard and recommendations page. The remaining routes handle login/signup, profile management, job listing with search and pagination, and a profile page. I also have an AJAX API for resume parsing so the form auto-fills. This completes an end-to-end job recommendation system using NLP and machine learning.”

User Interface Screenshots

Login -Page

<img width="975" height="607" alt="image" src="https://github.com/user-attachments/assets/10a76b78-0b5f-4fa1-a92a-c43382bc6daf" />

Profile

<img width="975" height="337" alt="image" src="https://github.com/user-attachments/assets/afc7dc72-585f-4631-8806-dcff25e0524e" />

Home Page

<img width="1097" height="533" alt="image" src="https://github.com/user-attachments/assets/a38de555-7cd9-49e1-bd40-3c9cc99cdc94" />

Dashboard

<img width="975" height="447" alt="image" src="https://github.com/user-attachments/assets/366f227d-92a5-4fb6-8ddd-d8d60ca191f2" />


Job Match AI improves productivity in job search and provides relevant recommendations using NLP-based skill mapping and machine learning.

The proposed JobMatch AI successfully:

✔ Extracts candidate skills from resumes

✔ Computes semantic similarity using ML

✔ Delivers highly relevant job recommendations

✔ Provides a modern and interactive user experience

This demonstrates that AI improves recruitment efficiency and supports smart career decision-making.
