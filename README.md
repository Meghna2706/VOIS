---
title: AI NIDS
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.52.2
app_file: app.py
pinned: false
---

# 🛡️ AI-Based Network Intrusion Detection System (AI-NIDS)

## 📌 Project Overview

This project implements an **AI-Based Network Intrusion Detection System (AI-NIDS)** that uses **Machine Learning** to detect malicious activities in network traffic. The system is designed to classify traffic as **benign or malicious** and provide **human-readable explanations** for the predictions.

The application is built using **Python**, **Random Forest algorithm**, and an interactive **Streamlit dashboard**, with optional **Groq AI integration** for explainable analysis.

---

## 🧠 Key Features

* Uses **Random Forest Classifier** for intrusion detection
* Works with **real-world CIC-IDS2017 dataset**
* Interactive **Streamlit dashboard**
* Live **random packet simulation**
* **Rule-based explainability** for predictions
* **Groq AI (LLM)** integration for natural-language explanation
* Visual performance metrics (Accuracy & Confusion Matrix)

---

## 👥 End Users

* Network Administrators
* Cyber Security Analysts
* IT Security Teams
* Educational Institutions
* Small & Medium Enterprises (SMEs)

---

## 🧪 Dataset Used

* **CIC-IDS2017 Dataset**
* File Example:
  `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`

The dataset contains labeled network traffic representing real-world attack scenarios.

---

## ⚙️ Technologies Used

| Category             | Technology                  |
| -------------------- | --------------------------- |
| Programming Language | Python                      |
| Machine Learning     | Random Forest               |
| Web Framework        | Streamlit                   |
| Libraries            | Pandas, NumPy, Scikit-learn |
| Visualization        | Matplotlib, Seaborn         |
| Dataset              | CIC-IDS2017                 |
| Explainable AI       | Groq LLM (LLaMA 3.3)        |

---

## 🧩 System Workflow

1. Load and preprocess network traffic data
2. Split dataset into training and testing sets
3. Train Random Forest model
4. Evaluate performance using accuracy and confusion matrix
5. Simulate random network packets
6. Classify packets as benign or malicious
7. Explain predictions using rules and Groq AI

---

## 🚀 How to Run the Project

### 1️⃣ Install Required Libraries

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn groq
```

### 2️⃣ Place Dataset

Make sure the dataset file is in the project directory:

```text
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

*(Rename your Python file accordingly)*

---

## 🤖 Groq AI Integration (Optional)

* Enter your **Groq API Key** in the sidebar
* Used for generating **AI-based explanations**
* Free key available at: [https://console.groq.com/keys](https://console.groq.com/keys)

---

## 📊 Results

* High classification accuracy using Random Forest
* Clear visualization using confusion matrix
* Real-time packet simulation
* Explainable intrusion detection for students and beginners

*(Screenshots included in PPT)*

---

## 📂 Project Structure (Example)

```text
AI-NIDS/
│
├── nids_main.py
├── README.md
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
└── screenshots/
```

---

## 👩‍🎓 Author

**Subramani Meghna**
AICTE ID: APPLY_176426115269287d2037e12


---
