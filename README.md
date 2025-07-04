# NeuroScan: Brain Tumor Detection System

An AI-powered brain tumor detection system using machine learning and deep learning models. Includes classification with CNN, VGG16, AdaBoost, CatBoost, and a weighted ensemble method. Built with Python, trained in Google Colab, and deployed using Flask.

---

## 📁 Project Structure

```
NeuroScan.BrainTumorDetection/
├── models/                  # Trained models (CNN, VGG16, AdaBoost, CatBoost)
├── data/ # Sample dataset (limited due to size)
|
|---Notebooks/
|   |---BrainTumorDetection.ipynb
|   |---model_evaluation.ipynb
|
├── src/
│   ├── app.py              # Flask backend
│   ├── utils/              # Helper functions
│   └── ...
├── templates/              # HTML interface for web app
├── static/                 # CSS, JS, and assets
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🚀 Features

* Brain tumor classification: Tumor / No Tumor
* Weighted ensemble voting system for robust predictions
* Flask web interface showing ensemble + individual model predictions
* Trained in Google Colab using GPU for fast experimentation

---

## 🧠 Models Used

* CNN (custom architecture)
* VGG16 (Transfer Learning)
* AdaBoostClassifier
* CatBoostClassifier (GPU-enabled)
* Weighted average ensemble

---

## 🛠️ Dependencies

Main libraries:

* TensorFlow / Keras
* scikit-learn
* CatBoost
* OpenCV
* Flask
* Albumentations

For a full list, see `requirements.txt`.

---

## 💾 Download

Due to the large size of this project, it's not fully included on CD. You can download the complete project from the GitHub repository:

**GitHub Repository:**
`https://github.com/josiaO/NeuroScan.BrainTumorDetection.git`

To clone:

```bash
git clone https://github.com/josiaO/NeuroScan.BrainTumorDetection.git
```

---

## 🧪 How to Run

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:

   ```bash
   python src/app.py
   ```

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 📬 Contact

**Developer:** Josiah Mosses

[josia.obeid@gmail.com](mailto:josia.obeid@gmail.com)
Feel free to connect on GitHub or via email (if needed for your institution).

---

> "Tech isn’t just code — it’s impact."

