# DigitAI — Handwritten Digit Recognition Using CNN

DigitAI is a full-stack handwritten digit recognition application built with **TensorFlow/Keras**, **OpenCV**, **Flask**, **Bootstrap**, and **React + Express**. It supports single-digit (`0–9`), double-digit (`48`, `96`), and multi-digit (`2026`, `987654`) recognition via live pointer/touch/stylus drawing and image uploads (`PNG`, `JPG`, `JPEG`).

## Project Structure

```text
DigitAI/
├── backend/
│   ├── __init__.py
│   ├── app.py                # Flask REST API (/api/health, /api/model-info, /api/predict, /api/samples, /api/evaluate)
│   ├── model_service.py      # Model loading, inference, sample gallery generator, live MNIST evaluation
│   └── preprocessing.py      # OpenCV grayscale, denoising, auto-inversion, segmentation & 28x28 MNIST normalization
├── training/
│   ├── train_cnn.py          # Trains Baseline & Enhanced CNNs, saves Keras model, metrics JSON, weights JSON & plots
│   └── evaluate_model.py     # Evaluates trained CNN on MNIST test set & synthetic multi-digit images
├── tests/
│   ├── test_digitai.py       # Automated Pytest suite (model, 10 classes, single/double/multi, edge cases, API)
│   └── verify_node_engine.mjs # Integration test against Express API
├── templates/
│   └── index.html            # Standalone Bootstrap 5 UI served directly by Flask
├── static/
│   ├── css/style.css
│   ├── js/app.js
│   ├── plots/                # Generated training curves & confusion matrix PNGs
│   └── samples/              # Sample handwriting gallery PNGs
├── src/                      # React + Vite + Tailwind frontend & Node Express TypeScript engine
├── models/                   # Saved Keras model (digitai_cnn.keras), metrics, and exported weights
├── server.ts                 # Express + Vite server (proxies to Flask or runs native TypeScript CNN engine)
├── requirements.txt
├── package.json
├── README.md
└── PROJECT_REPORT.md
```

## Windows Installation & Startup Commands

Open **PowerShell** or **Command Prompt** inside the project root directory and run:

```powershell
# 1. Create a Python virtual environment (.venv is automatically detected by DigitAI)
python -m venv .venv

# 2. Activate the virtual environment (PowerShell; use source .venv/bin/activate on Linux/macOS)
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install Python dependencies (includes tensorflow, opencv-python-headless, flask)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Train the Baseline & Enhanced CNN models on MNIST
python training/train_cnn.py

# 5. Evaluate the trained model on MNIST test images and multi-digit samples
python training/evaluate_model.py

# 6. Run automated Pytest suite
pytest tests/test_digitai.py -v

# 7. Start the Flask backend server (serves API + Bootstrap UI on port 5001)
set DIGITAI_USE_BOOTSTRAP_TEMPLATE=1
python backend/app.py
```

Open `http://127.0.0.1:5001` in your browser.

### Optional: Running the React + Express UI Locally

```powershell
npm install
npm run dev
```

When Flask (`backend/app.py`) is running on port `5001`, Express proxies `/api/*` requests to Flask. If Flask is stopped, Express runs the exported Keras weights (`models/cnn_weights.json`) natively via TypeScript.
