

# DigitAI — Handwritten Digit Recognition Using CNN

**DigitAI** is a web-based handwritten digit recognition application that uses a Convolutional Neural Network (CNN) to recognize digits drawn by users or uploaded as images. It supports single-digit, double-digit, and multi-digit recognition.

## 🚀 Live Demo

[[**Try DigitAI**](PASTE_YOUR_RENDER_URL_HERE)](https://handwritten-digit-cnn.onrender.com)


## Features

- Recognizes handwritten digits from 0–9.
- Supports single-digit, double-digit, and multi-digit input.
- Allows users to draw digits using a mouse, touch screen, or stylus.
- Supports image uploads in PNG, JPG, and JPEG formats.
- Displays predictions and confidence scores.
- Preprocesses images for CNN-based recognition.

## Technology Stack

- **Frontend:** React, Vite, Bootstrap, Tailwind CSS
- **Backend:** Node.js, Express, Python, Flask
- **Machine Learning:** TensorFlow/Keras, CNN
- **Image Processing:** OpenCV
- **Language:** TypeScript, JavaScript, Python

## Project Structure

```text
HANDWRITTEN_DIGIT_CNN/
├── backend/       # Flask API and image preprocessing
├── models/        # Saved CNN model and exported weights
├── src/           # React frontend
├── static/        # CSS, JavaScript, plots, and sample images
├── templates/     # Flask web interface
├── tests/         # Automated tests
├── training/      # CNN training and evaluation scripts
├── package.json
├── requirements.txt
├── server.ts
└── README.md
```

## Run Locally

### Requirements

- Node.js and npm
- Python 3
- Git

### Start the application

Clone the repository and open its folder:

```bash
git clone https://github.com/Divyasreeperuri/HANDWRITTEN_DIGIT_CNN.git
cd HANDWRITTEN_DIGIT_CNN
```

Install the Node.js dependencies:

```bash
npm install
```

Start the application:

```bash
npm run dev
```

Open the local URL displayed in the terminal.

### Optional: Run the Flask backend

Create and activate a Python virtual environment, then install the requirements:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Start Flask:

```bash
python backend/app.py
```

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `/api/health` | Checks backend health |
| `/api/model-info` | Returns model information |
| `/api/predict` | Predicts digits from input |
| `/api/samples` | Provides sample images |
| `/api/evaluate` | Runs model evaluation |

## Machine Learning

DigitAI uses a CNN trained for handwritten digit recognition. Image preprocessing and segmentation help prepare input images for prediction. The application can recognize individual digits and sequences of digits.

## Author

**Divyasreeperuri**

GitHub: [@Divyasreeperuri](https://github.com/Divyasreeperuri)

