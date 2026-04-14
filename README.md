# 🫁 Domain-Adaptive Synthetic Data Generation for Rare Disease Chest X-Ray Classification Using Conditional Diffusion Models

**Explainability & Deployment Developer**

A production-ready Streamlit application for chest X-ray disease classification with Grad-CAM explainability, user authentication, and comprehensive admin analytics.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Deployment](#deployment)
- [Database](#database)
- [Logging](#logging)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🎯 Overview

This application provides a complete end-to-end solution for:

1. **Medical Image Classification** - Classify chest X-rays into 3 disease categories using EfficientNet-B0
2. **Explainability** - Visualize model decisions using Grad-CAM heatmaps
3. **User Management** - Secure authentication with role-based access control
4. **Analytics** - Comprehensive prediction logging and admin dashboard
5. **Production Deployment** - Ready for cloud deployment (AWS, GCP, Heroku)

**Supported Disease Classes:**
- Normal
- Pleural Effusion
- Cardiomegaly

---

## ✨ Features

### 🔐 **Authentication & Security**
- User signup with email verification
- Secure password hashing (PBKDF2-HMAC-SHA256)
- Session-based authentication
- Role-based access control (User/Admin)
- Audit logging of all activities

### 🫁 **Disease Classification**
- EfficientNet-B0 deep learning model
- GPU/CPU inference support
- Real-time prediction with confidence scores
- Batch prediction capability
- Model caching with Streamlit `@st.cache_resource`

### 🔍 **Explainability (Grad-CAM)**
- Visual explanation of model predictions
- Gradient-weighted class activation maps
- Multiple colormap options (jet, hot, cool, plasma, etc.)
- Heatmap overlay on original images
- Region highlighting for medical interpretation

### 📊 **User Dashboard**
- Image upload interface with preview
- Automatic image preprocessing (224×224 normalization)
- Real-time prediction results
- Probability distribution charts
- Grad-CAM visualization (side-by-side comparison)
- Download prediction reports as TXT
- Prediction history and statistics

### ⚙️ **Admin Dashboard**
- System statistics and metrics
- Prediction logs viewer (searchable/filterable)
- Export reports (CSV, Excel, JSON)
- User activity monitoring
- Model status and configuration display
- System workflow documentation

### 💾 **Data Logging**
- CSV-based prediction logging
- User activity audit trail
- Detailed error logging
- Rotating file handlers to manage log size
- Export capabilities for reporting

---

## 🛠 Technical Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Framework** | Streamlit | 1.42.1 |
| **Deep Learning** | PyTorch | 2.1.2 |
| **Model** | EfficientNet-B0 | torchvision 0.16.2 |
| **Image Processing** | OpenCV, Pillow | 4.8.1 / 10.1.0 |
| **Data Processing** | Pandas, NumPy | 2.1.4 / 1.26.3 |
| **Database** | SQLite | - |
| **Visualization** | Matplotlib, Plotly | 3.8.3 / 5.18.0 |
| **Authentication** | bcrypt | 4.1.1 |
| **Python** | Python | 3.11+ |

---

## 📁 Project Structure

```
disease project/
├── app.py                          # Main Streamlit application
├── main.py                         # Alternative entry point
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create)
│
├── app/                           # Application package
│   ├── __init__.py
│   ├── config/                    # Configuration
│   │   ├── __init__.py
│   │   └── settings.py            # App settings and config
│   │
│   ├── auth/                      # Authentication module
│   │   ├── __init__.py
│   │   └── auth.py                # Login/signup logic
│   │
│   ├── utils/                     # Utility modules
│   │   ├── __init__.py
│   │   ├── auth.py                # Auth handler (JSON-based)
│   │   ├── preprocess.py          # Image preprocessing
│   │   ├── predict.py             # Model inference
│   │   ├── gradcam.py             # Grad-CAM explainability
│   │   └── logger.py              # Logging utilities
│   │
│   ├── database/                  # Database layer
│   │   ├── __init__.py
│   │   ├── db_handler.py          # Database operations
│   │   └── users.json             # User storage (auto-created)
│   │
│   ├── models/                    # Model modules
│   │   └── __init__.py
│   │
│   ├── components/                # Reusable UI components
│   │   └── __init__.py
│   │
│   ├── pages/                     # Page modules
│   │   ├── __init__.py
│   │   ├── auth_page.py           # Auth UI
│   │   ├── dashboard.py           # User dashboard
│   │   ├── prediction.py          # Prediction page
│   │   ├── explainability.py      # Grad-CAM page
│   │   ├── admin.py               # Admin dashboard
│   │   ├── logs_viewer.py         # Logs viewer
│   │   └── settings.py            # Settings page
│   │
│   ├── logs/                      # Application logs
│   │   ├── app.log                # General application logs
│   │   ├── predictions.csv        # Prediction records (auto-created)
│   │   └── audit.log              # User activity audit
│   │
│   └── assets/                    # Static assets
│       ├── css/
│       ├── images/
│       └── config/
│
├── pretrained_models/             # Model storage
│   └── efficientnet_model.pth     # EfficientNet checkpoint
│
├── data/                          # Data directory
│   └── uploads/                   # Uploaded images (temporary)
│
├── docs/                          # Documentation
│   ├── SETUP.md                   # Setup guide
│   ├── API.md                     # API documentation
│   └── WORKFLOW.md                # System workflow
│
├── tests/                         # Unit tests
│   └── test_*.py
│
└── .gitignore                     # Git ignore rules
```

---

## 📦 Installation

### Prerequisites

- **Python 3.11+** (installed and in PATH)
- **pip** (Python package manager)
- **Virtual Environment** (venv or conda)
- **Git** (for cloning)

### Step 1: Clone Repository

```bash
cd "c:\Users\ddevi\OneDrive\Desktop"
git clone <repository-url>
cd "disease project"
```

### Step 2: Create Virtual Environment

```bash
# Using venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Or using conda
conda create -n disease_classifier python=3.11
conda activate disease_classifier
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key packages installed:**
- streamlit==1.42.1
- torch==2.1.2
- torchvision==0.16.2
- pillow==10.1.0
- opencv-python==4.8.1.78
- pandas==2.1.4
- scikit-learn==1.3.2
- matplotlib==3.8.3

### Step 4: Download Pre-trained Model

Place your trained EfficientNet model at:

```
pretrained_models/efficientnet_model.pth
```

**Note:** If model is missing, the app will show a warning and use dummy predictions for demonstration.

### Step 5: Create Environment File (Optional)

Create `.env` in project root:

```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
CUDA_VISIBLE_DEVICES=0
MODEL_PATH=pretrained_models/efficientnet_model.pth
DATABASE_URL=sqlite:///app/database/app.db
```

---

## 🚀 Quick Start

### Run Locally (Development)

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run the app
streamlit run app.py
```

The application will:
1. Open in your default browser at `http://localhost:8501`
2. Create necessary directories and database files
3. Initialize logging system
4. Load cached models

### Access the Application

- **URL:** http://localhost:8501
- **Default Test Credentials:**
  - Username: `admin` (create one on first run if needed)
  - Password: Can be set during signup

---

## 📖 Usage Guide

### 1. **User Registration**

```
1. Click "Sign Up" tab
2. Enter:
   - Username (min 3 chars)
   - Email (valid format)
   - Password (min 6 chars)
3. Click "Sign Up" button
```

### 2. **User Login**

```
1. Click "Login" tab
2. Enter username and password
3. Click "Login" button
4. Access dashboard
```

### 3. **Make Predictions**

```
1. Navigate to "User Dashboard"
2. Upload chest X-ray image (JPG/PNG)
3. View:
   - Original image preview
   - Preprocessing status
   - Predicted disease class
   - Confidence score (0-100%)
   - Probability for all classes
4. (Optional) Generate Grad-CAM visualization
5. Download report as TXT file
6. Save prediction to logs
```

### 4. **View Grad-CAM Explainability**

```
1. After making prediction
2. Check "Generate Grad-CAM Heatmap"
3. View side-by-side comparison:
   - Original image
   - Heatmap (regions important for prediction)
   - Overlay (heatmap + original)
```

### 5. **Access Admin Dashboard** (Admin role only)

**Statistics Tab:**
- Total predictions counter
- Unique users metric
- Average confidence score
- Disease distribution chart
- Device usage pie chart

**Predictions Logs Tab:**
- View all predictions as table
- Filter by user
- Export to CSV/Excel/JSON
- Search and sort capabilities

**System Info Tab:**
- Model configuration display
- Project workflow explanation
- Security features documentation

---

## 🌐 Deployment

### Heroku Deployment

1. **Create Heroku App:**
```bash
heroku login
heroku create your-app-name
```

2. **Create `Procfile`** (in project root):
```
web: streamlit run app.py --logger.level=info --server.port=$PORT --server.address=0.0.0.0
```

3. **Create `runtime.txt`** (in project root):
```
python-3.11.0
```

4. **Deploy:**
```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### AWS EC2 Deployment

1. **SSH into EC2 instance:**
```bash
ssh -i key.pem ec2-user@your-instance-ip
```

2. **Install dependencies:**
```bash
sudo yum update -y
sudo yum install python311 python311-pip -y
```

3. **Clone and setup:**
```bash
git clone <repo-url>
cd disease\ project
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. **Run with systemd:**
```bash
sudo tee /etc/systemd/system/cxr-classifier.service > /dev/null <<EOF
[Unit]
Description=CXR Disease Classifier
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/disease\ project
ExecStart=/home/ec2-user/disease\ project/.venv/bin/streamlit run app.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable cxr-classifier
sudo systemctl start cxr-classifier
```

### Docker Deployment

1. **Create `Dockerfile`:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. **Build and run:**
```bash
docker build -t cxr-classifier .
docker run -p 8501:8501 cxr-classifier
```

---

## 🗄️ Database

### User Storage (JSON)

Users are stored in: `app/database/users.json`

**Structure:**
```json
{
  "users": [
    {
      "username": "john_doe",
      "email": "john@example.com",
      "password": "<hashed>",
      "salt": "<salt>",
      "role": "user",
      "is_active": true,
      "created_at": "2025-04-14T10:30:00",
      "last_login": "2025-04-14T11:00:00"
    }
  ]
}
```

### SQLite Database (Optional)

If using SQLite instead, located at: `app/database/app.db`

**Tables:**
- `users` - User accounts and credentials
- `predictions` - Prediction records
- `activity_logs` - Audit trail
- `model_metrics` - Performance metrics

---

## 📊 Logging

### Log Files

1. **`app/logs/app.log`** - General application logs
2. **`app/logs/predictions.csv`** - Prediction records
3. **`app/logs/audit.log`** - User activity audit

### Prediction CSV Fields

```csv
timestamp,username,email,filename,predicted_class,confidence,device,gradcam_applied,model_version
2025-04-14T10:30:45.123456,john_doe,john@example.com,xray_001.jpg,Normal,0.950000,cuda,True,1.0
```

### Log Rotation

- Max file size: 10MB per log file
- Backup count: 5 rotating files
- Configure in `app/config/settings.py`

---

## ⚙️ Configuration

### Edit `app/config/settings.py`

```python
# Model Configuration
MODEL_CONFIG = {
    "model_path": "pretrained_models/diffusion_model.pt",
    "device": "cuda",  # or "cpu"
    "input_size": (224, 224),
    "num_classes": 4,
}

# Grad-CAM Configuration
GRADCAM_CONFIG = {
    "layer_name": "layer4",
    "colormap": "jet",
    "alpha": 0.5
}

# Authentication
AUTH_CONFIG = {
    "password_hash_algorithm": "bcrypt",
    "session_timeout_minutes": 60,
    "max_login_attempts": 5
}
```

---

## 🔧 Troubleshooting

### Issue: Model not found

**Solution:**
```bash
# Download/place model at:
mkdir -p pretrained_models
# Copy your model to pretrained_models/efficientnet_model.pth

# App will use dummy predictions as fallback
```

### Issue: CUDA out of memory

**Solution:**
```python
# In app/config/settings.py, change:
MODEL_CONFIG["device"] = "cpu"  # Use CPU instead
```

### Issue: Port 8501 already in use

**Solution:**
```bash
# Use different port:
streamlit run app.py --server.port 8502
```

### Issue: Permission denied (logs directory)

**Solution:**
```bash
# Fix permissions:
chmod -R 755 app/logs/
```

### Issue: CORS errors in deployment

**Add to `.streamlit/config.toml`:**
```toml
[client]
headerFont = "sans serif"
bodyFont = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true
```

---

## 📈 Performance Optimization

### 1. **Model Caching**
- Model loaded once with `@st.cache_resource`
- Reused across predictions
- GPU memory managed efficiently

### 2. **Image Preprocessing**
- Vectorized operations with NumPy
- GPU acceleration with PyTorch
- Memory-efficient batch processing

### 3. **Streamlit Optimization**
- Session state for authentication
- Lazy loading of components
- Conditional rendering

---

## 🔐 Security Features

- ✅ Password hashing (PBKDF2-HMAC-SHA256)
- ✅ Session-based authentication
- ✅ Role-based access control
- ✅ Audit logging of all activities
- ✅ Input validation and sanitization
- ✅ Secure file uploads
- ✅ HTTPS ready for deployment

---

## 📝 API Documentation

### Authentication Functions

```python
from app.utils.auth import signup, login, user_exists

# Signup
success, msg = signup("username", "email@example.com", "password123")

# Login
success, msg, user = login("username", "password123")

# Check user exists
exists = user_exists("username")
```

### Prediction Functions

```python
from app.utils.predict import predict_image, get_prediction_dict

# Single prediction
probs, class_idx, class_name = predict_image(tensor)

# Detailed result
result = get_prediction_dict(tensor)
# Returns: {predicted_class, class_name, confidence, probabilities}
```

### Image Preprocessing

```python
from app.utils.preprocess import load_and_preprocess

# Load and preprocess
tensor = load_and_preprocess("path/to/image.jpg")
```

### Grad-CAM Visualization

```python
from app.utils.gradcam import generate_gradcam

# Generate visualization
heatmap, overlaid = generate_gradcam(model, tensor, image)
```

### Logging

```python
from app.utils.logger import log_prediction_csv, read_prediction_logs

# Log prediction
log_prediction_csv("username", "email", "file.jpg", "Normal", 0.95)

# Read logs
df = read_prediction_logs()
```

---

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit changes** (`git commit -m 'Add amazing feature'`)
4. **Push to branch** (`git push origin feature/amazing-feature`)
5. **Open Pull Request**

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👨‍💼 Authors

**Explainability & Deployment Developer**

---

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: your-email@example.com
- Check troubleshooting section above

---

## 🙏 Acknowledgments

- PyTorch and TorchVision teams for model architecture
- Streamlit team for fantastic framework
- Medical imaging research community

---

## 📅 Changelog

### Version 1.0.0 (2025-04-14)
- ✨ Initial release
- 🔐 User authentication
- 🫁 Disease classification
- 🔍 Grad-CAM explainability
- 📊 Admin dashboard
- 💾 CSV prediction logging

---

**Last Updated:** April 14, 2025

**Status:** ✅ Production Ready

