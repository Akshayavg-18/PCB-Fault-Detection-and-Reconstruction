# Generative AI Based PCB Fault Detection and Reconstruction

## 📌 Project Overview

**Generative AI Based Analysis of Fault Behaviour in Electronic Circuit (PCB) and Reconstruction Guidance** is a Python and Streamlit-based application that uses multimodal Generative AI and image processing techniques to analyze PCB images, identify possible faults, and provide guidance for reconstructing the board into a healthy state.

The system accepts a PCB image as input and performs image preprocessing using OpenCV before sending the image for AI-based fault analysis. The application generates a structured fault report, reconstruction/repair guidance, and an AI-assisted reconstructed PCB image.

---

## 🎯 Objectives

- Analyze PCB images for visible faults and defects.
- Identify defective components, traces, solder joints, and surface damage.
- Provide structured fault information with severity and confidence.
- Generate step-by-step reconstruction and repair guidance.
- Generate a reference description of a healthy PCB.
- Produce an AI-assisted reconstructed PCB image.
- Provide a local OpenCV-based fallback reconstruction when AI image generation is unavailable.
- Provide an easy-to-use web interface through Streamlit.

---

## ✨ Key Features

### 🔍 PCB Fault Detection

The application analyzes uploaded PCB images and provides:

- PASS / FAIL assessment
- Confidence score
- Detected defects
- Defect severity
- Defect location
- Component-wise analysis
- PCB quality score

### 🛠️ Reconstruction Guidance

The system generates structured reconstruction guidance including:

- Repair/reconstruction steps
- Area requiring attention
- Recommended action
- Expected result
- Tools required
- Estimated repair difficulty
- Healthy-board description
- Overall reconstruction summary

### 🖼️ Reconstructed PCB Image

The application attempts to generate an AI-assisted visualization of the reconstructed healthy PCB.

When AI image generation is unavailable because of quota or service limitations, the application can use an OpenCV-based local reconstruction approach as a fallback.

### 🖥️ Interactive Dashboard

The Streamlit interface provides three main sections:

- **Dashboard**
- **Fault Detection**
- **Reconstruction Guide**

---

## 🔄 System Workflow

PCB Image Upload
       ↓
File Validation
       ↓
OpenCV Image Preprocessing
       ↓
AI-Based Fault Detection
       ↓
Fault Analysis & Structured Report
       ↓
Reconstruction Guide Generation
       ↓
AI / OpenCV Reconstruction
       ↓
Dashboard Display
       ↓
Reconstructed Image Download


🧠 Technology Stack
Technology	Purpose
Python	Application development
Streamlit	Interactive web dashboard
OpenCV	Image processing and reconstruction fallback
Pillow	Image handling
NumPy	Numerical/image operations
python-dotenv	Environment variable management
Google Generative AI SDK	AI-based PCB analysis
Google GenAI SDK	Generative AI functionality

File Description                  File	Description
app.py	                 Main Streamlit application and dashboard control flow
detector.py	             PCB fault detection prompts and response normalization
reconstructor.py    	    Reconstruction and repair guidance generation
gemini_client.py          Google Generative AI API communication and response handling
image_utils.py          	Image validation, preprocessing, enhancement and local reconstruction
ui_components.py	        Reusable Streamlit UI components
requirements.txt	        Python dependencies
.env.example	            Template for required environment variables
start_dashboard.bat      	Windows batch launcher
start_dashboard.ps1	      PowerShell launcher

Requirements
Hardware
- Laptop or desktop computer
- Minimum 4 GB RAM recommended
- Internet connection for Generative AI API access
- PCB images/dataset or camera-captured PCB images
Software
- Python 3.10 or later
- Modern web browser
- Google Generative AI API access

Installation and Setup
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/PCB-Fault-Detection-and-Reconstruction.git
Move into the project directory:
cd PCB-Fault-Detection-and-Reconstruction
2. Create a Virtual Environment
python -m venv venv
Activate it on Windows:
venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Configure the API Key
Create a .env file in the project root.
GEMINI_API_KEY=your_actual_api_key
Never upload the .env file to GitHub.
The repository includes .env.example as a safe template:
GEMINI_API_KEY=your_key_here
5. Run the Application
Start the Streamlit dashboard using:
streamlit run app.py
The application will normally be available at:
http://localhost:8501
🧪 How to Use
Step 1 — Upload a PCB Image
Upload a PCB image in:
- JPG
- PNG
The application validates and preprocesses the image before analysis.
Step 2 — Fault Detection
Run the fault detection process.
The system analyzes the PCB image and provides information about possible faults, including their severity and location.
Step 3 — Review Fault Analysis
Review the generated:
- Fault status
- Confidence
- Defect list
- Severity
- Component analysis
- Quality score
Step 4 — Generate Reconstruction Guidance
Generate the reconstruction guide to obtain step-by-step repair recommendations.
Step 5 — Generate Reconstruction Image
The application attempts to create a visualization of the expected healthy PCB.
Step 6 — Download the Result
The reconstructed image can be downloaded from the application.
🖼️ Image Preprocessing
Before AI analysis, the uploaded PCB image can undergo preprocessing using OpenCV.
The preprocessing pipeline includes:
- File validation
- Image resizing
- RGB conversion
- Optional contrast enhancement
- CLAHE-based image enhancement
The application supports images up to the configured upload limit and resizes large images to improve processing efficiency.
📊 AI Output
The fault analysis produces structured information such as:
Status
Confidence
Defects
Severity
Location
Component Analysis
Quality Score
The reconstruction module produces structured information including:
Board Type
Reconstruction Steps
Healthy Board Description
Reference Image Prompt
Estimated Repair Difficulty
Summary
🛠️ Reconstruction Approach
The application uses two approaches for reconstruction visualization:
1. AI-Assisted Reconstruction
Generative AI is used to create a reference visualization of how the PCB may appear after restoration.
2. OpenCV-Based Fallback
If AI image generation is unavailable, the application can perform an approximate local reconstruction using image-processing techniques such as:
- Damaged-region detection
- Donor patch selection
- Image blending
- Inpainting
The fallback is intended as an approximate visual reconstruction rather than an exact physical PCB restoration.
⚠️ Limitations
The system has several practical limitations:
- Analysis accuracy depends on PCB image quality and lighting.
- Very small defects may not always be detected.
- AI-generated results require human verification.
- AI image generation may depend on API availability and quota.
- OpenCV-based reconstruction is approximate.
- The system does not perform physical electrical continuity testing.
- Visual reconstruction should not be treated as a manufacturing-ready PCB design.
🔮 Future Scope
Possible improvements include:
- Custom PCB defect detection models
- Automatic defect bounding boxes
- Larger and specialized PCB defect datasets
- Dedicated PCB restoration models
- Electrical continuity and circuit-level testing
- Batch PCB inspection
- Automated PDF inspection reports
- Camera-based PCB inspection
- Local network deployment
- Role-based access control
🔐 Security
API credentials should always be stored using environment variables.
Do not commit:
.env
to the GitHub repository.
Use:
.env.example
to show the required environment variable format without exposing credentials.
⚠️ Disclaimer
This application provides AI-assisted visual analysis and reconstruction guidance.
AI-generated fault analysis and reconstruction results are advisory and should be verified by a qualified electronics engineer or technician before any physical PCB repair, modification, or manufacturing decision.
👩‍💻 Project
Project: Generative AI Based Analysis of Fault Behaviour in Electronic Circuit (PCB) and Reconstruction Guidance
Platform: Python + Streamlit
Domain: Electronics / PCB Inspection / Computer Vision / Generative AI 
