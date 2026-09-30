# IADAI201-2505467--Sohapatel
For your GitHub repository description, use:  **AI-powered parking space detection and occupancy analysis using YOLO and Streamlit.**
#  ParkVision AI

### AI-Powered Parking Space Detection and Occupancy Analysis

ParkVision AI is an AI-powered parking space detection system that uses **YOLO-based computer vision** and **Streamlit** to identify parking spaces in an uploaded parking-lot image and classify them as **AVAILABLE** or **OCCUPIED**.

The system provides a visual parking analysis along with the total number of detected spaces, available spaces, occupied spaces, and availability information.

---

##  Project Objective

The main objective of ParkVision AI is to develop an intelligent parking analysis system that can automatically detect parking spaces from images and determine their occupancy status.

The system aims to reduce the need for manual parking-space monitoring and provide a clear visual representation of parking availability.

---

##  How ParkVision AI Works

The system follows these main steps:

1. **Upload an image** of a parking area.
2. The image is processed by the trained **YOLO model**.
3. Parking spaces are detected across the image.
4. Each detected parking space is classified as:

   * 🟢 **AVAILABLE** – parking space is empty.
   * 🔴 **OCCUPIED** – parking space contains a vehicle.
5. The system displays the detected parking spaces with colour-coded bounding boxes.
6. Parking statistics are calculated and displayed.
7. The results are presented through an interactive **Streamlit web application**.

---

##  Key Features

*  Parking-space detection
*  Available-space identification
*  Occupied-space identification
*  Parking analytics
*  Availability statistics
*  Parking map visualization
*  Image upload and AI analysis
*  YOLO-based computer vision
*  Interactive Streamlit interface

---

##  Artificial Intelligence Model

ParkVision AI uses a **YOLO object-detection model** trained specifically for parking-space detection.

### Classes

| Class            | Meaning                 |
| ---------------- | ----------------------- |
| `space-empty`    | Available parking space |
| `space-occupied` | Occupied parking space  |

The trained model used by the application is:

```text
best.pt
```

---

##  Dataset

The project uses the **PKLot parking dataset**, which contains images of parking areas used for parking-space detection and occupancy analysis.

The dataset was converted into a format suitable for YOLO-based training.

The project includes dataset configuration and preprocessing-related files such as:

```text
data.yaml
convert_coco.py
README.dataset.txt
README.roboflow.txt
```

The complete image dataset is not included in this repository because of its large size.

---

##  Technologies Used
* **Python**
* **YOLO**
* **Ultralytics**
* **Streamlit**
* **OpenCV**
* **NumPy**
* **Pandas**
* **Pillow**

---

##  Project Structure

```text
ParkVision_AI/
│
├── app.py
├── best.pt
├── convert_coco.py
├── data.yaml
├── requirements.txt
├── README.md
├── README.dataset.txt
├── README.roboflow.txt
└── .gitignore
```

### Important files

**`app.py`**
Main Streamlit application containing the user interface and parking detection logic.

**`best.pt`**
Trained YOLO model used for parking-space detection.

**`data.yaml`**
YOLO dataset configuration containing the dataset paths and class names.

**`convert_coco.py`**
Script used for dataset conversion/preparation.

**`requirements.txt`**
Python packages required to run the application.

---

##  Installation

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

### 2. Open the project folder

```bash
cd ParkVision_AI
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

#### Windows

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your web browser.

Upload a parking-lot image and ParkVision AI will analyse the image and display the detected parking spaces.

---

##  Output

The application provides:

* Total detected parking spaces
* Available parking spaces
* Occupied parking spaces
* Availability percentage
* Annotated parking-lot image
* Parking-space map
* Visual analytics

The detected spaces are displayed using:

🟢 **Green boxes → Available**

🔴 **Red boxes → Occupied**

---

##  Testing

The application was tested using parking-lot images that were not directly used during the development process.

Testing focused on:

* Detection of parking spaces
* Classification of empty and occupied spaces
* Detection across different parts of the image
* Visual correctness of bounding boxes
* Accuracy of parking statistics
* Streamlit application functionality

---

##  Research and Development

The development process involved:

1. Understanding the parking-space detection problem.
2. Collecting and preparing parking-lot image data.
3. Converting the dataset into YOLO-compatible format.
4. Training a YOLO-based object-detection model.
5. Testing the trained model.
6. Developing the Streamlit interface.
7. Implementing parking analytics.
8. Testing the complete system using unseen parking images.

---

##  Future Improvements

Possible future improvements include:

* Real-time parking detection using CCTV cameras
* Live parking availability monitoring
* Automatic parking-space recommendations
* Multi-camera parking analysis
* Improved detection under difficult lighting and weather conditions
* Integration with smart-city parking systems
* Mobile application integration

---

##  Project

**Project Name:** ParkVision AI
**Course:** IB Career-related Programme – Artificial Intelligence
**School:** Udgam School for Children
**Developer:** Soha Patel

---

##  Project Status

**Status:** Completed 

ParkVision AI currently provides an interactive AI-powered system for detecting and analysing parking-space occupancy from parking-lot images.

---

##  License

This project was developed as an academic project for educational purposes.
