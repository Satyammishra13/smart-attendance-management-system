# Smart-attendance-management-system


🚀 Key Features
Automated Face Recognition: Real-time face detection and identification via webcam or IP camera feed.

Instant Attendance Logging: Automatically logs attendance with precise timestamps into a local database/CSV file.

Duplicate Prevention: Prevents multiple log entries for the same individual within a configured time window.

Admin Dashboard: Interactive GUI/Web portal to manage student/employee records, view logs, and export reports.

Data Export: Generate and download daily/monthly attendance reports in Excel (.xlsx) or CSV formats.


🛠️ Tech Stack
Language: Python 3.x

Computer Vision: OpenCV, face_recognition (dlib)

GUI / Dashboard: Streamlit / Flask / Tkinter

Database: SQLite / MySQL / Pandas (CSV handling)


├── dataset/                 # Stored images of registered users
│   ├── User_1/
│   └── User_2/
├── models/                  # Pre-trained models and face encoding files
├── Attendance.csv           # Log file storing attendance records
└── README.md                # Project documentation
