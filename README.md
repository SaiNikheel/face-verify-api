# Face Recognition Attendance System

A Python-based REST API for face recognition and attendance tracking using FastAPI and InsightFace.

## Features

- Face registration with employee ID
- Face verification against registered faces
- Local storage of face embeddings and images
- RESTful API endpoints
- Swagger UI documentation

## Prerequisites

- Python 3.10+
- OpenCV
- InsightFace
- FastAPI
- Uvicorn
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face_recognition_attendance.git
cd face_recognition_attendance
```

2. Create and activate a virtual environment:
```bash
python -m venv local
source local/bin/activate  # On Windows: local\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
uvicorn main:app --host localhost --port 8081 --reload
```

2. Access the API documentation:
- Open your browser and go to `http://localhost:8081/docs`

## API Endpoints

### Register Face
- **POST** `/register`
- **Input**: JSON with `employee_id` and base64-encoded `image`
- **Output**: Success message with employee ID

### Verify Face
- **POST** `/verify`
- **Input**: JSON with base64-encoded `image`
- **Output**: Verification result with employee ID and similarity score

### Health Check
- **GET** `/health`
- **Output**: API health status

## Project Structure

```
face_recognition_attendance/
│
├── main.py            # FastAPI app with endpoints
├── face_engine.py     # Face detection and embedding
├── storage.py         # Storage management
├── config.py          # Configuration settings
├── requirements.txt   # Dependencies
└── storage/          # Local storage directory
    ├── embeddings.pkl
    └── images/
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 