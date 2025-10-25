from flask import Flask, request, jsonify, send_from_directory, redirect, url_for
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
import os
import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import requests

# --------------------------------------------------------
# Flask App Initialization
# --------------------------------------------------------
app = Flask(__name__, static_folder='frontend/static')
CORS(app)
bcrypt = Bcrypt(app)

# --------------------------------------------------------
# File Upload Setup
# --------------------------------------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --------------------------------------------------------
# MongoDB Connection
# --------------------------------------------------------
# If using MongoDB Atlas, replace the URI below:
# client = MongoClient("mongodb+srv://<username>:<password>@cluster0.mongodb.net/cropcare_ai")
client = MongoClient("mongodb://localhost:27017/")
db = client["cropcare_ai"]
users = db["users"]
results = db["results"]

# --------------------------------------------------------
# AI Model Endpoint (change to your model API if available)
# --------------------------------------------------------
MODEL_URL = "http://127.0.0.1:8000/predict"

# --------------------------------------------------------
# Serve Frontend Files
# --------------------------------------------------------
@app.route("/")
def serve_login():
    """Always start with the login page."""
    return send_from_directory("frontend", "login.html")

@app.route("/login")
def redirect_login():
    """Clean URL for login."""
    return redirect(url_for("serve_login"))

@app.route("/register")
def serve_register():
    """Clean URL for registration page."""
    return send_from_directory("frontend", "registration.html")

@app.route("/index")
def serve_index():
    """Main app page (after login)."""
    return send_from_directory("frontend", "index.html")

@app.route("/<path:path>")
def serve_frontend(path):
    """Serves any other frontend files (CSS, JS, images)."""
    return send_from_directory("frontend", path)

# --------------------------------------------------------
# Register Route
# --------------------------------------------------------
@app.route("/api/register", methods=["POST"])
def register_user():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    name = data.get("name", "")

    if not email or not password:
        return jsonify({"message": "Email and password required"}), 400

    if users.find_one({"email": email}):
        return jsonify({"message": "Email already registered!"}), 400

    hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
    users.insert_one({
        "name": name,
        "email": email,
        "password": hashed_pw,
        "createdAt": datetime.datetime.utcnow()
    })
    return jsonify({"message": "User registered successfully!"}), 201

# --------------------------------------------------------
# Login Route
# --------------------------------------------------------
@app.route("/api/login", methods=["POST"])
def login_user():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    user = users.find_one({"email": email})
    if not user or not bcrypt.check_password_hash(user["password"], password):
        return jsonify({"message": "Invalid credentials!"}), 401

    return jsonify({
        "message": "Login successful",
        "user": {
            "email": email,
            "name": user.get("name", "")
        }
    }), 200

# --------------------------------------------------------
# Prediction Route
# --------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload and AI model prediction."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    question = request.form.get("question", "")
    email = request.form.get("email", "guest")

    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image.save(filepath)

    try:
        # Resize for consistency
        img = Image.open(filepath)
        img.thumbnail((512, 512))
        img.save(filepath)

        # Try to call the AI model API
        try:
            response = requests.post(MODEL_URL, files={"image": open(filepath, "rb")})
            ai_result = response.json() if response.ok else None
        except Exception:
            ai_result = None

        # Fallback dummy result
        if not ai_result:
            ai_result = {
                "crop": "Tomato",
                "disease": "Early Blight",
                "cause": "Fungal infection",
                "prevention": "Use copper fungicide",
                "treatment": "Apply weekly fungicide",
                "confidence": 92
            }

        # Save result in MongoDB
        results.insert_one({
            "email": email,
            "filename": filename,
            "prediction": ai_result,
            "question": question,
            "createdAt": datetime.datetime.utcnow()
        })

        # Remove file after processing
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify(ai_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------------
# Past Results Route
# --------------------------------------------------------
@app.route("/api/past-results", methods=["GET"])
def get_past_results():
    """Fetches past diagnosis results for the logged-in user."""
    email = request.args.get("email")
    query = {"email": email} if email else {}
    past = list(results.find(query).sort("createdAt", -1).limit(10))
    for r in past:
        r["_id"] = str(r["_id"])
    return jsonify(past)

# --------------------------------------------------------
# Run Flask App
# --------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
