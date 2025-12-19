from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
import os, datetime, urllib.parse
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import clip

# -------------------------
# 🔹 Flask Setup
# -------------------------
app = Flask(__name__, static_folder='frontend/static')
CORS(app)
bcrypt = Bcrypt(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------
# 🔹 MongoDB Atlas
# -------------------------
username_atlas = "plant_user_18"
password_atlas = urllib.parse.quote_plus("Test1234!Test")
client = MongoClient(
    f"mongodb+srv://{username_atlas}:{password_atlas}@cluster0.i5lhstg.mongodb.net/?retryWrites=true&w=majority"
)
db = client["cropcare_ai"]
users = db["users"]
results = db["results"]

# -------------------------
# 🔹 CLIP MODEL (Multimodal Core)
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

DISEASES = [
    {
        "crop": "Tomato",
        "disease": "Early Blight",
        "scientific": "Alternaria solani",
        "prompt": "tomato leaf with early blight disease",
        "symptoms": [
            "yellow spots on leaves",
            "brown concentric rings",
            "leaf drying and falling"
        ],
        "precautions": [
            "use disease-free seeds",
            "avoid overhead irrigation",
            "apply copper-based fungicide"
        ]
    },
    {
        "crop": "Tomato",
        "disease": "Healthy",
        "scientific": "Solanum lycopersicum",
        "prompt": "healthy tomato leaf",
        "symptoms": [
            "green leaves",
            "no spots or discoloration"
        ],
        "precautions": [
            "maintain proper watering",
            "ensure balanced fertilization"
        ]
    },

    {
        "crop": "Potato",
        "disease": "Late Blight",
        "scientific": "Phytophthora infestans",
        "prompt": "potato leaf with late blight disease",
        "symptoms": [
            "dark water-soaked lesions",
            "white fungal growth under leaf",
            "rapid leaf wilting"
        ],
        "precautions": [
            "remove infected plants",
            "use resistant varieties",
            "apply fungicides like mancozeb"
        ]
    },
    {
        "crop": "Potato",
        "disease": "Healthy",
        "scientific": "Solanum tuberosum",
        "prompt": "healthy potato leaf",
        "symptoms": [
            "uniform green color",
            "no visible lesions"
        ],
        "precautions": [
            "proper spacing",
            "regular field monitoring"
        ]
    },

    {
        "crop": "Banana",
        "disease": "Black Sigatoka",
        "scientific": "Mycosphaerella fijiensis",
        "prompt": "banana leaf with black sigatoka disease",
        "symptoms": [
            "dark streaks on leaves",
            "yellowing of leaf margins",
            "reduced photosynthesis"
        ],
        "precautions": [
            "remove infected leaves",
            "ensure proper air circulation",
            "apply systemic fungicides"
        ]
    },
    {
        "crop": "Banana",
        "disease": "Healthy",
        "scientific": "Musa species",
        "prompt": "healthy banana leaf",
        "symptoms": [
            "broad green leaves",
            "no dark streaks"
        ],
        "precautions": [
            "regular pruning",
            "balanced nutrient supply"
        ]
    },

    {
        "crop": "Apple",
        "disease": "Apple Scab",
        "scientific": "Venturia inaequalis",
        "prompt": "apple leaf with apple scab disease",
        "symptoms": [
            "olive green spots",
            "leaf curling",
            "premature leaf drop"
        ],
        "precautions": [
            "remove fallen leaves",
            "use resistant cultivars",
            "apply sulfur fungicides"
        ]
    },
    {
        "crop": "Apple",
        "disease": "Healthy",
        "scientific": "Malus domestica",
        "prompt": "healthy apple leaf",
        "symptoms": [
            "smooth green leaves",
            "no spots"
        ],
        "precautions": [
            "regular pruning",
            "adequate sunlight exposure"
        ]
    },

    {
        "crop": "Rice",
        "disease": "Blast Disease",
        "scientific": "Magnaporthe oryzae",
        "prompt": "rice leaf with blast disease",
        "symptoms": [
            "diamond-shaped lesions",
            "gray center with brown margin",
            "leaf drying"
        ],
        "precautions": [
            "avoid excess nitrogen fertilizer",
            "ensure proper field drainage",
            "use blast-resistant varieties"
        ]
    },
    {
        "crop": "Rice",
        "disease": "Healthy",
        "scientific": "Oryza sativa",
        "prompt": "healthy rice leaf",
        "symptoms": [
            "long green leaves",
            "no lesions"
        ],
        "precautions": [
            "maintain optimal water levels",
            "apply balanced fertilizer"
        ]
    }
]



with torch.no_grad():
    text_tokens = clip.tokenize([d["prompt"] for d in DISEASES]).to(device)
    TEXT_FEATURES = clip_model.encode_text(text_tokens)
    TEXT_FEATURES /= TEXT_FEATURES.norm(dim=-1, keepdim=True)

def clip_predict(image_path):
    image = clip_preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = clip_model.encode_image(image)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        similarity = (img_feat @ TEXT_FEATURES.T).squeeze(0)
        idx = similarity.argmax().item()
        confidence = round(float(similarity[idx]) * 100, 2)
    return DISEASES[idx], confidence

# -------------------------
# 🔹 Serve HTML Pages
# -------------------------
@app.route("/")
def home():
    return send_from_directory("frontend", "home.html")

@app.route("/login")
def login_page():
    return send_from_directory("frontend", "login.html")

@app.route("/register")
def register_page():
    return send_from_directory("frontend", "registration.html")

@app.route("/index")
def index_page():
    return send_from_directory("frontend", "index.html")

@app.route("/forgot_password")
def forgot_page():
    return send_from_directory("frontend", "forgot_password.html")

@app.route("/pastresults")
def past_page():
    return send_from_directory("frontend", "pastresults.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("frontend", path)

# 🔹 Serve uploaded images (for past results)
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# -------------------------
# 🔹 Auth APIs
# -------------------------
@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    if users.find_one({"username": data["username"]}):
        return jsonify({"message": "Username already exists"}), 409
    users.insert_one({
        "username": data["username"],
        "password": bcrypt.generate_password_hash(data["password"]).decode(),
        "createdAt": datetime.datetime.utcnow()
    })
    return jsonify({"message": "Registered successfully"}), 201

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    user = users.find_one({"username": data["username"]})
    if not user or not bcrypt.check_password_hash(user["password"], data["password"]):
        return jsonify({"message": "Invalid credentials"}), 401
    return jsonify({"message": "Login successful", "user": {"username": data["username"]}})



# -------------------------
# 🔹 Prediction API
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    username = request.form.get("username", "guest")
    question = request.form.get("question", "").lower()

    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "Image required"}), 400

    filename = secure_filename(image_file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_file.save(path)

    prediction, confidence = clip_predict(path)

    cddm = {
        "image_id": filename,
        "crop": prediction["crop"],
        "disease_name": prediction["disease"],
        "scientific_name": prediction["scientific"],
        "symptoms": ["Detected using CLIP multimodal inference"],
        "causes": ["Vision–language similarity matching"],
        "solutions": {
            "cultural": ["Crop rotation", "Remove infected leaves"],
            "biological": ["Bio-fungicide"],
            "chemical": ["Apply Mancozeb if severe"]
        },
        "prevention_summary": "AI-based multimodal crop disease diagnosis"
    }

    results.insert_one({
        "user": username,
        "filename": filename,
        "prediction": {
            "disease": cddm["disease_name"],
            "confidence": confidence
        },
        "record": cddm,
        "createdAt": datetime.datetime.utcnow()
    })

    # Do NOT delete the file; needed for past results image display.
    # os.remove(path)

    return jsonify({**cddm, "confidence": confidence}), 200

# -------------------------
# 🔹 Past Results
# -------------------------
@app.route("/api/past-results", methods=["GET"])
def past_results():
    username = request.args.get("username")
    past = list(results.find({"user": username}).sort("createdAt", -1))
    for p in past:
        p["_id"] = str(p["_id"])
        p["createdAt"] = p["createdAt"].isoformat()
    return jsonify(past)

# -------------------------
# 🔹 Clear History
# -------------------------
@app.route("/api/clear-history", methods=["DELETE"])
def clear_history():
    username = request.args.get("username")
    deleted = results.delete_many({"user": username}).deleted_count
    return jsonify({"message": f"Deleted {deleted} records"})

# -------------------------
# 🔹 Debug
# -------------------------
@app.route("/api/debug-counts")
def debug_counts():
    pipeline = [{"$group": {"_id": "$user", "count": {"$sum": 1}}}]
    data = list(results.aggregate(pipeline))
    for d in data:
        d["user"] = d.pop("_id")
    return jsonify(data)

# -------------------------
# 🔹 Run App
# -------------------------
if __name__ == "__main__":
    print("Connected DB:", client.list_database_names())
    app.run(debug=True, port=5000)


