from flask import Flask, jsonify, request
from app.database import connect_to_mongodb, assets_collection
from bson.objectid import ObjectId

app = Flask(__name__)

# Route: Home
@app.route("/")
def home():
    return jsonify({"message": "Warehouse Tracking Backend is running!"})

# Route: Add an asset
@app.route("/assets", methods=["POST"])
def add_asset():
    data = request.json
    asset_data = {
        "_id": str(ObjectId()),  # Generate a unique MongoDB ObjectId
        "name": data["name"],
        "type": data["type"],  # "person", "forklift", etc.
        "location_id": data["location_id"],
        "status": data["status"],
        "last_updated": data["last_updated"],
    }
    assets_collection.insert_one(asset_data)
    return jsonify({"message": "Asset added successfully!", "asset": asset_data})

# Route: Get all assets
@app.route("/assets", methods=["GET"])
def get_assets():
    assets = list(assets_collection.find())
    for asset in assets:
        asset["_id"] = str(asset["_id"])  # Convert ObjectId to string
    return jsonify(assets)

if __name__ == "__main__":
    app.run(debug=True)