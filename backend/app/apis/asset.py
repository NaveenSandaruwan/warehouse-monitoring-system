from flask import Blueprint, jsonify, request
from app.database import assets_collection
from bson.objectid import ObjectId

# Create a Blueprint for the asset API
asset = Blueprint('asset', __name__)

# Route: Add an asset
@asset.route("/assets", methods=["POST"])
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
@asset.route("/assets", methods=["GET"])
def get_assets():
    assets = list(assets_collection.find())
    for asset in assets:
        asset["_id"] = str(asset["_id"])  # Convert ObjectId to string
    return jsonify(assets)

# Route: Update an asset
@asset.route("/assets/<asset_id>", methods=["PUT"])
def update_asset(asset_id):
    updates = request.json
    assets_collection.update_one({"_id": ObjectId(asset_id)}, {"$set": updates})
    return jsonify({"message": f"Asset {asset_id} updated successfully!"})

# Route: Delete an asset
@asset.route("/assets/<asset_id>", methods=["DELETE"])
def delete_asset(asset_id):
    assets_collection.delete_one({"_id": ObjectId(asset_id)})
    return jsonify({"message": f"Asset {asset_id} deleted successfully!"})
