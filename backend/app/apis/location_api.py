from flask import Blueprint, jsonify, request
from app.database import locations_collection
from bson.objectid import ObjectId

# Create a Blueprint for the location API
location_api = Blueprint('location_api', __name__)

# Route: Add a new location
@location_api.route("/locations", methods=["POST"])
def add_location():
    try:
        data = request.json
        location_data = {
            "_id": str(ObjectId()),  # Generate a unique MongoDB ObjectId
            "id": data["id"],
            "name": data["name"],
            "coordinates": data["coordinates"],
            "type": data["type"]
        }
        locations_collection.insert_one(location_data)
        return jsonify({"message": "Location added successfully!", "location": location_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route: Get all locations
@location_api.route("/locations", methods=["GET"])
def get_locations():
    try:
        locations = list(locations_collection.find())
        for location in locations:
            location["_id"] = str(location["_id"])  # Convert ObjectId to string
        return jsonify(locations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route: Get a location by ID
@location_api.route("/locations/<location_id>", methods=["GET"])
def get_location_by_id(location_id):
    try:
        location = locations_collection.find_one({"_id": ObjectId(location_id)})
        if location:
            location["_id"] = str(location["_id"])  # Convert ObjectId to string
            return jsonify(location)
        else:
            return jsonify({"error": "Location not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route: Update a location
@location_api.route("/locations/<location_id>", methods=["PUT"])
def update_location(location_id):
    try:
        data = request.json
        update_data = {
            "id": data["id"],
            "name": data["name"],
            "coordinates": data["coordinates"],
            "type": data["type"]
        }
        result = locations_collection.update_one({"_id": ObjectId(location_id)}, {"$set": update_data})
        if result.modified_count > 0:
            return jsonify({"message": "Location updated successfully!"})
        else:
            return jsonify({"error": "Location not found or no changes made"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route: Delete a location
@location_api.route("/locations/<location_id>", methods=["DELETE"])
def delete_location(location_id):
    try:
        result = locations_collection.delete_one({"_id": ObjectId(location_id)})
        if result.deleted_count > 0:
            return jsonify({"message": "Location deleted successfully!"})
        else:
            return jsonify({"error": "Location not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500