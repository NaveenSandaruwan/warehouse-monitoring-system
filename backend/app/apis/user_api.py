from flask import Blueprint, jsonify, request
from app.database import users_collection
from bson.objectid import ObjectId

# Create a Blueprint for the user API
user_api = Blueprint('user_api', __name__)

# Route: Add a new user
@user_api.route("/users", methods=["POST"])
def add_user():
    try:
        data = request.json
        user_data = {
            "_id": str(ObjectId()),  # Generate a unique MongoDB ObjectId
            "name": data["name"],
            "wid": data["wid"],
            "type": data["type"],
            "current_location": data["current_location"],
            "Work_done": data["Work_done"],
            "status": data["status"],
            "last_updated": data["last_updated"]
        }
        users_collection.insert_one(user_data)
        return jsonify({"message": "User added successfully!", "user": user_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route: Get all users
@user_api.route("/users", methods=["GET"])
def get_users():
    try:
        users = list(users_collection.find())
        for user in users:
            user["_id"] = str(user["_id"])  # Convert ObjectId to string
        return jsonify(users)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route: Get a user by ID
@user_api.route("/users/<user_id>", methods=["GET"])
def get_user_by_id(user_id):
    try:
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if user:
            user["_id"] = str(user["_id"])  # Convert ObjectId to string
            return jsonify(user)
        else:
            return jsonify({"error": "User not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route: Get a user by WID
@user_api.route("/users/wid/<wid>", methods=["GET"])
def get_user_by_wid(wid):
    try:
        user = users_collection.find_one({"wid": int(wid)})
        if user:
            user["_id"] = str(user["_id"])  # Convert ObjectId to string
            return jsonify(user)
        else:
            return jsonify({"error": "User not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route: Update a user
@user_api.route("/users/<user_id>", methods=["PUT"])
def update_user(user_id):
    try:
        data = request.json
        update_data = {
            "name": data["name"],
            "wid": data["wid"],
            "type": data["type"],
            "current_location": data["current_location"],
            "Work_done": data["Work_done"],
            "status": data["status"],
            "last_updated": data["last_updated"]
        }
        result = users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": update_data})
        if result.modified_count > 0:
            return jsonify({"message": "User updated successfully!"})
        else:
            return jsonify({"error": "User not found or no changes made"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route: Update current location by ID
@user_api.route("/users/<user_id>/location", methods=["PUT"])
def update_user_location(user_id):
    try:
        data = request.json
        result = users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": {"current_location": data["current_location"]}})
        if result.modified_count > 0:
            return jsonify({"message": "User location updated successfully!"})
        else:
            return jsonify({"error": "User not found or no changes made"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route: Update work done by ID
@user_api.route("/users/<user_id>/work_done", methods=["PUT"])
def update_user_work_done(user_id):
    try:
        data = request.json
        result = users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": {"Work_done": data["Work_done"]}})
        if result.modified_count > 0:
            return jsonify({"message": "User work done updated successfully!"})
        else:
            return jsonify({"error": "User not found or no changes made"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route: Delete a user
@user_api.route("/users/<user_id>", methods=["DELETE"])
def delete_user(user_id):
    try:
        result = users_collection.delete_one({"_id": ObjectId(user_id)})
        if result.deleted_count > 0:
            return jsonify({"message": "User deleted successfully!"})
        else:
            return jsonify({"error": "User not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500