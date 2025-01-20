from flask import Blueprint, jsonify, request
from app.database import works_collection
from bson.objectid import ObjectId
from datetime import datetime

# Create a Blueprint for the work API
work_api = Blueprint('work_api', __name__)

# Route: Add or update work data
@work_api.route("/works", methods=["POST"])
def add_or_update_work():
    try:
        data = request.json
        wid = data["wid"]
        date = data["date"]
        work_done = data["work_done"]

        # Check if a record for the worker exists
        work_record = works_collection.find_one({"wid": wid})

        if work_record:
            # Check if the date already exists in the work array
            work_exists = next((item for item in work_record["work"] if item["date"] == date), None)
            if work_exists:
                # Increment the work_done value for the existing date
                works_collection.update_one(
                    {"wid": wid, "work.date": date},
                    {"$inc": {"work.$.work_done": work_done}}
                )
            else:
                # Add a new date entry to the work array
                works_collection.update_one(
                    {"wid": wid},
                    {"$push": {"work": {"date": date, "work_done": work_done}}}
                )
        else:
            # Create a new record if none exists
            work_data = {
                "wid": wid,
                "work": [
                    {"date": date, "work_done": work_done}
                ]
            }
            works_collection.insert_one(work_data)

        return jsonify({"message": "Work data added or updated successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Route: Get all work records
@work_api.route("/works", methods=["GET"])
def get_all_works():
    try:
        work_records = list(works_collection.find())
        for work_record in work_records:
            work_record["_id"] = str(work_record["_id"])  # Convert ObjectId to string
        return jsonify(work_records), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        

# Route: Get work record by WID
@work_api.route("/works/wid/<wid>", methods=["GET"])
def get_work_by_wid(wid):
    try:
        work_record = works_collection.find_one({"wid": int(wid)})
        if work_record:
            work_record["_id"] = str(work_record["_id"])  # Convert ObjectId to string
            return jsonify(work_record), 200
        else:
            return jsonify({"error": "Work record not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route: Get all work records by date
@work_api.route("/works/date/<date>", methods=["GET"])
def get_works_by_date(date):
    try:
        # Find all documents where the date exists in the work array
        work_records = list(works_collection.find({"work.date": date}))
        for work_record in work_records:
            work_record["_id"] = str(work_record["_id"])  # Convert ObjectId to string
        return jsonify(work_records), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
