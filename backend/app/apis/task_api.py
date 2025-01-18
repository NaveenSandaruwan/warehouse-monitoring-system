from flask import Blueprint, jsonify, request
from app.database import tasks_collection
from bson.objectid import ObjectId

# Create a Blueprint for the task API
task_api = Blueprint('task_api', __name__)

# Route: Add a new task
@task_api.route("/tasks", methods=["POST"])
def add_task():
    data = request.json
    task_data = {
        "_id": str(ObjectId()),  # Generate a unique MongoDB ObjectId
        "asset_id": data["asset_id"],
        "task_description": data["task_description"],
        "status": data["status"],  # e.g., "pending", "completed"
        "created_at": data["created_at"],  # ISO format timestamp
        "updated_at": data["updated_at"],  # ISO format timestamp
        "route_id": data["route_id"]
    }
    tasks_collection.insert_one(task_data)
    return jsonify({"message": "Task added successfully!", "task": task_data})

# Route: Get all tasks
@task_api.route("/tasks", methods=["GET"])
def get_tasks():
    tasks = list(tasks_collection.find())
    for task in tasks:
        task["_id"] = str(task["_id"])  # Convert ObjectId to string
    return jsonify(tasks)

# Route: Get a specific task by ID
@task_api.route("/tasks/<task_id>", methods=["GET"])
def get_task(task_id):
    task = tasks_collection.find_one({"_id": ObjectId(task_id)})
    if not task:
        return jsonify({"error": "Task not found"}), 404
    task["_id"] = str(task["_id"])  # Convert ObjectId to string
    return jsonify(task)

# Route: Update a task
@task_api.route("/tasks/<task_id>", methods=["PUT"])
def update_task(task_id):
    updates = request.json
    tasks_collection.update_one({"_id": ObjectId(task_id)}, {"$set": updates})
    return jsonify({"message": f"Task {task_id} updated successfully!"})

# Route: Delete a task
@task_api.route("/tasks/<task_id>", methods=["DELETE"])
def delete_task(task_id):
    tasks_collection.delete_one({"_id": ObjectId(task_id)})
    return jsonify({"message": f"Task {task_id} deleted successfully!"})
