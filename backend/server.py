from flask import Flask, jsonify
from app.apis.asset import asset  # Import the asset API Blueprint
from app.apis.task_api import task_api
from app.apis.location_api import location_api  # Import the location API Blueprint
from app.apis.user_api import user_api
from app.apis.worker import work_api
app = Flask(__name__)

# Register Blueprints
app.register_blueprint(asset)
app.register_blueprint(task_api)
app.register_blueprint(location_api)
app.register_blueprint(user_api)
app.register_blueprint(work_api)

@app.route("/")
def home():
    return jsonify({"message": "Warehouse Tracking Backend is running!"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
