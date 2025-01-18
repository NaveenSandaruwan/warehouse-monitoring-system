import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.app.database import db

assets_collection = db.getCollection("assets") if db else None

def add_asset(asset_id, name, asset_type, location_id, status, last_updated):
    asset_data = {
        "_id": asset_id,
        "name": name,
        "type": asset_type,  # e.g., "person", "forklift"
        "location_id": location_id,
        "status": status,  # e.g., "idle", "active"
        "last_updated": last_updated,
    }
    assets_collection.insert_one(asset_data)
    print("Asset added:", asset_data)

def get_asset(asset_id):
    return assets_collection.find_one({"_id": asset_id})

def update_asset(asset_id, updates):
    assets_collection.update_one(
        {"_id": asset_id},
        {"$set": updates}
    )
    print(f"Asset {asset_id} updated with {updates}")

def delete_asset(asset_id):
    assets_collection.delete_one({"_id": asset_id})
    print(f"Asset {asset_id} deleted")

get_asset("678a32467bb2f69799c158fd")