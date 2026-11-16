from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from React (localhost:3000)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Extract inputs
    land_size = data.get("land_size")          # sq ft
    floors = data.get("floors")
    ceiling_height = data.get("ceiling_height")  # ft
    building_type = data.get("building_type")   # 0 = house, 1 = apartment

    # Basic validation
    if not all([land_size, floors, ceiling_height]) or building_type is None:
        return jsonify({"error": "Invalid input"}), 400

    # -------- Simple estimation logic --------
    built_up_area = land_size * floors

    # Material estimation (rule-of-thumb values)
    cement_bags = int(built_up_area * 0.4)
    steel_tons = round(built_up_area * 0.004, 2)
    bricks = int(built_up_area * 8)
    sand_tons = round(built_up_area * 0.02, 2)
    stone_tons = round(built_up_area * 0.015, 2)

    # Cost assumptions (₹)
    cement_cost = cement_bags * 420
    steel_cost = steel_tons * 65000
    bricks_cost = bricks * 8
    sand_cost = sand_tons * 1500
    stone_cost = stone_tons * 1800

    total_cost = int(
        cement_cost + steel_cost + bricks_cost + sand_cost + stone_cost
    )

    # Apartment adjustment
    if building_type == 1:
        total_cost = int(total_cost * 1.1)

    # -------- Response --------
    return jsonify({
        "cement_bags": cement_bags,
        "steel_tons": steel_tons,
        "bricks": bricks,
        "sand_tons": sand_tons,
        "stone_tons": stone_tons,
        "total_cost": total_cost
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
