import numpy as np
from sklearn.tree import DecisionTreeClassifier


# ==========================================
# AI-POWERED SMART TOWER
# ==========================================

print("=" * 50)
print("       AI-POWERED SMART TOWER")
print("        Energy Prediction System")
print("=" * 50)


# Training data
# Features:
# [Temperature, People, Active Floors, Previous Energy]

X = np.array([
    [18, 30, 2, 20],
    [20, 40, 3, 25],
    [22, 50, 4, 30],
    [25, 60, 5, 40],
    [28, 80, 6, 50],
    [30, 100, 8, 65],
    [32, 120, 10, 80],
    [35, 150, 12, 100],
    [24, 45, 3, 28],
    [27, 70, 5, 45],
    [31, 110, 9, 75],
    [34, 140, 11, 90]
])


# Energy demand:
# 0 = Low
# 1 = Medium
# 2 = High

y = np.array([
    0, 0, 0, 1,
    1, 1, 2, 2,
    0, 1, 2, 2
])


# Create AI model
model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

# Train model
model.fit(X, y)


print("\nEnter Tower Information")
print("-" * 40)

temperature = float(
    input("Temperature (°C): ")
)

people = int(
    input("Number of people inside: ")
)

floors = int(
    input("Number of active floors: ")
)

previous_energy = float(
    input("Previous energy usage (kWh): ")
)


# Prepare input
tower_data = np.array([[
    temperature,
    people,
    floors,
    previous_energy
]])


# AI prediction
prediction = model.predict(tower_data)[0]


# Probability
probabilities = model.predict_proba(tower_data)[0]

confidence = max(probabilities) * 100


# Convert prediction into label
if prediction == 0:
    demand = "LOW"
elif prediction == 1:
    demand = "MEDIUM"
else:
    demand = "HIGH"


print("\n" + "=" * 50)
print("              AI PREDICTION")
print("=" * 50)

print(f"Energy Demand: {demand}")
print(f"Model Confidence: {confidence:.2f}%")


# Smart recommendations

print("\nSmart Tower Recommendation:")

if prediction == 0:

    print("✓ Energy consumption is expected to be low.")
    print("✓ Normal tower operations can continue.")

elif prediction == 1:

    print("⚡ Moderate energy consumption expected.")
    print("✓ Consider optimizing lighting and cooling.")

else:

    print("⚠ High energy consumption expected.")
    print("✓ Reduce unnecessary lighting.")
    print("✓ Optimize air-conditioning.")
    print("✓ Monitor high-energy equipment.")

print("\nAI analysis completed.")
