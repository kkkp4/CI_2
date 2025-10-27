import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd

# -----------------------------
# 1. สร้างตัวแปร Input/Output
# -----------------------------
goals = ctrl.Antecedent(np.arange(0, 6, 1), 'goals')              # 0-5 ประตู
running = ctrl.Antecedent(np.arange(0, 13, 1), 'running')         # 0-12 km
passing = ctrl.Antecedent(np.arange(0, 101, 1), 'passing')        # 0-100 %
performance = ctrl.Consequent(np.arange(0, 101, 1), 'performance') # 0-100 คะแนน

# -----------------------------
# 2. กำหนด Membership Functions
# -----------------------------
goals['Low'] = fuzz.trimf(goals.universe, [0, 0, 2])
goals['Medium'] = fuzz.trimf(goals.universe, [1, 3, 5])
goals['High'] = fuzz.trimf(goals.universe, [3, 5, 5])

running['Low'] = fuzz.trimf(running.universe, [0, 0, 4])
running['Medium'] = fuzz.trimf(running.universe, [3, 6, 9])
running['High'] = fuzz.trimf(running.universe, [8, 12, 12])

passing['Poor'] = fuzz.trimf(passing.universe, [0, 0, 50])
passing['Average'] = fuzz.trimf(passing.universe, [40, 60, 80])
passing['Excellent'] = fuzz.trimf(passing.universe, [85, 100, 100])

performance['Low'] = fuzz.trimf(performance.universe, [0, 0, 40])
performance['Medium'] = fuzz.trimf(performance.universe, [30, 50, 70])
performance['High'] = fuzz.trimf(performance.universe, [60, 100, 100])

# -----------------------------
# 3. สร้างกฎ Fuzzy Rules
# -----------------------------
rule1 = ctrl.Rule(goals['High'] | passing['Excellent'], performance['High'])
rule2 = ctrl.Rule(goals['Medium'] & running['High'], performance['High'])
rule3 = ctrl.Rule(goals['Low'] & passing['Poor'], performance['Low'])
rule4 = ctrl.Rule(running['Medium'] & passing['Average'], performance['Medium'])
rule5 = ctrl.Rule(goals['Low'] & running['High'], performance['Medium'])
rule6 = ctrl.Rule(goals['Medium'] & passing['Excellent'], performance['High'])
rule7 = ctrl.Rule(goals['High'] & running['High'] & passing['Excellent'], performance['High'])
rule8 = ctrl.Rule(goals['Low'] & running['Low'] & passing['Excellent'], performance['Medium'])
rule9 = ctrl.Rule(goals['Low'] & running['Low'] & passing['Poor'], performance['Low'])
rule10 = ctrl.Rule(goals['Low'] & running['Low'] & passing['Average'], performance['Low'])

# -----------------------------
# 4. สร้าง Control System
# -----------------------------
performance_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10])
performance_sim = ctrl.ControlSystemSimulation(performance_ctrl)

# -----------------------------
# 5. ฟังก์ชันประเมินนักเตะ
# -----------------------------
def eval_player(g, r, p):
    sim = ctrl.ControlSystemSimulation(performance_ctrl)
    sim.input['goals'] = g
    sim.input['running'] = r
    sim.input['passing'] = p
    sim.compute()
    return sim.output.get('performance', 0)   # 👈 ป้องกัน KeyError

# -----------------------------
# 6. ข้อมูลนักเตะตัวอย่าง
# -----------------------------
players = {
    "Player A": (2, 10, 80),   # ยิง 2 ประตู, วิ่ง 10km, passing 80%
    "Player B": (0, 12, 90),
    "Player C": (3, 5, 60),
    "Player D": (1, 3, 40),
    "Player E": (5, 7, 95)
}

# -----------------------------
# 7. คำนวณผลลัพธ์
# -----------------------------
results = []
for name, (g, r, p) in players.items():
    score = eval_player(g, r, p)
    results.append([name, g, r, p, round(score, 2)])

# -----------------------------
# 8. สร้าง Simulation Surface Plot
# -----------------------------
G = np.arange(0, 6, 1)        # Goals 0-5
R = np.arange(0, 13, 1)       # Running 0-12
Z = np.zeros((len(G), len(R)))

for i, g in enumerate(G):
    for j, r in enumerate(R):
        Z[i, j] = eval_player(g, r, 50)               

R_mesh, G_mesh = np.meshgrid(R, G)

df = pd.DataFrame(results, columns=["Player", "Goals", "Running (km)", "Passing (%)", "Performance Score"])
print(df)

plt.figure(figsize=(8,6))
cp = plt.contourf(R_mesh, G_mesh, Z, levels=50, cmap='viridis')
'''plt.colorbar(cp, label='Performance Score')
plt.xlabel('Running Distance (km)')
plt.ylabel('Goals')
plt.title('Performance surface (Passing fixed=50%)')
plt.show()'''

# 3D
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(R_mesh, G_mesh, Z, cmap='viridis')
ax.set_xlabel('Running Distance (km)')
ax.set_ylabel('Goals')
ax.set_zlabel('Performance Score')
plt.title('3D Performance Surface (Passing=50%)')
plt.show()