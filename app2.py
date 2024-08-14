import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

# Title of the app
st.title("Solar Thermal Collector Performance Prediction")

# Load and display image
photo = Image.open('gui2.png')
st.image(photo)
# Dictionary to map fluid names to saturation temperatures
fluid_temps = {
    'Acetone': 56.24,
    'Methanol': 65,
    'Ethanol': 78.4,
    'Water': 100
}
# Sidebar inputs
manifold_fluid = st.sidebar.radio('Manifold fluid', ['Water', 'Air'])
#using dct
fluid_name = st.sidebar.radio('Heatpipe Fluid', list(fluid_temps.keys()))
T_sat = fluid_temps[fluid_name] # Get saturation temperature from dictionary

N = st.sidebar.number_input('Number of tubes (N)', min_value=1)
D_o = st.sidebar.number_input('Outer diameter of tube (D_o)')
L_tube = st.sidebar.number_input('Length of the tube (L_tube)')
L_c = st.sidebar.number_input('Length of the condenser (L_c)')
L_e = st.sidebar.number_input('Length of the evaporator (L_e)')
L_a = st.sidebar.number_input('Length of the adiabatic section (L_a)')
De_o = st.sidebar.number_input('Outer diameter of evaporator (De_o)')
t_e = st.sidebar.number_input('Thickness of evaporator (t_e)')
Dc_o = st.sidebar.number_input('Outer diameter of condenser (Dc_o)')
t_c = st.sidebar.number_input('Thickness of condenser (t_c)')
theta = st.sidebar.number_input('Angle of radiation (theta)')
t_g = st.sidebar.number_input('Thickness of glass (t_g)')
D_H = st.sidebar.number_input('Hydraulic diameter of tube (D_H)')
A_man = st.sidebar.number_input('Area of manifold (A_man)')
alpha_ab = st.sidebar.number_input('Absorptivity of absorber (alpha_ab)')
epsilon_ab = st.sidebar.number_input('Emissivity of absorber (epsilon_ab)')
epsilon_g = st.sidebar.number_input('Emissivity of glass (epsilon_g)')
tau_g = st.sidebar.number_input('Transmissivity of glass (tau_g)')
I = st.sidebar.number_input('Solar irradiance (I)')
T_amb = st.sidebar.number_input('Ambient temperature (T_amb)')
U_amb = st.sidebar.number_input('Velocity of wind (U_amb)')
T_in = st.sidebar.number_input('Inlet temperature (T_in)')
m_dot = st.sidebar.number_input('Mass flow rate of manifold fluid (m_dot)')

# Prepare input data for prediction
input_data = pd.DataFrame({
    'T_sat': [T_sat],
    'N': [N],
    'D_o': [D_o],
    'L_tube': [L_tube],
    'L_c': [L_c],
    'L_e': [L_e],
    'L_a': [L_a],
    'De_o': [De_o],
    't_e': [t_e],
    'Dc_o': [Dc_o],
    't_c': [t_c],
    'theta': [theta],
    't_g': [t_g],
    'D_H': [D_H],
    'A_man': [A_man],
    'alpha_ab': [alpha_ab],
    'epsilon_ab': [epsilon_ab],
    'epsilon_g': [epsilon_g],
    'tau_g': [tau_g],
    'I': [I],
    'T_amb': [T_amb],
    'U_amb': [U_amb],
    'T_in': [T_in],
    'm_dot': [m_dot]
})

# Load data based on fluid type selection
if manifold_fluid == 'Water':
    path = "Water.csv"
else:
    path = "Air.csv"

df = pd.read_csv(path)
df = df.drop("Fluid", axis=1, errors='ignore') # Drop Fluid column if it exists
df = df.sample(frac=1).reset_index(drop=True)

X = df.iloc[:, :24].values
y = df.iloc[:, 24].values

# Split data into training and testing sets
x_train, x_test_and_val, y_train, y_test_and_val = train_test_split(X, y, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test_and_val, y_test_and_val, test_size=0.5, random_state=20)

# Scale features
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(x_train)
X_val_scaled = scaler_X.transform(x_val)
X_test_scaled = scaler_X.transform(x_test)

# Scale target variable
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# Scale input data for prediction
X_input_scaled = scaler_X.transform(input_data)

# Train model
reg_mod = xgb.XGBRegressor(n_estimators=500)
reg_mod.fit(X_train_scaled, y_train_scaled)

#Make predictions
y_pred_scaled = reg_mod.predict(X_input_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# Display prediction result
st.subheader("Predicted Output")
st.write("Predicted Performance:", y_pred[0][0])