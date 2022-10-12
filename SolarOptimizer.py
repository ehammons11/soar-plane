import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.library import power_solar as lib_solar
from aerosandbox.library import propulsion_electric as lib_prop_elec
from aerosandbox.library import propulsion_propeller as lib_prop_prop
from aerosandbox.tools import units as u
import math 

### Constants
form_factor = 1.2  # form factor [-]
oswalds_efficiency = 0.95  # Oswald efficiency factor [-]
viscosity = 1.78e-5  # viscosity of air [kg/m/s]
density = 1.23  # density of air [kg/m^3]
airfoil_thickness_fraction = 0.12  # airfoil thickness to chord ratio [-]
ultimate_load_factor = 3.8  # ultimate load factor [-]
airspeed_takeoff = 22  # takeoff speed [m/s]
CL_max = 1.5  # max CL with flaps down [-]
wetted_area_ratio = 2.05  # wetted area ratio [-]
W_W_coeff1 = 1.71e-5  # Wing Weight Coefficient 1 [1/m]
W_W_coeff2 = 10.24  # Wing Weight Coefficient 2 [Pa]
drag_area_fuselage = 0.031  # fuselage drag area [m^2]
g = 9.81  # gravitational acceleration [m/s^2]

# Solar Panel / Power System Constants
solar_cell_efficiency = 0.285 * 0.9 # solar panel efficiency [-]
rho_solar_cells = 0.255 * 1.1 # kg/m^2 # solar panel weight [kg]
solar_flux_on_horizontal = 800  # solar flux on horizontal [W/m^2]
battery_pack_cell_percentage = 0.89 

opti = asb.Opti()  # initialize an optimization environment

## Parameters
energy_generation_margin = opti.parameter(value=1.05)
battery_specific_energy_Wh_kg = opti.parameter(value=450)
allowable_battery_depth_of_discharge = opti.parameter(
    value=0.85) 

### Variables
aspect_ratio = opti.variable(init_guess=10, lower_bound=0)  # aspect ratio
wing_area = opti.variable(init_guess=10, lower_bound=0)  # total wing area [m^2]
airspeed = opti.variable(init_guess=100, lower_bound=0)  # cruising speed [m/s]
weight = opti.variable(init_guess=10000, lower_bound=0)  # total aircraft weight [N]
CL = opti.variable(init_guess=1, lower_bound=0)  # Lift coefficient of wing [-]
weight_fuselage = opti.variable(init_guess=100, lower_bound=0) # weight of fuselage [N]

### Models
# Solar Panel Model
MPPT_efficiency = 1 / 1.04  # maximum power point tracking efficiency [-]

solar_area_fraction = opti.variable(  # TODO log-transform?
    init_guess=0.80,
    scale=0.5,
    category="des"
)
opti.subject_to([
    solar_area_fraction > 0,
    solar_area_fraction < 0.80,  # TODO check
])

area_solar = (
                 wing_area
             ) * solar_area_fraction

# Energy generation cascade
power_in_from_sun = solar_flux_on_horizontal * area_solar / energy_generation_margin
power_in_after_panels = power_in_from_sun * solar_cell_efficiency
power_in = power_in_after_panels * MPPT_efficiency

mass_solar_cells = rho_solar_cells * area_solar

# mass_wires = lib_prop_elec.mass_wires(
#    wire_length=math.sqrt(aspect_ratio/wing_area) / 2,
#    max_current=power_out_propulsion_max / battery_voltage,
#    allowable_voltage_drop=battery_voltage * 0.01,
#    material="aluminum"
#)  # buildup model
# mass_wires = 0.868  # Taken from Avionics spreadsheet on 4/10/20
# https://docs.google.com/spreadsheets/d/1nhz2SAcj4uplEZKqQWHYhApjsZvV9hme9DlaVmPca0w/edit?pli=1#gid=0

# Calculate MPPT power requirement
power_in_after_panels_max = opti.variable(
    init_guess=5e3,
    scale=5e3,
    category="des"
)
opti.subject_to([
    power_in_after_panels_max > power_in_after_panels,
    power_in_after_panels_max > 0
])

n_MPPT = 5
mass_MPPT = n_MPPT * lib_solar.mass_MPPT(
    power_in_after_panels_max / n_MPPT)  # Model taken from Avionics spreadsheet on 4/10/20
# https://docs.google.com/spreadsheets/d/1nhz2SAcj4uplEZKqQWHYhApjsZvV9hme9DlaVmPca0w/edit?pli=1#gid=0

mass_power_systems_misc = 0.314  # Taken from Avionics spreadsheet on 4/10/20, includes HV-LV convs. and fault isolation mechs
# https://docs.google.com/spreadsheets/d/1nhz2SAcj4uplEZKqQWHYhApjsZvV9hme9DlaVmPca0w/edit?pli=1#gid=0

# Total system mass
mass_power_systems = mass_solar_cells + mass_MPPT + mass_power_systems_misc

# Aerodynamics model
CD_fuselage = drag_area_fuselage / wing_area
Re = (density / viscosity) * airspeed * (wing_area / aspect_ratio) ** 0.5
Cf = 0.074 / Re ** 0.2
CD_profile = form_factor * Cf * wetted_area_ratio
CD_induced = CL ** 2 / (np.pi * aspect_ratio * oswalds_efficiency)
CD = CD_fuselage + CD_profile + CD_induced
dynamic_pressure = 0.5 * density * airspeed ** 2
drag = dynamic_pressure * wing_area * CD
lift_cruise = dynamic_pressure * wing_area * CL
lift_takeoff = 0.5 * density * wing_area * CL_max * airspeed_takeoff ** 2

# Wing weight model
weight_wing_structural = W_W_coeff1 * (
        ultimate_load_factor * aspect_ratio ** 1.5 *
        (weight_fuselage * weight * wing_area) ** 0.5
) / airfoil_thickness_fraction
weight_wing_surface = W_W_coeff2 * wing_area
weight_wing = weight_wing_surface + weight_wing_structural


# Objective
opti.minimize(-power_in + drag * 1e-2)

# Design Constraints
opti.subject_to([
    weight < 533 * u.lbf,
    weight <= lift_cruise,
    weight <= lift_takeoff,
    weight == weight_fuselage + weight_wing + mass_power_systems*g,
    aspect_ratio > 5,
    lift_cruise/drag > 5,
    airspeed > 24 * u.knot,
    airspeed < 55 * u.knot,
    ])

try:
    sol = opti.solve(max_iter=1000)
except RuntimeError:
    sol = opti.debug

print(f"Minimum drag = {sol.value(drag)} N")
print(f"Aspect ratio = {sol.value(aspect_ratio)}")
print(f"Wing area = {sol.value(wing_area)} m^2")
print(f"Airspeed = {sol.value(airspeed)} m/s")
print(f"Weight = {sol.value(weight)} N")
print(f"C_L = {sol.value(CL)}")
print(f"L/D = {sol.value(lift_cruise/drag)}")
print(f"Power in = {sol.value(power_in)} W")
print(f'solar_area_fraction = {sol.value(solar_area_fraction)}')