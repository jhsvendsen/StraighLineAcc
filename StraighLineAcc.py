# Script to calculate straigh line performance
# The car is presumed to be a rear wheel driven car

# Downloading Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vehiclemodel import vehicle

##############################################################################
# Downloading and processing vehicle model
##############################################################################

vehicle_model = vehicle.vehicle_model_download('AudiR8.csv')

# Define parameters of Acceleration event
wind_speed = 1 # Negative against car
road_incl = 0 # Negative is 
rho = 1.29 # Density of air
g = 9.81 # gravity

# Car parameters
aero_100 = vehicle_model.iloc[0,0]
k = vehicle_model.iloc[1,0]
Mu = vehicle_model.iloc[2,0]
tire_press = vehicle_model.iloc[3,0]
torq_loss = vehicle_model.iloc[4,0]
Cd = vehicle_model.iloc[5,0]
frontal_area = vehicle_model.iloc[6,0]
wheel_rad = vehicle_model.iloc[7,0]
wheel_avg_rad = vehicle_model.iloc[8,0]
wheel_mass = vehicle_model.iloc[9,0]
wheel_inertia = vehicle_model.iloc[10,0]
car_mass = vehicle_model.iloc[11,0]
cog_h = vehicle_model.iloc[12,0]
wb = vehicle_model.iloc[13,0]
cog_a = vehicle_model.iloc[14,0]
Cl = vehicle_model.iloc[15,0]

# Engine parameters
torque = vehicle_model.iloc[:,1:3]
gearbox = pd.DataFrame()
gearbox['gear'] =  vehicle_model.iloc[:6,4]
final_drive = vehicle_model.iloc[6,4]
gearbox['final'] = gearbox['gear']*final_drive
gearbox['gears'] = [1,2,3,4,5,6]

plt.plot(torque.iloc[:,0],torque.iloc[:,1])
plt.title('Torque vs RPM')
plt.show()

# Set a revlimit
engine_stall_rpm = 1000
revlimit = 7000
idle_limit = 1000

# Calculating constants
f_road_incl = car_mass*np.sin(road_incl*(3.14/180))
rMR_ML = wheel_rad*(car_mass+wheel_mass)
lengthR = (4*wheel_inertia)/wheel_rad 

##############################################################################
# Setting up and running simulation
##############################################################################

v_end = 90 # meters of acceleration in m/s
v0 = 0 # Starting velocity of acceleration m/s
v_step = 1 # Velocity step between calculations
i = 0


# Initialisation arrays

n_datapoints = int((v_end-v0)/v_step)
results = pd.DataFrame()
namearray = ['vCar','w_omega','wheel_rpm','first_rpm','second_rpm','third_rpm',
             'fourth_rpm','fifth_rpm','sixth_rpm','used_rpm','torque_rpm',
             'gear_used','wheel_torque','force_x_wheel','aero_load',
             'force_x_max','actual_force_x','rear_load','v_air','drag','road_fric',
             'road_force','nett_force','acc_x','weight_transfer','new_rear_load',
             'new_max_x_force','new_actual_force_x','new_nett_force','new_acc_x','timer']
for name in namearray:
    results[name] = [0.00]*n_datapoints

# Simulation
for v in range(v0,v_end,v_step):
    
    # Velocity of car
    results['vCar'][i] = v
    
    # Wheel speed
    results['w_omega'][i] = v/wheel_avg_rad
    results['wheel_rpm'][i] = (60/(2*3.14))*results['w_omega'][i]
    
    # Calculating RPM's
    results['first_rpm'][i] = results['wheel_rpm'][i]*gearbox.iloc[0,0]*final_drive
    results['second_rpm'][i] = results['wheel_rpm'][i]*gearbox.iloc[1,0]*final_drive
    results['third_rpm'][i] = results['wheel_rpm'][i]*gearbox.iloc[2,0]*final_drive
    results['fourth_rpm'][i] = results['wheel_rpm'][i]*gearbox.iloc[3,0]*final_drive
    results['fifth_rpm'][i] = results['wheel_rpm'][i]*gearbox.iloc[4,0]*final_drive
    results['sixth_rpm'][i] = results['wheel_rpm'][i]*gearbox.iloc[5,0]*final_drive
    
    # Finding used gear
    if results['first_rpm'][i] < revlimit:
        if results['first_rpm'][i]  < engine_stall_rpm:
            results['used_rpm'][i]  = 1000
        else:
            results['used_rpm'][i] = results['first_rpm'][i]
        results['gear_used'][i] = 1
    elif results['second_rpm'][i] < revlimit:
        results['used_rpm'][i] = results['second_rpm'][i]
        results['gear_used'][i] = 2
    elif results['third_rpm'][i] < revlimit:
        results['used_rpm'][i] = results['third_rpm'][i]
        results['gear_used'][i] = 3
    elif results['fourth_rpm'][i] < revlimit:
        results['used_rpm'][i] = results['fourth_rpm'][i]
        results['gear_used'][i] = 4
    elif results['fifth_rpm'][i] < revlimit:
        results['used_rpm'][i] = results['fifth_rpm'][i]
        results['gear_used'][i] = 5
    else:
        results['used_rpm'][i] = results['sixth_rpm'][i]
        results['gear_used'][i] = 6
    
    # Interpeting the torque from gear
    results['torque_rpm'][i] = np.interp(results['used_rpm'][i],
                                         torque.iloc[:,0],
                                         torque.iloc[:,1])
    # Interperting the gearbox value
    results['wheel_torque'][i] =  results['torque_rpm'][i]*np.interp(results['gear_used'][i],
                                                                             gearbox.iloc[:,0],
                                                                             gearbox.iloc[:,2])
    
    # Possible force from tires
    results['force_x_wheel'][i] = results['wheel_torque'][i]/wheel_avg_rad
    
    results['aero_load'][i] = Cl*0.5*rho*frontal_area*np.power(v,2)
    
    results['force_x_max'][i] = Mu*((car_mass*g)+results['aero_load'][i])
    
    
    # Whichever force is smaller is the actual force
    if results['force_x_wheel'][i]  < results['force_x_max'][i]:
        results['actual_force_x'][i] = results['force_x_wheel'][i]
    else:
        results['actual_force_x'][i] = results['force_x_max'][i]
    
    # Standard load w. weight transfer    
    results['rear_load'][i] = ((car_mass*g*cog_a)/wb)+results['aero_load'][i]
    
    # Corrected air for wind direction
    results['v_air'][i] = v + wind_speed
    
    # Calculating drag
    results['drag'][i] = Cd*0.5*rho*frontal_area*np.power(results['v_air'][i],2)
    
    # Resulting road friction
    results['road_fric'][i] = 0.005+(1/tire_press)*(0.01+0.0095*np.power((v*2.24)/100,2))
    
    # Friction coefficient into force
    results['road_force'][i] = results['road_fric'][i]*(car_mass*g)
    
    # Calculating total force
    results['nett_force'][i] = results['actual_force_x'][i] - results['road_force'][i] - results['drag'][i] - f_road_incl 
    
    # Acceleration from force
    results['acc_x'][i] = results['nett_force'][i]/car_mass
    
    # Weight transfer from acceleration
    results['weight_transfer'][i] = (cog_h*(results['acc_x'][i]*car_mass))/wb
    
    # Causes new rear load
    results['new_rear_load'][i] = results['weight_transfer'][i] + results['rear_load'][i] 
    
    # Resulting new max force
    results['new_max_x_force'][i] = Mu * (results['new_rear_load'][i] + results['aero_load'][i])
    
    # Whichever force is smaller is the actual force
    if results['force_x_wheel'][i]  <  results['new_max_x_force'][i]:
        results['new_actual_force_x'][i] = results['force_x_wheel'][i]
    else:
        results['new_actual_force_x'][i] =  results['new_max_x_force'][i]
    
    # Calculating the net force again
    results['new_nett_force'][i] = results['new_actual_force_x'][i] - results['road_force'][i] - results['drag'][i] - f_road_incl 
    
    # Final acceleration
    results['new_acc_x'][i] = results['new_nett_force'][i] /car_mass
    
    if i == 0:
        results['timer'][i]  = v_step/results['new_acc_x'][i]
    else:
        results['timer'][i] = results['timer'][i-1] + v_step/results['new_acc_x'][i]

    # Updating index
    i = i+1

##############################################################################
# Results
##############################################################################
print('Time:' ,round(results['timer'][i-1],2))
print('Acc max:', round(max(results['new_acc_x']),2))
print('Max gear', round(max(results['gear_used'])))

plt.plot(results['vCar'],results['new_acc_x'])
plt.title('Acceleration')
plt.show()
