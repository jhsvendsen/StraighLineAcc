class vehicle:
    # Importing the Vehicle parameters
    def vehicle_model_download(name):
        import pandas as pd
        vehicle_model = pd.read_csv(name)
        vehicle_model = vehicle_model.drop(['Parameters'], axis=1)
        vehicle_model.columns = ['Parameters','RPM','Torque','Gear','Ratio']

        return vehicle_model


