"""Class to simulate a patient with diabetes using the simglucose library."""
import pandas as pd
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from simglucose.simulation.user_interface import simulate
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.basal_bolus_ctrller import BBController


class Simulation:
    def __init__(self, days=10, patient='adult#001', sim_type='homogeneous', variability='normal',
                 save_path='/Users/danasour/PycharmProjects/GLUSAP/data/processed/simglucose'):
        self.days = days
        self.patient = patient
        self.sim_type = sim_type
        self.variability = variability
        self.save_path = f"{save_path}/{sim_type}_{variability if sim_type == 'homogeneous' else ''}"
        self.start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.meals = self.generate_scenario()

    def generate_scenario(self):
        meals = [
            (self.start_time + timedelta(hours=8) + timedelta(minutes=15), 50.0),
            (self.start_time + timedelta(hours=13), 80.0),
            (self.start_time + timedelta(hours=20), 70.0)
        ]

        for day in range(1, self.days + 1):
            if self.sim_type == 'homogeneous':
                variability_factor = 10 if self.variability == 'low' else 30
                meals.append((self.start_time + timedelta(days=day, hours=8) + timedelta(minutes=15),
                              50.0 + random.randint(-variability_factor, variability_factor)))
                meals.append((self.start_time + timedelta(days=day, hours=13),
                              80.0 + random.randint(-variability_factor, variability_factor)))
                meals.append((self.start_time + timedelta(days=day, hours=20),
                              70.0 + random.randint(-variability_factor, variability_factor)))
            else:  # Heterogeneous
                meals.append((self.start_time + timedelta(days=day, hours=8) + timedelta(
                    minutes=random.randint(-50, 50)), 50.0 + random.randint(-10, 30)))
                meals.append((self.start_time + timedelta(days=day, hours=13) + timedelta(
                    minutes=random.randint(-100, 200)), 80.0 + random.randint(-10, 50)))
                meals.append((self.start_time + timedelta(days=day, hours=20) + timedelta(
                    minutes=random.randint(-100, 200)), 70.0 + random.randint(-20, 60)))
                if random.random() < 0.4:
                    meals.append((self.start_time + timedelta(days=day, hours=18), 20.0 + random.randint(-1, 20)))
                if random.random() < 0.2:
                    meals.append((self.start_time + timedelta(days=day, hours=11), 10.0 + random.randint(-1, 10)))
        return meals

    def run_simulation(self, recalculate=False):
        scenario = CustomScenario(start_time=self.start_time, scenario=self.meals)
        try:
            df = pd.read_csv(f'{self.save_path}/{self.patient}.csv')
            if not recalculate:
                return df
        except FileNotFoundError:
            simulate(animate=False,
                     parallel=False,
                     sim_time=timedelta(days=self.days),
                     patient_names=[self.patient],
                     cgm_name='Dexcom',
                     controller=BBController(),
                     insulin_pump_name='Insulet',
                     cgm_seed=1,
                     scenario=scenario,
                     start_time=self.start_time,
                     save_path=self.save_path)

    def generate_day_model(self):
        df = pd.read_csv(f'{self.save_path}/{self.patient}.csv')
        df['Time'] = pd.to_datetime(df['Time'])
        df['Time_of_Day'] = df['Time'].dt.time
        average_cgm = df.groupby('Time_of_Day')['CGM'].mean()
        average_cgm_df = average_cgm.reset_index()
        average_cgm_df['Time_of_Day'] = average_cgm_df['Time_of_Day'].astype(str)

        # Save the model
        average_cgm_df.to_csv(f'{self.save_path}/{self.patient}_day_model.csv', index=False)

    def visualize_results(self):
        df = pd.read_csv(f'{self.save_path}/{self.patient}.csv')
        df['Time'] = pd.to_datetime(df['Time'])
        meals_df = df[df['CHO'] > 0]

        hypoglycemia_threshold = 70
        hyperglycemia_threshold = 180

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

        ax1.plot(df['Time'], df['CGM'], label='CGM')
        ax1.axhline(y=hypoglycemia_threshold, color='red', linestyle='--', label='Hipoglucemia')
        ax1.axhline(y=hyperglycemia_threshold, color='red', linestyle='--', label='Hiperglucemia')
        ax1.set_title('CGM')
        ax1.set_ylabel('CGM')
        ax1.legend()

        ax2.bar(meals_df['Time'], meals_df['CHO'], width=0.01, color='orange')
        ax2.set_title('CHO (Ingesta de Carbohidrats)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('CHO')

        plt.tight_layout()
        plt.show()
