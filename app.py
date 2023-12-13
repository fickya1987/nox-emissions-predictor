import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from sklearn.model_selection import train_test_split

df = pd.read_csv('final_with_winf_clean.csv')
# get the locations
X = df.iloc[:, 1:]  # Exclude the first column as it is the target variable
y = df.iloc[:, 0]   # Use the first column as the target variable
# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)

expected_inputs=['facility_id', 'wind_speed', 'wind_direction', 'day', 'month', 'year', 'average_temp_K_', 'average_solarrad_Wm2_', 'average_RH_perc_', 'average_cloud_fraction_', 'average_no2_kgm2_', 'average_no2flux_kgm2s_']
models=[("Random Forest","random_forest"),("K-Nearest Neighbor", "knn"),("Support Vector Regression", "svr")]

def make_prediction(model, *args):
    input_data = pd.DataFrame([args], columns=expected_inputs)
    if model =="random_forest":
        with open("rf.pkl", "rb") as f:
            rf  = pickle.load(f)
            preds = rf.predict(input_data)
            output_value = preds[0]
            formatted_output = "{:.2f}".format(output_value)
            result_string = formatted_output + " kg/hour"
            return result_string
    if model =="knn":
        with open("knn.pkl", "rb") as f:
            knn  = pickle.load(f)
            preds = knn.predict(input_data)
            output_value = preds[0]
            formatted_output = "{:.2f}".format(output_value)
            result_string = formatted_output + " kg/hour"
            return result_string
    if model =="svr":
        with open("svr.pkl", "rb") as f:
            svr  = pickle.load(f)
            preds = svr.predict(input_data)
            output_value = preds[0]
            formatted_output = "{:.2f}".format(output_value)
            result_string = formatted_output + " kg/hour"
            return result_string
        

data = pd.read_csv('final_with_winf_clean.csv')
def make_plot(plot_type):
    if plot_type == "Distribution of NOx value Across Days of the Week":
        # Convert date columns to datetime
        data['date'] = pd.to_datetime(data[['year', 'month', 'day']])

        # Extract day of the week and create a binary indicator for weekends
        data['day_of_week'] = data['date'].dt.dayofweek
        data['is_weekend'] = (data['date'].dt.dayofweek // 5 == 1).astype(int)

        # Visualize the distribution of nox_value across days of the week
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='day_of_week', y='nox_value', data=data)
        plt.title('Distribution of nox_value Across Days of the Week')
        plt.xlabel('Day of the Week')
        plt.ylabel('nox_value')
        return plt
        
    elif plot_type == "Wind Speed vs. NOx Emissions":

        # Assuming df is your DataFrame
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # Scatter plot between Wind Speed, Wind Direction, and NOx Emissions
        sns.scatterplot(x='wind_speed', y='nox_value', hue='wind_direction', data=df, palette='viridis', alpha=0.7)

        # Set plot labels and title
        plt.xlabel('Wind Speed')
        plt.ylabel('NOx Emissions')
        plt.title('Scatter Plot: Wind Speed vs. NOx Emissions (Color-coded by Wind Direction)')
        plt.legend(title='Wind Direction')

        return plt
    
    elif plot_type == "Relationship between Temperature, Solar Radiation, and NOx value":
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='average_temp_K_', y='average_solarrad_Wm2_', hue='nox_value', data=data, palette='viridis')
        plt.title('Relationship between Temperature, Solar Radiation, and nox_value')
        plt.xlabel('Average Temperature (K)')
        plt.ylabel('Average Solar Radiation (W/m^2)')
        return plt
    
    elif plot_type == "Distribution of NOx value Across Facilities":
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='facility_id', y='nox_value', data=data)
        plt.title('Distribution of nox_value Across Facilities')
        plt.xlabel('Facility ID')
        plt.ylabel('nox_value')
        return plt
    
    elif plot_type == "Correlation Heatmap":
        corr_matrix = data.corr()
        # Create a heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
        plt.title('Correlation Heatmap')
        return plt


with gr.Blocks(theme=gr.themes.Base()) as demo:

    gr.Markdown(
    """
    <h1 align = "center"> NitroEmissions Predictor</h1>
    """)    
    with gr.Tabs() as tabs:
        with gr.Tab("Feature Visualization", id=0):
            button = gr.Radio(label="Plot type", choices=['Distribution of NOx value Across Days of the Week', 'Wind Speed vs. NOx Emissions', 'Relationship between Temperature, Solar Radiation, and NOx value',
                               'Distribution of NOx value Across Facilities', "Correlation Heatmap"], value='Distribution of NOx value Across Days of the Week')
            plot = gr.Plot(label="Plot")
            button.change(make_plot, inputs=button, outputs=[plot])
            demo.load(make_plot, inputs=[button], outputs=[plot])

        with gr.Tab("Predictive Models", id=1):
            with gr.Row():
                with gr.Column(scale=50):
                    model = gr.Dropdown(models, label="Select Model")
                    facility_id = gr.Number(label = "Enter the Facility ID")
                    wind_speed = gr.Slider(10000, 50000, step=1, label="Enter the Wind Speed")
                    wind_direction = gr.Number(label = "Enter the Wind Direction")
                    with gr.Row():
                        day=gr.Number(label = "Enter the Day of the Year")
                        month=gr.Number(label = "Enter the Month of the Year")
                        year=gr.Number(label = "Enter the Year")
                with gr.Column(scale=50):
                    average_temp_K_=gr.Slider(1000, 50000, step=1, label="Enter the Temperature")
                    average_solarrad_Wm2_=gr.Slider(-500, 50000, step=1, label="Enter the Solar Radiation")
                    average_RH_perc_=gr.Slider(0,100, step=1, label="Enter the Humidity %")
                    average_cloud_fraction_=gr.Slider(0, 1, step=0.0001, label="Enter the Cloud Fraction")
                    average_no2_kgm2_=gr.Number(label = "Enter the average NO2 in Kg/m^2")
                    average_no2flux_kgm2s_=gr.Number(label = "Enter the average NO2 flux")
            with gr.Row():
                    btn = gr.Button("Submit")
            with gr.Row():
                    preds = gr.Text(label="Predicted NOx Emission")
                    
        btn.click(make_prediction, inputs=[model, facility_id, wind_speed, wind_direction, day, month, year, average_temp_K_, average_solarrad_Wm2_, average_RH_perc_, average_cloud_fraction_, average_no2_kgm2_, average_no2flux_kgm2s_], outputs=[preds])
            

if __name__ == "__main__":
    demo.launch(debug=True)