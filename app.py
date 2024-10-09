from dotenv import load_dotenv
import os

# Add these imports
from threading import Thread
import queue
from openai import AssistantEventHandler
from typing_extensions import override
import json

load_dotenv()

import openai
import time
import gradio as gr
from autogen import UserProxyAgent, config_list_from_json
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
from gradio_datetimerange import DateTimeRange
import os
from time import sleep
from gradio_pdf import PDF

from pandasai.llm.openai import OpenAI
from pandasai import Agent
import matplotlib.pyplot as plt
import io
from pandasai import SmartDataframe
from collections import Counter
from gradio_pdf import PDF  # Ensure you have installed gradio_pdf

from tavily import TavilyClient  # Ensure you have installed the tavily library


# llmmodel = OpenAI(api_token=os.environ["OPENAI_API_KEY"], model='gpt-4o')


import requests


# Define the directory containing the PDFs
PDF_DIR = "usedpdfs"  # Replace with your directory path

# Define your desired default PDF file
DEFAULT_PDF = "s41597-024-03770-7.pdf"  # Replace with your actual PDF filename



# Ensure the PDF_DIR exists
if not os.path.isdir(PDF_DIR):
    raise ValueError(f"The directory '{PDF_DIR}' does not exist. Please check the path.")



# Get list of PDF files in the directory
pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]

if DEFAULT_PDF not in pdf_files:
    raise ValueError(f"Default PDF '{DEFAULT_PDF}' not found in '{PDF_DIR}'.")

# Check if there are PDF files in the directory
if not pdf_files:
    raise ValueError(f"No PDF files found in the directory '{PDF_DIR}'.")

def display_pdf(selected_file):
    """
    Given the selected file name, return the full path to display in the PDF viewer.
    """
    file_path = os.path.join(PDF_DIR, selected_file)
    return file_path





def web_search(query: str) -> str:
    """
    Performs a web search using the Tavily API and returns the context string.

    Parameters:
    - query (str): The search query.

    Returns:
    - str: The context string from the Tavily API or an error message.
    """
    try:
        # Step 1: Instantiate the TavilyClient
        tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

        # Step 2: Execute the search query
        context = tavily_client.get_search_context(query=query)

        # Step 3: Return the context
        return f"**Web Search Context:**\n{context}"
    except Exception as e:
        return f"Error performing web search: {str(e)}"



# Ensure the PDF_DIR exists
if not os.path.isdir(PDF_DIR):
    raise ValueError(f"The directory '{PDF_DIR}' does not exist. Please check the path.")

# Get list of PDF files in the directory
pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]

# Check if there are PDF files in the directory
if not pdf_files:
    raise ValueError(f"No PDF files found in the directory '{PDF_DIR}'.")

def display_pdf(selected_file):
    """
    Given the selected file name, return the full path to display in the PDF viewer.
    """
    file_path = os.path.join(PDF_DIR, selected_file)
    return file_path



# Function to generate a date range
def generate_date_range(start_date, end_date, freq="D"):
    return pd.date_range(start=start_date, end=end_date, freq=freq)


# Function to generate synthetic data for each component
def generate_synthetic_data(dates):
    # Define random seed for reproducibility
    np.random.seed(0)

    # Generate random data for each component
    data = {
        "Temperature_Pressure_Relief_Valve": np.random.choice(
            [0, 1], size=len(dates)
        ),  # 0 = OK, 1 = Faulty
        "Outlet_Nipple_Assembly": np.random.normal(
            loc=80, scale=10, size=len(dates)
        ),  # Temperature in °F
        "Inlet_Nipple": np.random.normal(
            loc=50, scale=5, size=len(dates)
        ),  # Temperature in °F
        "Upper_Element": np.random.normal(
            loc=150, scale=20, size=len(dates)
        ),  # Wattage (Watts)
        "Lower_Element": np.random.normal(
            loc=150, scale=20, size=len(dates)
        ),  # Wattage (Watts)
        "Anode_Rod": np.random.normal(
            loc=7, scale=1.5, size=len(dates)
        ),  # Length in inches
        "Drain_Valve": np.random.choice(
            [0, 1], size=len(dates)
        ),  # 0 = Closed, 1 = Open
        "Upper_Thermostat": np.random.normal(
            loc=120, scale=10, size=len(dates)
        ),  # Temperature in °F
        "Lower_Thermostat": np.random.normal(
            loc=120, scale=10, size=len(dates)
        ),  # Temperature in °F
        "Operating_Time": np.random.randint(
            1, 25, size=len(dates)
        ),  # Operating time in hours
    }

    # Inject an anomaly in the Upper Thermostat values around the midpoint
    midpoint_index = len(dates) // 2
    anomaly_range = (midpoint_index - 5, midpoint_index + 5)

    # Create a spike in Upper Thermostat values
    data["Upper_Thermostat"][anomaly_range[0] : anomaly_range[1]] = np.random.normal(
        loc=200, scale=5, size=anomaly_range[1] - anomaly_range[0]
    )

    return pd.DataFrame(data, index=dates)


# Generate the dataset
start_date = datetime(2023, 10, 1)
end_date = datetime(2024, 10, 1)
dates = generate_date_range(start_date, end_date)

# Create a DataFrame with synthetic data
synthetic_dataset = generate_synthetic_data(dates)

now = datetime.now()

synthetic_dataset["time"] = [
    now - timedelta(hours=5 * i) for i in range(synthetic_dataset.shape[0])
]

# something whcky happened with the vector store. i don't know what the fuck happened.
# have to create a new assistant. 

# you need to have system instructions ilke this
# You are a helpful assistant and expert at ansewring building automation questions. Always carry out a file search for the desired information. You can augment that information with your general knowledge, but alwasy carry out a file seaach with every query first to see if the relevant information is there, and then add to that afterwards. 

# name : Building Energy and Efficiency Expert

# And also added repitiion of the instructions in the thread / run creation.

VECTOR_STORE_ID = os.environ["VECTOR_STORE_ID"] # will need to be updated. what the hell happened??
ASSISTANT_ID = os.environ["ASSISTANT_ID"]



# small medium offices are waht is relevant to this dataset.

# Initialize the client
client = openai.OpenAI()


# Step 2: Create a Thread
thread = client.beta.threads.create()
thread_id = thread.id


# Define the EventHandler class
class EventHandler(AssistantEventHandler):
    def __init__(self, response_queue):
        super().__init__()
        self.response_queue = response_queue

    @override
    def on_text_created(self, text) -> None:
        pass

    @override
    def on_text_delta(self, delta, snapshot):
        text = delta.value
        self.response_queue.put(text)
    
    @override
    def on_event(self, event):
      # Retrieve events that are denoted with 'requires_action'
      # since these will have our tool_calls
      if event.event == 'thread.run.requires_action':
        run_id = event.data.id  # Retrieve the run ID from the event data
        self.handle_requires_action(event.data, run_id)
 
    def handle_requires_action(self, data, run_id):
      tool_outputs = []
        
      for tool in data.required_action.submit_tool_outputs.tool_calls:
        if tool.function.name == "update_weather_forecast":
            print(tool.function.arguments)
            args = json.loads(tool.function.arguments)
            loc = args["location"]
            tool_outputs.append({"tool_call_id": tool.id, "output": update_weather_forecast(loc)})
        elif tool.function.name == "update_weather":
            print(tool.function.arguments)
            args = json.loads(tool.function.arguments)
            loc = args["location"]
            tool_outputs.append({"tool_call_id": tool.id, "output": update_weather(loc)})
        elif tool.function.name == "web_search":
            print(tool.function.arguments)
            args = json.loads(tool.function.arguments)
            query = args["query"]
            tool_outputs.append({"tool_call_id": tool.id, "output": web_search(query)})
        
      # Submit all tool_outputs at the same time
      self.submit_tool_outputs(tool_outputs, run_id)

    def submit_tool_outputs(self, tool_outputs, run_id):
        # Use the submit_tool_outputs_stream helper
        with client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self.current_run.thread_id,
            run_id=self.current_run.id,
            tool_outputs=tool_outputs,
            event_handler=EventHandler(self.response_queue),
        ) as stream:
            for text in stream.text_deltas:
                print(text, end="", flush=True)
                print()


def chat(usr_message, history):
    global thread_id
    # start_conversation()
    user_input = usr_message

    if not thread_id:
        print("Error: Missing thread_id")  # Debugging line
        return json.dumps({"error": "Missing thread_id"}), 400

    print(
        f"Received message: {user_input} for thread ID: {thread_id}"
    )  # Debugging line

    # Add the user's message to the thread
    client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=user_input
    )

    # Create a queue to hold the assistant's response chunks
    response_queue = queue.Queue()

    # Instantiate the event handler with the queue

    # Start the streaming run in a separate thread
    def run_stream():
        with client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID,
            tool_choice = "required",
            event_handler=EventHandler(response_queue),
        ) as stream:
            stream.until_done()

    stream_thread = Thread(target=run_stream)
    stream_thread.start()

    assistant_response = ""
    while True:
        try:
            # Get response chunks from the queue
            chunk = response_queue.get(timeout=0.1)
            assistant_response += chunk
            yield assistant_response
        except queue.Empty:
            # Check if the stream has finished
            if not stream_thread.is_alive():
                break

    # Wait for the stream thread to finish
    stream_thread.join()


def update_weather(location):
    api_key = os.environ["OPENWEATHERMAP_API_KEY"]
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "appid": api_key, "units": "imperial"}
    response = requests.get(base_url, params=params)
    weather_data = response.json()

    if response.status_code != 200:
        return f"Error fetching weather data: {weather_data.get('message', 'Unknown error')}"

    lon = weather_data["coord"]["lon"]
    lat = weather_data["coord"]["lat"]
    main = weather_data["weather"][0]["main"]
    feels_like = weather_data["main"]["feels_like"]
    temp_min = weather_data["main"]["temp_min"]
    temp_max = weather_data["main"]["temp_max"]
    pressure = weather_data["main"]["pressure"]
    visibility = weather_data["visibility"]
    wind_speed = weather_data["wind"]["speed"]
    wind_deg = weather_data["wind"]["deg"]
    sunrise = datetime.fromtimestamp(weather_data["sys"]["sunrise"]).strftime('%H:%M:%S')
    sunset = datetime.fromtimestamp(weather_data["sys"]["sunset"]).strftime('%H:%M:%S')
    temp = weather_data["main"]["temp"]
    humidity = weather_data["main"]["humidity"]
    condition = weather_data["weather"][0]["description"]

    return f"""**Weather in {location}:**
- **Coordinates:** (lon: {lon}, lat: {lat})
- **Temperature:** {temp:.2f}°F (Feels like: {feels_like:.2f}°F)
- **Min Temperature:** {temp_min:.2f}°F, **Max Temperature:** {temp_max:.2f}°F
- **Humidity:** {humidity}%
- **Condition:** {condition.capitalize()}
- **Pressure:** {pressure} hPa
- **Visibility:** {visibility} meters
- **Wind Speed:** {wind_speed} m/s, **Wind Direction:** {wind_deg}°
- **Sunrise:** {sunrise}, **Sunset:** {sunset}"""



def update_weather_forecast(location: str) -> str:
    """ Fetches the weather forecast for a given location and returns a formatted string
    Parameters:
    - location: the search term to find weather information
    Returns:
    A formatted string containing the weather forecast data
    """

    api_key = os.environ["OPENWEATHERMAP_API_KEY"]
    base_url = "http://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": location,
        "appid": api_key,
        "units": "imperial",
        "cnt": 40  # Request 40 data points (5 days * 8 three-hour periods)
    }
    response = requests.get(base_url, params=params)
    weather_data = response.json()
    if response.status_code != 200:
        return f"Error fetching weather data: {weather_data.get('message', 'Unknown error')}"

    # Organize forecast data per date
    forecast_data = {}
    for item in weather_data['list']:
        dt_txt = item['dt_txt']  # 'YYYY-MM-DD HH:MM:SS'
        date_str = dt_txt.split(' ')[0]  # 'YYYY-MM-DD'
        time_str = dt_txt.split(' ')[1]  # 'HH:MM:SS'
        forecast_data.setdefault(date_str, [])
        forecast_data[date_str].append({
            'time': time_str,
            'temp': item['main']['temp'],
            'feels_like': item['main']['feels_like'],
            'humidity': item['main']['humidity'],
            'pressure': item['main']['pressure'],
            'wind_speed': item['wind']['speed'],
            'wind_deg': item['wind']['deg'],
            'condition': item['weather'][0]['description'],
            'visibility': item.get('visibility', 'N/A'),  # sometimes visibility may be missing
        })

    # Process data to create daily summaries
    daily_summaries = {}
    for date_str, forecasts in forecast_data.items():
        temps = [f['temp'] for f in forecasts]
        feels_likes = [f['feels_like'] for f in forecasts]
        humidities = [f['humidity'] for f in forecasts]
        pressures = [f['pressure'] for f in forecasts]
        wind_speeds = [f['wind_speed'] for f in forecasts]
        conditions = [f['condition'] for f in forecasts]

        min_temp = min(temps)
        max_temp = max(temps)
        avg_temp = sum(temps) / len(temps)
        avg_feels_like = sum(feels_likes) / len(feels_likes)
        avg_humidity = sum(humidities) / len(humidities)
        avg_pressure = sum(pressures) / len(pressures)
        avg_wind_speed = sum(wind_speeds) / len(wind_speeds)

        # Find the most common weather condition
        condition_counts = Counter(conditions)
        most_common_condition = condition_counts.most_common(1)[0][0]

        daily_summaries[date_str] = {
            'min_temp': min_temp,
            'max_temp': max_temp,
            'avg_temp': avg_temp,
            'avg_feels_like': avg_feels_like,
            'avg_humidity': avg_humidity,
            'avg_pressure': avg_pressure,
            'avg_wind_speed': avg_wind_speed,
            'condition': most_common_condition,
        }

    # Build the formatted string
    city_name = weather_data['city']['name']
    ret_str = f"**5-Day Weather Forecast for {city_name}:**\n"

    for date_str in sorted(daily_summaries.keys()):
        summary = daily_summaries[date_str]
        ret_str += f"\n**{date_str}:**\n"
        ret_str += f"- **Condition:** {summary['condition'].capitalize()}\n"
        ret_str += f"- **Min Temperature:** {summary['min_temp']:.2f}°F\n"
        ret_str += f"- **Max Temperature:** {summary['max_temp']:.2f}°F\n"
        ret_str += f"- **Average Temperature:** {summary['avg_temp']:.2f}°F (Feels like {summary['avg_feels_like']:.2f}°F)\n"
        ret_str += f"- **Humidity:** {summary['avg_humidity']:.0f}%\n"
        ret_str += f"- **Pressure:** {summary['avg_pressure']:.0f} hPa\n"
        ret_str += f"- **Wind Speed:** {summary['avg_wind_speed']:.2f} m/s\n"

    return ret_str


llmmodel = OpenAI(api_token=os.environ["OPENAI_API_KEY"], model='gpt-4o')

# Load dataframes
dfcleaned = pd.read_csv("dfcleaned.csv")
dfcleaned['Timestamp'] = pd.to_datetime(dfcleaned['Timestamp'])
dfcleaned['off-nominal'] = dfcleaned['off-nominal'].apply(str)
dfshaps = pd.read_csv("shaps.csv")
dfshaps['Timestamp'] = pd.to_datetime(dfshaps['Timestamp'])

# Initialize Agent
agent = Agent([dfcleaned, dfshaps], config={"llm": llmmodel})

sdfshaps = SmartDataframe(dfshaps, config={"llm": llmmodel})
sdfcleaned = SmartDataframe(dfcleaned, config={"llm": llmmodel})



def process_query(query):
    response = agent.chat(query)  # Replace with your actual agent chat implementation
    print(response)
    
    # Initialize outputs and visibility flags
    text_output = None
    image_output = None
    dataframe_output = None
    text_visible = False
    image_visible = False
    dataframe_visible = False
    
    if isinstance(response, str) and ".png" not in response:
        text_output = response
        text_visible = True
    elif isinstance(response, str) and ".png" in response:
            image_output = response  # Assuming response is a filepath or URL to the image
            image_visible = True
    elif isinstance(response, pd.DataFrame):
        dataframe_output = response
        dataframe_visible = True
    
    return (
        text_output,
        image_output,
        dataframe_output,
        gr.update(visible=text_visible),
        gr.update(visible=image_visible),
        gr.update(visible=dataframe_visible)
    )



def gradio_app():
    iface = gr.Interface(
        fn=process_query,
        inputs="text",
        outputs=[
            gr.Textbox(label="Response"),
            gr.Image(label="Plot"),
            gr.DataFrame(label="Dataframe")
        ],
        title="pandasai Query Processor",
        description="Enter your query related to the csv data files."
    )
    return iface

with gr.Blocks(
 #   theme=gr.themes.Monochrome(primary_hue="green"), 
    theme = gr.themes.Soft(),
) as demo:
    with gr.Row():  # Combine the two weather functions into a single row
        with gr.Column():
            location1 = gr.Textbox(label="Enter location for weather (e.g., Rio Rancho, New Mexico)",
                                  value="Cambridge, Massachusetts")
            weather_button = gr.Button("Get Weather")
         #   output1 = gr.Markdown(label="Weather Information")
            output1 = gr.Textbox(label="Weather Information", lines=8, max_lines=8, show_label=True, show_copy_button=True)
            weather_button.click(
                fn=update_weather,
                inputs=location1,
                outputs=output1,
                api_name="update_weather",
            )
        with gr.Column():
            location2 = gr.Textbox(label="Enter location for weather forecast (e.g., Rio Rancho, New Mexico)",
                                  value="Cambridge, Massachusetts")
            weather_forecast_button = gr.Button("Get 5-Day Weather Forecast")
          #  output2 = gr.Markdown(label="Weather Forecast Information")
            output2 = gr.Textbox(label="Weather 5-Day Forecast Information", lines=8, max_lines=8,
                                show_label=True, show_copy_button=True)
            weather_forecast_button.click(
                fn=update_weather_forecast,
                inputs=location2,
                outputs=output2,
                api_name="update_weather_forecast",
            )
    gr.Markdown("# 📄 PDF Viewer Section")
    gr.Markdown("Select a PDF from the dropdown below to view it.")
    
    with gr.Accordion("Open PDF Selection", open=False):
        with gr.Row():
            # Assign a larger scale to the dropdown
            dropdown = gr.Dropdown(
                choices=pdf_files,
                label="Select a PDF",
                value=DEFAULT_PDF,  # Set a default value
                scale=1  # This component takes twice the space
            )
            # Assign a smaller scale to the PDF viewer
            pdf_viewer = PDF(
                label="PDF Viewer",
                interactive=True,
                scale=3 ,
                value=display_pdf(DEFAULT_PDF)# This component takes half the space compared to dropdown
            )
    
        # Set up the event: when dropdown changes, update the PDF viewer
        dropdown.change(
            fn=display_pdf,
            inputs=dropdown,
            outputs=pdf_viewer
        )
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# Building Automation Assistant")

            gr.Markdown(
                "I'm an AI assistant that can help with building maintenance and equipment questions."
            )

            gr.Markdown("---")

            # Update the ChatInterface to handle streaming
            chat_interface = gr.ChatInterface(
                chat,
                #show_label=True,
              #  show_copy_button=True,
                chatbot=gr.Chatbot(height=750, show_copy_button=True, show_copy_all_button=True,
                                    avatar_images=("user_avatar.png", "assistant_avatar.png")),
                title="Ask Me Anything",
                examples_per_page= 5,
             #   theme="soft", # glass
                description="Type your question about building automation here.",
                examples=[
                    "Give the weather forecast for Cambridge, MA",
                    "Give me the weather forecast for New York, NY. express the windspeed in miles per hour.",
                    "list the authors on the academic paper associated with the homezero project.",
                    "What are some good API services that i can use to help fetch relevant data for building automation purposes? include hyperlinks in your response.",
                    "show the first few rows of each of the uploaded csv files",
                    "What are the current maintenance protocols for HouseZero?",
                    "How do the maintenance protocols for HouseZero compare to industry best practices?",
                    "What are the most common maintenance challenges faced by net-zero energy buildings?",
                    "How does the Uponor Climate Control Network System contribute to building maintenance?",
                    "What role do smart systems play in the maintenance of sustainable buildings like HouseZero?",
                    "Can you provide data on the energy performance of HouseZero over the past year?",                    
                     "Tell me about the HouseZero dataset. Retrieve information from the publication you have access to. Use your file retrieval tool.",
                    "Describe in detail the relationshp between the columns and values in the uploaded CSV files and the information you have access to regarding the HouseZero dataset. Be verbose. Use your file retrieval tool.",
                    "Please comment on the zone relative humidity features, specifically if they indicate a problem withthe building",
                    "Give me in great detail any advice you have to maintain a small to midsize office building, like the HouseZero data corresponds to. Be verbose. Use your file retrieval tool.",
                    "Is there any information in the datafiles that indicates a problem with the building?",
                    "Show Massachusetts electricity billing rates during the same time span as the CSV data",
                    "Use those rates and the relevant columns in the CSV files to estimate how much it costs to operate this building per month.",
                     "What is the estimated average electricity cost for operating the building using massachusetts energy rates. use your file retrieval tool. use data csv files for building data. Limit your response to 140 characters. Use your file retrieval tool.",
                    "Based on the data in these CSV files, can you assign an EnergyIQ score from 1-10 that reflects how well the building is operating? Explain the reason for your score and provide any recommendations on actions to take that can improve it in the future. Be verbose. Use your file retrieval tool.",
                    "Please summarize information concerning sensor networks that may be leading to faulty meaurements.",
                    "Tell me how to properly install the PVC sky lights.",
                    "Based on data and insights, what specific changes should be made to HouseZero's maintenance protocols?",
                    "what recommendations do you have to mitigate against high relative humidity zone measurements in structures like the housezero building?"
                ],
                fill_height=True,
            )

            gr.Markdown("---")
    with gr.Accordion("Example Plots Section", open=False):
        with gr.Column():
            #    with gr.Column():
            # Define the three ScatterPlot components
            anomaly_plot = gr.ScatterPlot(
                dfcleaned, 
                x="Timestamp", 
                y="Z5_RH", 
                color="off-nominal",
                title="Anomaly Score"
            )
        
            zone3_plot = gr.ScatterPlot(
                dfcleaned,
                x="Timestamp",
                y="Z3_RH",
                color="off-nominal",
                title="Zone 3 Relative Humidity",
            )

            zone4_plot = gr.ScatterPlot(
                dfcleaned,
                x="Timestamp",
                y="Z4_RH",
                color="off-nominal",
                title="Zone 4 Relative Humidity",
            )
    
    # Group all plots into a list for easy management
            plots = [anomaly_plot, zone3_plot, zone4_plot]

            def select_region(selection: gr.SelectData):
                """
                Handles the region selection event.

                Args:
                selection (gr.SelectData): The data from the selection event.

                Returns:
                List[gr.Plot.update]: A list of update instructions for each plot.
                """
                if selection is None or selection.index is None:
                    return [gr.Plot.update() for _ in plots]
        
                min_x, max_x = selection.index
        # Update the x_lim for each plot
                return [gr.ScatterPlot(x_lim=(min_x, max_x)) for _ in plots]

            def reset_region():
                """
                Resets the x-axis limits for all plots.

                Returns:
                    List[gr.Plot.update]: A list of update instructions to reset x_lim.
                """
                return [gr.ScatterPlot(x_lim=None) for _ in plots]

    # Attach event listeners to each plot
            for plot in plots:
                plot.select(
                    select_region, 
                    inputs=None, 
                    outputs=plots  # Update all plots
                )
                plot.double_click(
                    reset_region, 
                    inputs=None, 
                    outputs=plots  # Reset all plots
                )

           # plots = [plt, first_plot, second_plot]

           # def select_region(selection: gr.SelectData):
           #     min_w, max_w = selection.index
           #     return gr.ScatterPlot(x_lim=(min_w, max_w)) 

           # for p in plots:
           #     p.select(select_region, None, plots)
           #     p.double_click(lambda: [gr.LinePlot(x_lim=None)] * len(plots), None, plots)
        
       # second_plot.select(select_second_region, None, plt)
       # second_plot.double_click(lambda: gr.ScatterPlot(x_lim=None), None, plt)
      #  gr.Column([anomaly_plot, first_plot, second_plot])

       # anomaly_info = gr.Markdown("Anomaly detected around October 15, 2023")
    with gr.Column():
        query = gr.Textbox(label="Enter your question about the data",
                           value="Plot the anomaly_score as a function of time and highlight the highest 20 values")
        query_button = gr.Button("Submit Data Query")
        with gr.Row():
            with gr.Column(visible=False) as output_col1:
                out1 = gr.Textbox(label="Response")
            with gr.Column(visible=False) as output_col2:
                out2 = gr.Image(label="Plot")
            with gr.Column(visible=False) as output_col3:
                out3 = gr.DataFrame(label="DataFrame")
        query_button.click(
            fn=process_query,
            inputs=query,
            outputs=[
                out1,        # Text output
                out2,        # Image output
                out3,        # DataFrame output
                output_col1, # Visibility for Text output
                output_col2, # Visibility for Image output
                output_col3  # Visibility for DataFrame output
            ],
            api_name="process_query"
        )
       
    # hide visibility until its ready
    
        
    # Weather input
  #  with gr.Row():
  #      iface = gradio_app()


demo.launch(share=True)