import streamlit as st
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import pandas as pd
import os
from langchain.chains import create_tagging_chain_pydantic
from enum import Enum
from pydantic import BaseModel, Field

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = '{personal OPENAI key}'

# Set page title, icon, and layout
st.set_page_config(page_title="Data Sherlock", page_icon="🔍", layout="wide")

# Apply custom CSS for background color
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
        font-family: Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a Streamlit app
st.title("DATA SHERLOCK")

# Sidebar for selecting the interaction mode
interaction_mode = st.sidebar.radio("Select Interaction Mode", ("Chat for Data Analysis", "Visualize Data"))

if interaction_mode == "Chat for Data Analysis":
    st.title('Analyze Data with AI')
    # Upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Create an agent with the DataFrame
        agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model=""),  
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            language="vi" 
        )

        # Get user input/question
        user_input = st.text_input("Ask a question about the dataset:")

        if st.button("Ask"):
            # Use the agent to respond to the user's question
            response = agent.run(user_input)  # Provide a question related to the dataset
            st.write("AI's response:")
            st.write(response)

elif interaction_mode == "Visualize Data":
    st.title('Visualize data with AI')
    def show_chart(chartType, column, df):
        """
        Display the selected chart type for the given column in the DataFrame.
        """
        if chartType == "bar_chart":
            st.bar_chart(df, x=column)
        elif chartType == "area_chart":
            st.area_chart(df, x=column)
        elif chartType == "line_chart":
            st.line_chart(df, x=column)

    # Initialize variables
    df_now = None

    # Allow the user to upload a CSV dataset
    uploaded_file = st.file_uploader("Choose a dataset file (CSV)", type=['csv'])

    if uploaded_file is not None:
        # Read the uploaded CSV file into a DataFrame
        df_now = pd.read_csv(uploaded_file)

        # Display the DataFrame
        st.dataframe(df_now, use_container_width=True)
    else:
        # Display instructions to upload a dataset
        st.write("Please upload a CSV dataset to get started.")

    # Check if a dataset is available for processing
    if df_now is not None:
        # Convert the DataFrame to a string for processing with the language model
        text_input = df_now.to_string()

        class DataFeature(BaseModel):
            chartType: str = Field(
                ...,
                enum=["bar_chart", "area_chart", "line_chart"],
                description="Choose 'bar_chart' for categorical data, 'area_chart' for monthly/daily data, or 'line_chart' for time-series data."
            )
            column: str = Field(..., description="Enter the column name for the x-axis.")

        # Initialize the OpenAI language model
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

        # Create a tagging chain for data feature selection
        tagging_chain = create_tagging_chain_pydantic(DataFeature, llm)

        # Run the tagging chain to obtain chart type and column information
        res = tagging_chain.run(text_input)

        # Display the selected chart
        show_chart(res.chartType, res.column, df_now)
