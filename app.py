import streamlit as st
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.chains import create_tagging_chain_pydantic
from pydantic import BaseModel, Field
import pandas as pd
import os

# Set your OpenAI API key (Replace with your actual key)
os.environ["OPENAI_API_KEY"] = "..."
# Streamlit app setup
st.title("DATA ANALYSIS")

# Upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Sidebar for selecting the interaction mode
interaction_mode = st.sidebar.radio("Select Interaction Mode", ("Chat for Data Analysis", "Visualize Data"))

if interaction_mode == "Chat for Data Analysis":
    st.title('Analyze Data with AI')

    if uploaded_file is not None:
        # Read CSV into DataFrame
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("The uploaded CSV file is empty. Please upload a valid file.")
        else:
            # Try to create an agent with the DataFrame
            try:
                agent = create_pandas_dataframe_agent(
                    ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
                    df,
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    allow_dangerous_code=True  # Opt-in for dangerous code execution
                )

                # Get user input/question
                user_input = st.text_input("Ask a question about the dataset:")

                if st.button("Ask"):
                    response = agent.run(user_input)
                    st.write("AI's response:")
                    st.write(response)

            except ValueError as e:
                st.error(f"An error occurred while creating the agent: {e}")
    else:
        st.info("Please upload a CSV file to get started.")



elif interaction_mode == "Visualize Data":
    st.title('Visualize data with AI')

    def show_chart(chart_type, column, dataframe):
        """
        Display the selected chart type for the given column in the DataFrame.
        """
        if column not in dataframe.columns:
            st.error(f"Column '{column}' not found in the dataset.")
            return

        if chart_type == "bar_chart":
            st.bar_chart(dataframe[column].value_counts())
        elif chart_type == "area_chart":
            st.area_chart(dataframe[column])
        elif chart_type == "line_chart":
            try:
                dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')
                dataframe = dataframe.dropna(subset=[column])
                st.line_chart(dataframe[column])
            except Exception as e:
                st.error(f"An error occurred while generating the line chart: {e}")
        else:
            st.error("Unsupported chart type.")

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file into a DataFrame
            df_now = pd.read_csv(uploaded_file)

            if df_now.empty:
                st.error("The uploaded CSV file is empty. Please upload a valid dataset.")
            else:
                # Display the DataFrame
                st.dataframe(df_now, use_container_width=True)

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
                llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

                # Create a tagging chain for data feature selection
                tagging_chain = create_tagging_chain_pydantic(DataFeature, llm)

                try:
                    # Run the tagging chain to obtain chart type and column information
                    res = tagging_chain.run(text_input)

                    # Display the selected chart
                    show_chart(res.chartType, res.column, df_now)
                except Exception as e:
                    st.error(f"An error occurred while generating the visualization: {e}")
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")
    else:
        st.info("Please upload a CSV dataset to get started.")