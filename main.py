import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from streamlit_option_menu import option_menu
import data_analysis_functions as function
import data_preprocessing_function as preprocessing_function
import home_page

# Set page configuration
st.set_page_config(page_icon="✨", page_title="AutoEDA")

# Hide Streamlit branding
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Sidebar and title setup
st.sidebar.title("AutoEDA: Automated Exploratory Data Analysis and Processing")
st.title("Welcome to AutoEDA")

selected = option_menu(
    menu_title=None,
    options=['Home', 'Data Exploration', 'Data Preprocessing'],
    icons=['house-heart', 'bar-chart-fill', 'hammer'],
    orientation='horizontal'
)

if selected == 'Home':
    home_page.show_home_page()

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload Your File Here", type=["csv", "xls", "xlsx", "json"])
use_example_data = st.sidebar.checkbox("Use Example Titanic Dataset", value=False)

if uploaded_file:
    # Determine the file type and load data accordingly
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_extension in ["xls", "xlsx"]:
        df = pd.read_excel(uploaded_file)
    elif file_extension == "json":
        df = pd.read_json(uploaded_file)
    else:
        st.error("Unsupported file format!")
        df = None

    if df is not None:
        if 'new_df' not in st.session_state:
            st.session_state.new_df = df.copy()

elif use_example_data:
    # Load the example dataset
    df = pd.read_csv("example_dataset/titanic.csv")
    if 'new_df' not in st.session_state:
        st.session_state.new_df = df

else:
    df = None

if df is None and selected != 'Home' and not use_example_data:
    st.markdown("#### Use the sidebar to upload a file.")
else:
    if selected == 'Data Exploration':
        tab1, tab2 = st.tabs(['📊 Dataset Overview', "🔎 Data Exploration and Visualization"])
        num_columns, cat_columns = function.categorical_numerical(df)

        with tab1:
            st.subheader("Dataset Overview")
            function.display_dataset_overview(df, cat_columns, num_columns)
            function.display_missing_values(df)
            function.display_statistics_visualization(df, cat_columns, num_columns)
            function.display_data_types(df)
            function.search_column(df)

        with tab2:
            function.display_individual_feature_distribution(df, num_columns)
            function.display_scatter_plot_of_two_numeric_features(df, num_columns)

            if len(cat_columns) != 0:
                function.categorical_variable_analysis(df, cat_columns)
            else:
                st.info("The dataset does not have any categorical columns")

            if len(num_columns) != 0:
                function.feature_exploration_numerical_variables(df, num_columns)
            else:
                st.warning("The dataset does not contain any numerical variables")

            if len(num_columns) != 0 and len(cat_columns) != 0:
                function.categorical_numerical_variable_analysis(df, cat_columns, num_columns)
            else:
                st.warning("The dataset does not allow for categorical and numerical variable analysis.")

    if selected == 'Data Preprocessing':
        revert = st.button("Revert to Original Dataset", key="revert_button")
        if revert:
            st.session_state.new_df = df.copy()

        # Remove unwanted columns
        st.subheader("Remove Unwanted Columns")
        columns_to_remove = st.multiselect(label='Select Columns to Remove', options=st.session_state.new_df.columns)
        if st.button("Remove Selected Columns"):
            if columns_to_remove:
                st.session_state.new_df = preprocessing_function.remove_selected_columns(st.session_state.new_df, columns_to_remove)
                st.success("Selected columns removed successfully.")

        st.dataframe(st.session_state.new_df)

        # Handle missing data
        st.subheader("Handle Missing Data")
        missing_count = st.session_state.new_df.isnull().sum()
        if missing_count.any():
            selected_missing_option = st.selectbox(
                "Select how to handle missing data:",
                ["Remove Rows in Selected Columns", "Fill Missing Data in Selected Columns (Numerical Only)"]
            )
            if selected_missing_option == "Remove Rows in Selected Columns":
                columns_to_remove_missing = st.multiselect("Select columns to remove rows with missing data",
                                                            options=st.session_state.new_df.columns)
                if st.button("Remove Rows with Missing Data"):
                    st.session_state.new_df = preprocessing_function.remove_rows_with_missing_data(
                        st.session_state.new_df, columns_to_remove_missing)
                    st.success("Rows with missing data removed successfully.")
            elif selected_missing_option == "Fill Missing Data in Selected Columns (Numerical Only)":
                numerical_columns_to_fill = st.multiselect(
                    "Select numerical columns to fill missing data",
                    options=st.session_state.new_df.select_dtypes(include=['number']).columns)
                fill_method = st.selectbox("Select fill method:", ["mean", "median", "mode"])
                if st.button("Fill Missing Data"):
                    if numerical_columns_to_fill:
                        st.session_state.new_df = preprocessing_function.fill_missing_data(
                            st.session_state.new_df, numerical_columns_to_fill, fill_method)
                        st.success(f"Missing data in numerical columns filled with {fill_method} successfully.")
                    else:
                        st.warning("Please select a column to fill in the missing data")
            function.display_missing_values(st.session_state.new_df)
        else:
            st.info("The dataset does not contain any missing values")

        # Feature scaling
        st.subheader("Feature Scaling")
        new_df_numerical_columns = st.session_state.new_df.select_dtypes(include=['number']).columns
        selected_columns = st.multiselect("Select Numerical Columns to Scale", new_df_numerical_columns)
        scaling_method = st.selectbox("Select Scaling Method:", ['Standardization', 'Min-Max Scaling'])
        if st.button("Apply Scaling"):
            if selected_columns:
                if scaling_method == "Standardization":
                    st.session_state.new_df = preprocessing_function.standard_scale(st.session_state.new_df, selected_columns)
                    st.success("Standardization Applied Successfully.")
                elif scaling_method == "Min-Max Scaling":
                    st.session_state.new_df = preprocessing_function.min_max_scale(st.session_state.new_df, selected_columns)
                    st.success("Min-Max Scaling Applied Successfully.")
            else:
                st.warning("Please select numerical columns to scale.")

        st.dataframe(st.session_state.new_df)

        # Outlier handling
        st.subheader("Identify and Handle Outliers")
        selected_numeric_column = st.selectbox("Select Numeric Column for Outlier Handling:", new_df_numerical_columns)
        fig, ax = plt.subplots()
        sns.boxplot(data=st.session_state.new_df, x=selected_numeric_column, ax=ax)
        st.pyplot(fig)

        outliers = preprocessing_function.detect_outliers_zscore(st.session_state.new_df, selected_numeric_column)
        if outliers:
            st.warning("Detected Outliers:")
            st.write(outliers)
        else:
            st.info("No outliers detected.")

        outlier_handling_method = st.selectbox("Select Outlier Handling Method:", ["Remove Outliers", "Transform Outliers"])
        if st.button("Apply Outlier Handling"):
            if outlier_handling_method == "Remove Outliers":
                st.session_state.new_df = preprocessing_function.remove_outliers(
                    st.session_state.new_df, selected_numeric_column, outliers)
                st.success("Outliers removed successfully.")
            elif outlier_handling_method == "Transform Outliers":
                st.session_state.new_df = preprocessing_function.transform_outliers(
                    st.session_state.new_df, selected_numeric_column, outliers)
                st.success("Outliers transformed successfully.")

        # Download processed data
        if st.session_state.new_df is not None:
            csv = st.session_state.new_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'data:file/csv;base64,{b64}'
            st.markdown(f'<a href="{href}" download="preprocessed_data.csv"><button>Download Preprocessed Data</button></a>', unsafe_allow_html=True)
        else:
            st.warning("No preprocessed data available to download.")