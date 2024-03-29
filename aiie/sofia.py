import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.figure_factory as ff
st.set_page_config(layout="wide")

# Load the dataset
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("processed_dataset.csv").dropna(how="all")
    return df

df = load_data()
df_processed = df.copy()

# Function to clean, split, and normalize column values
def clean_split_normalize(values):
    if pd.isna(values):
        return []
    return [value.strip().lower() for value in re.split(';|,', values) if value.strip()]

# Preprocess 'Technology(ies)', 'Sector(s)', and 'Issue(s)' columns
def preprocess_column(df, column_name, prefix, merge_rename_operations={}):
    df[column_name] = df[column_name].str.lstrip()
    unique_set = set()
    df[column_name].apply(lambda x: [unique_set.add(item) for item in clean_split_normalize(x)])
    encoding_dict = {f'{prefix}_{item}': [] for item in unique_set}
    for _, row in df.iterrows():
        items = clean_split_normalize(row[column_name])
        for item in unique_set:
            encoding_dict[f'{prefix}_{item}'].append(int(item in items))
    encoded_df = pd.DataFrame(encoding_dict)
    for new_name, old_names in merge_rename_operations.items():
        encoded_df[new_name] = encoded_df[old_names].max(axis=1)
        encoded_df.drop(columns=old_names, inplace=True, errors='ignore')
    processed_df = pd.concat([df.drop(columns=[column_name]), encoded_df], axis=1)
    return processed_df, unique_set



merge_rename_operations_technology = {
    'tech_unclear/unknown': ['tech_unclear/unknown', 'tech_unknown'],
    'tech_facial recognition/detection/identification': ['tech_facial recognition', 'tech_facial detection', 'tech_facial recogniton', 'tech_facial detection'],
    'tech_(performance) scoring algorithm': ['tech_performance scoring algorithm', 'tech_scoring algorithm'],
    'tech_location analytics': ['tech_location tracking', 'tech_location recognition',  'tech_location analytics'],
    'tech_social media (monitoring)': ['tech_social media monitoring', 'tech_social media'],
    'tech_emotion recognition/detection': ['tech_emotion recognition', 'tech_emotion detection'],
    'tech_neural networks': ['tech_neural networks', 'tech_neural network'],
    'tech_image generation': ['tech_image generator', 'tech_image generation'],
    'tech_large language model (llm)': ['tech_large language model', 'tech_large language model (llm)'],
    'tech_speech/voice recognition': ['tech_speech/voice recognition', 'tech_speech ecognition', 'tech_speech recognition', 'tech_voice recognition'],
    'tech_image recognition/filtering': ['tech_image recognition/filtering', 'tech_image recognition'],
    'tech_vehicle detection': ['tech_vehicle detection', 'tech_vehicle detection system'],
    'tech_object recognition/detection/identification': ['tech_object identification', 'tech_object recognition', 'tech_object detection'],
    'tech_voice generation/synthesis': ['tech_voice synthesis', 'tech_voice generation'],
    'tech_gaze recognition/detection': ['tech_gaze detection', 'tech_gaze recognition'],
    'tech_text-to-speech': ['tech_text-to-speech', 'tech_text to speech'],
    'tech_virtual reality (vr)': ['tech_virtual reality (vr)', 'tech_virtual reality'],
    'tech_behavioural analysis': ['tech_behavioural monitoring', 'tech_behavioural monitoring system', 'tech_behavioural analysis',],
    'tech_predictive (statistical) analytics': ['tech_predictive statistical analysis', 'tech_predictive analytics'],
    'tech_content management/moderation system': ['tech_content moderation system', 'tech_content management system'],
    'tech_deepfake - audio': [ 'tech_deepfake - audio', 'tech_audio'],
    'tech_deepfake - image': [ 'tech_deepfake - image',  'tech_image', ],
    'tech_deepfake - video': [ 'tech_deepfake - video',  'tech_video'],
    'tech_gesture analysis': ['tech_gesture analysis', 'tech_gesture recognition',  'tech_smile recognition'],
    'tech_pricing algorithm': ['tech_pricing algorithm', 'tech_price adjustment algorithm', 'tech_pricing automation'],
    'tech_facial analysis': ['tech_facial analysis', 'tech_facial matching',  'tech_facial scanning'],
    'tech_fingerprint analysis': [ 'tech_fingerprint biometrics', 'tech_fingerprint scanning', 'tech_fingerprint recognition'],
    'tech_risk assessment algorithm/system': [ 'tech_risk assessment algorithm',  'tech_risk assessment/classification algorithm', 'tech_recidivism risk assessment system',  'tech_automated risk assessment'],
    'tech_scheduling algorithm/software': [ 'tech_scheduling algorithm', 'tech_crew scheduling software'],
}

# Merge and rename specific columns as requested for 'Technology(ies)'
merge_rename_operations_sector = {
    'sector_sector_real estate sales/management': ['sector_real estate sales/management',  'sector_real estate'],
    'sector_govt - health': [ 'sector_gov - health', 'sector_govt - health'],
    'sector_business/professional services': [ 'sector_professional/business services',  'sector_business/professional services'],
    'sector_govt - police': [ 'sector_govt - police', 'sector_police'],
    'sector_govt - agriculture': [ 'sector_govt - agriculture', 'sector_agriculture'],
    'sector_private - individual': [ 'sector_private - individual', 'sector_private'],
    'sector_banking/financial services': [ 'sector_banking/financial services', 'sector_govt - finance'],
    'sector_education': [ 'sector_education', 'sector_govt - education'],
    'sector_telecoms': ['sector_telecoms',  'sector_govt - telecoms']
}

merge_rename_operations_issue = {
    'issue_bias/discrimination - lgbtqi+': ['issue_bias/discrimination - transgender',  'issue_transgender',  'issue_bias/discrimination - sexual preference (lgbtq)', 'issue_bias/discrimination - lgbtq', 'issue_lgbtq',],
    'issue_necessity/proportionality': [ 'issue_necessity/proportionality', 'issue_proportionality'],
    'issue_bias/discrimination - race/ethnicity': [ 'issue_bias/discimination - race', 'issue_race','issue_bias/disrimination - race','issue_ethnicity','issue_bias/discrimination - racial', 'issue_bias/discrimination - ethnicity',  'issue_bias/discrimination - race',  'issue_bias/disrimination - ethnicity',],
    'issue_bias/discrimination - political': ['issue_bias/discrimination - politics', 'issue_bias/discrimination - political',  'issue_political'],
    'issue_mis/dis-information': ['issue_mis-disinformation',  'issue_mis/dsinformation', 'issue_mis/disinformation'],
    'issue_autonomous lethal weapons': ['issue_autonomous lethal weapons', 'issue_lethal autonomous weapons'],
    'issue_governance/accountability - capability/capacity': [ 'issue_capability/capacity','issue_governance/accountability - capability/capacity',  'issue_governance/accountability', 'issue_governance - capability/capacity'],
    'issue_ethics/values': ['issue_ethics', 'issue_ethics/values'],
    'issue_ownership/accountability': [ 'issue_ownership/accountability',  'issue_accountability'],
    'issue_ip/copyright': ['issue_copyright','issue_ip/copyright'],
    'issue_reputational damage': [ 'issue_reputation', 'issue_reputational damage'],
    'issue_bias/discrimination - employment/income': [ 'issue_bias/discrimination - employment', 'issue_employment', 'issue_bias/discrimination - income', 'issue_income', 'issue_bias/discrimination - profession/job'],
    'issue_legal - liability': [ 'issue_legal - liability',  'issue_liability'],
    'issue_identity theft/impersonation': [ 'issue_identity theft/impersonation', 'issue_impersonation'],
    'issue_bias/discrimination - disability': ['issue_bias/discrimination - disability', 'issue_disability'],
    'issue_bias/discrimination - economic': [ 'issue_bias/discrimination - economic',  'issue_economic'],
    'issue_bias/discrimination - political opinion/persuasion': [ 'issue_bias/discrimination - political opinion','issue_bias/discrimination - political persuasion'],
    'issue_nationality': ['issue_national origin', 'issue_nationality',  'issue_national identity'],
    'issue_employment - pay/compensation': ['issue_pay', 'issue_employment - pay',  'issue_employment - pay/compensation'],
    'issue_corruption/fraud': ['issue_fraud', 'issue_corruption/fraud', 'issue_legal - fraud', 'issue_safety - fraud'],
    'issue_bias/discrimination - gender': [ 'issue_bias/discrimination - gender',  'issue_gender'],
    'issue_accuracy/reliabiity': [ 'issue_accuracy/reliabiity', 'issue_accuracy/reliability',  'issue_accuracy/reliabilty', 'issue_accuracy/reliablity',  'issue_accuray/reliability'],
    'issue_bias/discrimination - body size/weight': ['issue_size',  'issue_body size', 'issue_weight',],
    'issue_bias/discrimination - location': [ 'issue_bias/discrimination - location', 'issue_location'],
    'issue_employment - unionisation': [ 'issue_unionisation', 'issue_employment - unionisation'],
    'issue_employment - jobs': ['issue_employment - jobs', 'issue_jobs'],
    'issue_bias/discrimination - religion': [ 'issue_bias/discrimination - religion', 'issue_religion'],
    'issue_employment - health & safety': [ 'issue_employment - health & safety',  'issue_employment - safety'],
    'issue_bias/discrimination - age': [ 'issue_bias/discrimination - age', 'issue_age'],
    'issue_privacy - consent': ['issue_privacy - consent', 'issue_privacy'],
    'issue_value/effectiveness': ['issue_value/effectiveness', 'issue_effectiveness/value'],
    'issue_oversight/review': ['issue_oversight','issue_oversight/review'],
    'issue_legal - defamation/libel': ['issue_legal - defamation/libel',  'issue_defamation'],
    'issue_misleading marketing': ['issue_misleading marketing','issue_misleading'],
    'issue_bias/discrimination - education':['issue_education'],
    'issue_employment - termination': ['issue_employment - termination','issue_termination'],
    'issue_anthropomorphism': [ 'issue_robot rights','issue_anthropomorphism'],
    'issue_humanrights_freedom' :['issue_freedom of expression - right of assembly',  'issue_freedom of expression - censorship',  'issue_freedom of expression', 'issue_freedom of information'],

}

merge_rename_operations_transparency = {
    'transp_black box': [ 'transp_back box', 'transp_black box','transp_governance: black box'],
    'transp_legal': ['transp_legal', 'transp_legal - mediation', 'transp_legal - foi request blocks'],
    'transp_marketing: privacy': ['transp_marketing: privacy', 'transp_marketing privacy'],
    'transp_privacy/consent': ['transp_privacy - consent', 'transp_consent', 'transp_privacy'],
    'transp_complaints/appeals': [ 'transp_complaints/appeals', 'transp_complaints & appeals','transp_appeals/complaints']


}
# Function to get the top N or all columns based on frequency
def get_top_columns(df, columns, top_n):
    frequency = df[columns].sum().sort_values(ascending=False)
    if top_n != 'All':
        return frequency.head(top_n).index.tolist()
    return frequency.index.tolist()


# Streamlit app layout
st.title('Data Exploration Heatmaps')



# Replace the slider with a dropdown for selecting the number of top frequencies
top_n_options = ["Top 10", "Top 20", "Top 50", "All"]
frequency_option = st.selectbox("Select the number of top frequencies to display:", options=top_n_options)

# Mapping selected option to a value
top_n_values = {'Top 10': 10, 'Top 20': 20, 'Top 50': 50, 'All': 'All'}
top_n = top_n_values[frequency_option]

# Continue with your logic using the selected `top_n` value


df_processed, technology_set = preprocess_column(df, 'Technology(ies)', 'tech', merge_rename_operations_technology)
df_processed, sector_set = preprocess_column(df_processed, 'Sector(s)', 'sector', merge_rename_operations_sector)
df_processed, issue_set = preprocess_column(df_processed, 'Issue(s)', 'issue', merge_rename_operations_issue)
df_processed, transparency_set = preprocess_column(df_processed, 'Transparency', 'transp', merge_rename_operations_transparency)

# Extracting technology and issue columns with dynamic top N filtering
tech_columns = get_top_columns(df_processed, [col for col in df_processed.columns if col.startswith('tech_')], top_n)
issue_columns = get_top_columns(df_processed, [col for col in df_processed.columns if col.startswith('issue_')], top_n)
sector_columns = get_top_columns(df_processed, [col for col in df_processed.columns if col.startswith('sector_')], top_n)
transp_columns = get_top_columns(df_processed, [col for col in df_processed.columns if col.startswith('transp_')], top_n)



def generate_interactive_heatmap(df_processed, index_columns, column_columns, title):
    # Initialize the occurrence matrix with zeros
    occurrence_matrix = pd.DataFrame(0, index=index_columns, columns=column_columns)

    # Calculate occurrences
    for index_col in index_columns:
        for column_col in column_columns:
            if index_col in df_processed.columns and column_col in df_processed.columns:  # Ensure columns exist
                occurrence_matrix.loc[index_col, column_col] = np.logical_and(df_processed[index_col] == 1, df_processed[column_col] == 1).sum()

    # Dynamic adjustments for figure size based on category count
    min_width_per_column = 80  # Minimum width per column
    min_height_per_row = 35  # Minimum height per row
    base_width = 300  # Base width to start with
    base_height = 200  # Base height to start with

    num_columns = len(column_columns)
    num_rows = len(index_columns)
    fig_width = base_width + num_columns * min_width_per_column
    fig_height = base_height + num_rows * min_height_per_row

    # Make sure the figure is not too small when there are few categories
    fig_width = max(fig_width, 500)  # Adjust minimum figure width as needed
    fig_height = max(fig_height, 300)  # Adjust minimum figure height as needed

    # Custom colorscale from white to blue
    custom_colorscale = [[0, 'white'], [1, 'blue']]

    # Create the heatmap with plotly
    fig = ff.create_annotated_heatmap(
        z=occurrence_matrix.values,
        x=[col.split('_', 1)[1] for col in occurrence_matrix.columns],
        y=[idx.split('_', 1)[1] for idx in occurrence_matrix.index],
        annotation_text=occurrence_matrix.values.astype(str),
        showscale=True,
        colorscale=custom_colorscale,
    )
    fig.update_layout(
        title=title,
        autosize=False,
        width=fig_width,
        height=fig_height,
        margin=dict(t=50, l=50, b=150, r=50),
    )
    fig.update_xaxes(tickangle=-45)

    return fig


# Function to generate and display a heatmap
def display_heatmap(data, index_cols, column_cols, title):
    occurrence_matrix = pd.DataFrame(0, index=index_cols, columns=column_cols)
    for index_col in index_cols:
        for column_col in column_cols:
            count = (data[index_col] & data[column_col]).sum()
            occurrence_matrix.loc[index_col, column_col] = count
    fig, ax = plt.subplots(figsize=(20, 10))  # Adjust figure size as needed
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(occurrence_matrix, annot=True, fmt="d", cmap=cmap, ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    return fig



# Continue with your app logic...
option = st.selectbox(
    'Choose the heatmap you want to display',
    ('Technology vs Issue', 'Issue vs Technology', 'Technology vs Sector', 'Sector vs Issue', 'Transparency vs Issue', 'Issue vs Transparency', 'Sector vs Transparency')
)

# Mapping option to function call
if option == 'Technology vs Issue':
    fig = generate_interactive_heatmap(df_processed, tech_columns, issue_columns, 'Heatmap of Technology(ies) vs Issue(s)')
elif option == 'Issue vs Technology':
    fig = generate_interactive_heatmap(df_processed, issue_columns, tech_columns, 'Heatmap of Issue(s) vs Technology(ies)')
elif option == 'Technology vs Sector':
    fig = generate_interactive_heatmap(df_processed, tech_columns, sector_columns, 'Heatmap of Technology(ies) vs Sector(s)')
elif option == 'Sector vs Issue':
    fig = generate_interactive_heatmap(df_processed, sector_columns, issue_columns, 'Heatmap of Sector(s) vs Issue(s)')
elif option == 'Transparency vs Issue':
    fig = generate_interactive_heatmap(df_processed, transp_columns, issue_columns, 'Heatmap of Transparency vs Issue(s)')
elif option == 'Issue vs Transparency':
    fig = generate_interactive_heatmap(df_processed, issue_columns, transp_columns, 'Heatmap of Issue(s) vs Transparency')
elif option == 'Sector vs Transparency':
    fig = generate_interactive_heatmap(df_processed, sector_columns, transp_columns, 'Heatmap of Sector(s) vs Transparency')

st.plotly_chart(fig, use_container_width=False)
