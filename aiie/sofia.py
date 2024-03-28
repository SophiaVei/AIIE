import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.figure_factory as ff


# Load the dataset
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

# Process each column with the preprocess_column function and your configurations
df_processed, technology_set = preprocess_column(df, 'Technology(ies)', 'tech', merge_rename_operations_technology)
df_processed, sector_set = preprocess_column(df_processed, 'Sector(s)', 'sector', merge_rename_operations_sector)
df_processed, issue_set = preprocess_column(df_processed, 'Issue(s)', 'issue', merge_rename_operations_issue)
df_processed, transparency_set = preprocess_column(df_processed, 'Transparency', 'transp', merge_rename_operations_transparency)

# Extracting technology and issue columns
tech_columns = [col for col in df_processed.columns if col.startswith('tech_')]
issue_columns = [col for col in df_processed.columns if col.startswith('issue_')]
sector_columns = [col for col in df_processed.columns if col.startswith('sector_')]
transp_columns = [col for col in df_processed.columns if col.startswith('transp_')]


def generate_interactive_heatmap(df_processed, index_columns, column_columns, title):
    occurrence_matrix = pd.DataFrame(0, index=index_columns, columns=column_columns)
    for index_col in index_columns:
        for column_col in column_columns:
            count = (df_processed[index_col] & df_processed[column_col]).sum()
            occurrence_matrix.loc[index_col, column_col] = count

    # Define the size of the figure
    heatmap_width = max(10, len(column_columns) * 20)
    heatmap_height = max(10, len(index_columns) * 20)

    fig = ff.create_annotated_heatmap(
        z=occurrence_matrix.values,
        x=occurrence_matrix.columns.tolist(),
        y=occurrence_matrix.index.tolist(),
        annotation_text=None,  # Removing the annotations
        showscale=True,
        colorscale='Viridis'
    )
    fig.update_layout(
        title=title,
        autosize=False,
        width=heatmap_width,  # Set the width of the heatmap
        height=heatmap_height,  # Set the height of the heatmap
        margin=dict(t=50, l=50, b=50, r=50),  # Adjust margins to fit titles, labels, etc
    )
    fig.update_xaxes(side="top")  # Move x-axis labels to the top

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

# Interactive Streamlit components
st.title('Data Exploration Heatmaps')

# Process and prepare data
df = load_data()
df_processed, technology_set = preprocess_column(df, 'Technology(ies)', 'tech', merge_rename_operations_technology)
df_processed, sector_set = preprocess_column(df_processed, 'Sector(s)', 'sector', merge_rename_operations_sector)
df_processed, issue_set = preprocess_column(df_processed, 'Issue(s)', 'issue', merge_rename_operations_issue)
df_processed, transparency_set = preprocess_column(df_processed, 'Transparency', 'transp', merge_rename_operations_transparency)

# Example usage in Streamlit
option = st.selectbox(
    'Choose the heatmap you want to display',
    ('Technology vs Issue', 'Issue vs Technology', 'Technology vs Sector', 'Sector vs Issue', 'Transparency vs Issue', 'Issue vs Transparency', 'Sector vs Transparency'))

# Mapping option to function call
if option == 'Technology vs Issue':
    fig = generate_interactive_heatmap(df_processed, tech_columns, issue_columns, 'Heatmap of Technology(ies) vs Issue(s)')
    st.plotly_chart(fig, use_container_width=False)  # Set to False to use the specified figure dimensions
elif option == 'Issue vs Technology':
    fig = generate_interactive_heatmap(df_processed, issue_columns, tech_columns, 'Heatmap of Issue(s) vs Technology(ies)')
    st.plotly_chart(fig, use_container_width=False)
elif option == 'Technology vs Sector':
    fig = generate_interactive_heatmap(df_processed, tech_columns, sector_columns, 'Heatmap of Technology(ies) vs Sector(s)')
    st.plotly_chart(fig, use_container_width=False)
elif option == 'Sector vs Issue':
    fig = generate_interactive_heatmap(df_processed, sector_columns, issue_columns, 'Heatmap of Sector(s) vs Issue(s)')
    st.plotly_chart(fig, use_container_width=False)
elif option == 'Transparency vs Issue':
    fig = generate_interactive_heatmap(df_processed, transp_columns, issue_columns, 'Heatmap of Transparency vs Issue(s)')
    st.plotly_chart(fig, use_container_width=False)
elif option == 'Issue vs Transparency':
    fig = generate_interactive_heatmap(df_processed, issue_columns, transp_columns, 'Heatmap of Issue(s) vs Transparency')
    st.plotly_chart(fig, use_container_width=False)
elif option == 'Sector vs Transparency':
    fig = generate_interactive_heatmap(df_processed, sector_columns, transp_columns, 'Heatmap of Sector(s) vs Transparency')
    st.plotly_chart(fig, use_container_width=False)

# No need for st.pyplot(fig) when using Plotly charts
