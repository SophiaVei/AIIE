import pandas as pd

# Read the dataset
df = pd.read_csv(r"C:\Users\Sofia\PycharmProjects\AIIE\processed_dataset.csv").dropna(how="all")

# Save original 'Technology(ies)' values for comparison
original_values_technology = df['Technology(ies)'].copy()

# Remove leading spaces for 'Technology(ies)' column
df['Technology(ies)'] = df['Technology(ies)'].str.lstrip()

# Adjusted function to split technologies by both ',' and ';' and normalize to lowercase, excluding empty strings
def split_and_normalize(technologies):
    if pd.isna(technologies):
        return []
    return [tech.strip().lower() for tech in technologies.replace(';', ',').split(',') if tech.strip()]

# Using the adjusted function for processing 'Technology(ies)'
technology_set = set()
df['Technology(ies)'].apply(lambda x: [technology_set.add(tech) for tech in split_and_normalize(x)])

# Initialize a dictionary to hold one-hot encoding for 'Technology(ies)' with normalized keys
technology_encoding = {f'tech_{tech}': [] for tech in technology_set}

# Populate the dictionary with 1s and 0s for each technology
for _, row in df.iterrows():
    technologies = split_and_normalize(row['Technology(ies)'])
    for tech in technology_set:
        technology_encoding[f'tech_{tech}'].append(int(tech in technologies))

# Convert the dictionary to a DataFrame for 'Technology(ies)'
tech_df = pd.DataFrame(technology_encoding)

# Save original 'Sector(s)' values for comparison
original_values_sector = df['Sector(s)'].copy()

# Remove leading spaces for 'Sector(s)' column
df['Sector(s)'] = df['Sector(s)'].str.lstrip()

# Using the adjusted function for processing 'Sector(s)'
sector_set = set()
df['Sector(s)'].apply(lambda x: [sector_set.add(sector) for sector in split_and_normalize(x)])

# Initialize a dictionary to hold one-hot encoding for 'Sector(s)' with normalized keys
sector_encoding = {f'sector_{sector}': [] for sector in sector_set}

# Populate the dictionary with 1s and 0s for each sector
for _, row in df.iterrows():
    sectors = split_and_normalize(row['Sector(s)'])
    for sector in sector_set:
        sector_encoding[f'sector_{sector}'].append(int(sector in sectors))

# Convert the dictionary to a DataFrame for 'Sector(s)'
sector_df = pd.DataFrame(sector_encoding)

# Concatenate the new columns with the original dataframe for both 'Technology(ies)' and 'Sector(s)'
df_processed_technology = pd.concat([df.drop(columns=['Technology(ies)']), tech_df], axis=1)
df_processed_sector = pd.concat([df.drop(columns=['Sector(s)']), sector_df], axis=1)

# Identify, print, and remove columns that contain only 0 values for both 'Technology(ies)' and 'Sector(s)'
columns_to_remove_technology = [column for column in df_processed_technology if df_processed_technology[column].nunique() == 1 and df_processed_technology[column].unique()[0] == 0]
df_processed_technology.drop(columns=columns_to_remove_technology, inplace=True)

columns_to_remove_sector = [column for column in df_processed_sector if df_processed_sector[column].nunique() == 1 and df_processed_sector[column].unique()[0] == 0]
df_processed_sector.drop(columns=columns_to_remove_sector, inplace=True)

# Merge and rename specific columns as requested for 'Technology(ies)'
merge_rename_operations_technology = {
    'tech_unclear/unknown': ['tech_unclear/unknown', 'tech_unknown'],
    'tech_facial recognition/detection/identification': ['tech_facial recognition', 'tech_facial detection', 'tech_facial recogniton', 'tech_facial detection'],
    'tech_(performance) scoring algorithm': ['tech_performance scoring algorithm', 'tech_scoring algorithm'],
    'tech_location recognition/tracking': ['tech_location tracking', 'tech_location recognition'],
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
    'tech_behavioural monitoring (system)': ['tech_behavioural monitoring', 'tech_behavioural monitoring system'],
    'tech_predictive (statistical) analytics': ['tech_predictive statistical analysis', 'tech_predictive analytics'],
    'tech_content management/moderation system': ['tech_content moderation system', 'tech_content management system'],
    'tech_deepfake - audio': [ 'tech_deepfake - audio', 'tech_audio'],
    'tech_deepfake - image': [ 'tech_deepfake - image',  'tech_image', ],
    'tech_deepfake - video': [ 'tech_deepfake - video',  'tech_video']
}

for new_column, old_columns in merge_rename_operations_technology.items():
    # Sum the one-hot encoded values of the columns to be merged for 'Technology(ies)'
    df_processed_technology[new_column] = df_processed_technology[old_columns].max(axis=1)
    # Drop the old columns for 'Technology(ies)'
    df_processed_technology.drop(columns=old_columns, inplace=True, errors='ignore')

# Print the number of newly created/merged column names for 'Technology(ies)'
print("Number of newly created/merged column names for Technology(ies):", len(df_processed_technology.columns))
print("Newly created/merged column names for Technology(ies):", list(technology_encoding.keys()))
# Merge and rename specific columns as requested for 'Sector(s)'
merge_rename_operations_sector = {
    # Define your merge and rename operations for sectors here
}

# Merge and rename specific columns as requested for 'Technology(ies)'
merge_rename_operations_sector = {
    'sector_sector_real estate sales/management': ['sector_real estate sales/management',  'sector_real estate'],
     'sector_govt - health': [ 'sector_gov - health', 'sector_govt - health'],
     'sector_professional/business services': [ 'sector_professional/business services',  'sector_business/professional services'],
     'sector_govt - police': [ 'sector_govt - police', 'sector_police'],
}

for new_column, old_columns in merge_rename_operations_sector.items():
    # Sum the one-hot encoded values of the columns to be merged for 'Sector(s)'
    df_processed_sector[new_column] = df_processed_sector[old_columns].max(axis=1)
    # Drop the old columns for 'Sector(s)'
    df_processed_sector.drop(columns=old_columns, inplace=True, errors='ignore')

# Print the number of newly created/merged column names for 'Sector(s)'
print("Number of newly created/merged column names for Sector(s):", len(df_processed_sector.columns))
print("Newly created/merged column names for Sector(s):", list(sector_encoding.keys()))

# Save the processed DataFrames
df_processed_technology.to_csv(r"C:\Users\Sofia\PycharmProjects\AIIE\final_processed_technology_dataset.csv", index=False)
df_processed_sector.to_csv(r"C:\Users\Sofia\PycharmProjects\AIIE\final_processed_sector_dataset.csv", index=False)