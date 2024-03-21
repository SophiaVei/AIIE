import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

# Set display options to ensure all columns are printed
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# Load the dataset
df = pd.read_csv(r"C:\Users\Sofia\PycharmProjects\AIIE\processed_dataset.csv").dropna(how="all")


# Function to clean, split, and normalize column values
def clean_split_normalize(values):
    if pd.isna(values):
        return []
    return [value.strip().lower() for value in re.split(';|,', values) if value.strip()]


# Preprocess 'Technology(ies)', 'Sector(s)', and 'Issue(s)' columns
def preprocess_column(df, column_name, prefix, merge_rename_operations={}):
    # Remove leading spaces
    df[column_name] = df[column_name].str.lstrip()

    # Initialize sets and dictionaries
    unique_set = set()
    df[column_name].apply(lambda x: [unique_set.add(item) for item in clean_split_normalize(x)])
    encoding_dict = {f'{prefix}_{item}': [] for item in unique_set}

    # Populate the dictionary with 1s and 0s for each item
    for _, row in df.iterrows():
        items = clean_split_normalize(row[column_name])
        for item in unique_set:
            encoding_dict[f'{prefix}_{item}'].append(int(item in items))

    # Create encoded DataFrame
    encoded_df = pd.DataFrame(encoding_dict)

    # Merge and rename columns if operations are provided
    for new_name, old_names in merge_rename_operations.items():
        encoded_df[new_name] = encoded_df[old_names].max(axis=1)
        encoded_df.drop(columns=old_names, inplace=True, errors='ignore')

    # Concatenate the new columns with the original dataframe and drop the processed column
    processed_df = pd.concat([df.drop(columns=[column_name]), encoded_df], axis=1)

    return processed_df, unique_set
# Define merge and rename operations for 'Technology(ies)' and 'Sector(s)'
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
    'tech_deepfake - video': [ 'tech_deepfake - video',  'tech_video'],
    'tech_gesture analysis': ['tech_gesture analysis', 'tech_gesture recognition'],
    'tech_pricing algorithm': ['tech_pricing algorithm', 'tech_price adjustment algorithm']
}

# Merge and rename specific columns as requested for 'Technology(ies)'
merge_rename_operations_sector = {
    'sector_sector_real estate sales/management': ['sector_real estate sales/management',  'sector_real estate'],
    'sector_govt - health': [ 'sector_gov - health', 'sector_govt - health'],
    'sector_professional/business services': [ 'sector_professional/business services',  'sector_business/professional services'],
    'sector_govt - police': [ 'sector_govt - police', 'sector_police'],
    'sector_govt - agriculture': [ 'sector_govt - agriculture', 'sector_agriculture']
}

merge_rename_operations_issue = {
    'issue_bias/discrimination - lgbtqi+': ['issue_bias/discrimination - transgender',  'issue_transgender',  'issue_bias/discrimination - sexual preference (lgbtq)', 'issue_bias/discrimination - lgbtq', 'issue_lgbtq',],
    'issue_necessity/proportionality': [ 'issue_necessity/proportionality', 'issue_proportionality'],
    'issue_bias/discrimination - race/ethnicity': [ 'issue_bias/discimination - race', 'issue_race','issue_bias/disrimination - race','issue_ethnicity','issue_bias/discrimination - racial', 'issue_bias/discrimination - ethnicity',  'issue_bias/discrimination - race',  'issue_bias/disrimination - ethnicity',],
    'issue_bias/discrimination - political': ['issue_bias/discrimination - politics', 'issue_bias/discrimination - political',  'issue_political'],
    'issue_mis/dis-information': ['issue_mis-disinformation',  'issue_mis/dsinformation', 'issue_mis/disinformation'],
    'issue_autonomous lethal weapons': ['issue_autonomous lethal weapons', 'issue_lethal autonomous weapons'],
    'issue_governance/accountability - capability/capacity': ['issue_governance/accountability - capability/capacity',  'issue_governance/accountability', 'issue_governance - capability/capacity'],
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
    'issue_employment - termination': ['issue_employment - termination','issue_termination']
}

# Process each column and keep track of the unique sets
df_processed, technology_set = preprocess_column(df, 'Technology(ies)', 'tech', merge_rename_operations_technology)
df_processed, sector_set = preprocess_column(df_processed, 'Sector(s)', 'sector', merge_rename_operations_sector)
df_processed, issue_set = preprocess_column(df_processed, 'Issue(s)', 'issue', merge_rename_operations_issue)

# Print the newly created/merged column names and their count for each category
tech_columns = [col for col in df_processed.columns if col.startswith('tech_')]
sector_columns = [col for col in df_processed.columns if col.startswith('sector_')]
issue_columns = [col for col in df_processed.columns if col.startswith('issue_')]

print(f"Newly created/merged column names for Technology(ies): {len(tech_columns)}, columns: {tech_columns}")
print(f"Newly created/merged column names for Sector(s): {len(sector_columns)}, columns: {sector_columns}")
print(f"Newly created/merged column names for Issue(s): {len(issue_columns)}, columns: {issue_columns}")

# Save the fully processed DataFrame
df_processed.to_csv(r"C:\Users\Sofia\PycharmProjects\AIIE\final_processed_dataset.csv", index=False)

print("Full preprocessing completed and saved.")



# Extracting technology and issue columns
tech_columns = [col for col in df_processed.columns if col.startswith('tech_')]
issue_columns = [col for col in df_processed.columns if col.startswith('issue_')]
sector_columns = [col for col in df_processed.columns if col.startswith('sector_')]



################################################# TECHNOLOGY VS ISSUE #################################################
# Creating a matrix to count occurrences
occurrence_matrix = pd.DataFrame(0, index=tech_columns, columns=issue_columns)

# Populate the occurrence matrix with counts
for tech_col in tech_columns:
    for issue_col in issue_columns:
        # Count how many times each technology is associated with each issue
        count = (df_processed[tech_col] & df_processed[issue_col]).sum()
        occurrence_matrix.loc[tech_col, issue_col] = count

# Applying a top-k threshold if necessary
top_k_threshold = 5  # Example threshold, adjust as needed
occurrence_matrix = occurrence_matrix[occurrence_matrix.max(axis=1) > top_k_threshold]

# Creating the heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(occurrence_matrix, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Heatmap of Technology(ies) and Issue(s) Occurrences")
plt.xlabel("Issue(s)")
plt.ylabel("Technology(ies)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# Display the heatmap
plt.show()


# Flatten the matrix to get the top occurrences and reset index to make 'tech' and 'sector' columns
top_occurrences = occurrence_matrix.stack().reset_index()
top_occurrences.columns = ['Technology', 'Issue', 'Count']

# Sort by count to get the highest occurrences
top_occurrences_sorted = top_occurrences.sort_values(by='Count', ascending=False)

# Print top 10 occurrences, adjust the number as needed
print(top_occurrences_sorted.head(10).to_string(index=False))



################################################# HEATMAP - TECHNOLOGY VS SECTOR #################################################
# Creating a matrix to count occurrences
occurrence_matrix = pd.DataFrame(0, index=tech_columns, columns=sector_columns)

# Populate the occurrence matrix with counts
for tech_col in tech_columns:
    for sector_col in sector_columns:
        # Count how many times each technology is associated with each issue
        count = (df_processed[tech_col] & df_processed[sector_col]).sum()
        occurrence_matrix.loc[tech_col, sector_col] = count

# Applying a top-k threshold if necessary
top_k_threshold = 5  # Example threshold, adjust as needed
occurrence_matrix = occurrence_matrix[occurrence_matrix.max(axis=1) > top_k_threshold]

# Creating the heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(occurrence_matrix, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Heatmap of Technology(ies) and Sector(s) Occurrences")
plt.xlabel("Sector(s)")
plt.ylabel("Technology(ies)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# Display the heatmap
plt.show()


# Flatten the matrix to get the top occurrences and reset index to make 'tech' and 'sector' columns
top_occurrences = occurrence_matrix.stack().reset_index()
top_occurrences.columns = ['Technology', 'Sector', 'Count']

# Sort by count to get the highest occurrences
top_occurrences_sorted = top_occurrences.sort_values(by='Count', ascending=False)

# Print top 10 occurrences, adjust the number as needed
print(top_occurrences_sorted.head(10).to_string(index=False))



################################################# HEATMAP - SECTOR VS ISSUE #################################################
# Creating a matrix to count occurrences
occurrence_matrix = pd.DataFrame(0, index=sector_columns, columns=issue_columns)

# Populate the occurrence matrix with counts
for sector_col in sector_columns:
    for issue_col in issue_columns:
        # Count how many times each technology is associated with each issue
        count = (df_processed[sector_col] & df_processed[issue_col]).sum()
        occurrence_matrix.loc[sector_col, issue_col] = count

# Applying a top-k threshold if necessary
top_k_threshold = 5  # Example threshold, adjust as needed
occurrence_matrix = occurrence_matrix[occurrence_matrix.max(axis=1) > top_k_threshold]

# Creating the heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(occurrence_matrix, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Heatmap of Sector(s) and Issue(s) Occurrences")
plt.xlabel("Issue(S)")
plt.ylabel("Sector(s)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# Display the heatmap
plt.show()


# Flatten the matrix to get the top occurrences and reset index to make 'tech' and 'sector' columns
top_occurrences = occurrence_matrix.stack().reset_index()
top_occurrences.columns = ['Sector', 'Issue', 'Count']

# Sort by count to get the highest occurrences
top_occurrences_sorted = top_occurrences.sort_values(by='Count', ascending=False)

# Print top 10 occurrences, adjust the number as needed
print(top_occurrences_sorted.head(10).to_string(index=False))




