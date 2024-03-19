import pandas as pd

df = pd.read_csv(r"C:\Users\Sofia\PycharmProjects\AIIE\processed_dataset.csv").dropna(how="all")

# Function to split technologies by both ',' and ';' and normalize to lowercase
def split_and_normalize(technologies):
    if pd.isna(technologies):
        return []
    return [tech.strip().lower() for tech in technologies.replace(';', ',').split(',')]

# Prepare to encode technologies with normalization for case sensitivity and multiple separators
technology_set = set()
df['Technology(ies)'].apply(lambda x: [technology_set.add(tech) for tech in split_and_normalize(x)])

# Initialize a dictionary to hold our one-hot encoding with normalized keys
technology_encoding = {f'tech_{tech}': [] for tech in technology_set}

# Populate the dictionary with 1s and 0s for each technology
for _, row in df.iterrows():
    technologies = split_and_normalize(row['Technology(ies)'])
    for tech in technology_set:
        technology_encoding[f'tech_{tech}'].append(int(tech in technologies))

# Convert the dictionary to a DataFrame
tech_df = pd.DataFrame(technology_encoding)

# Concatenate the new columns with the original dataframe and drop the 'Technology(ies)' column
df_processed = pd.concat([df.drop(columns=['Technology(ies)']), tech_df], axis=1)

# List of all new columns added in place of "Technology(ies)" based on the provided logic
new_columns = [f"tech_{tech}" for tech in technology_set]

# Print all new columns
print(new_columns)

# Print the number of these new columns
print(len(new_columns))

# Save the processed DataFrame to a new CSV file to avoid fragmentation issues
df_processed.to_csv(r"C:\Users\Sofia\PycharmProjects\AIIE\final_processed_dataset.csv", index=False)

