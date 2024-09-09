import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
import warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler
warnings.filterwarnings('ignore')


def preprocessing(data, Type, ch):
    print(data.info())
    data['Subtitle'] = data['Subtitle'].fillna('')  # No data will be taken as empty string (Object type)
    data['Subtitle'] = data['Subtitle'].apply(lambda x: 0 if x == '' else 1)

    # apply cleaning on In-app Purchases
    data['In-app Purchases'] = data['In-app Purchases'].fillna('0.0')  # No data will be taken as $0  (float type)
    data['In-app Purchases'] = data['In-app Purchases'].apply(lambda x: '0.0' if x == '0' else x)  # for consistency
    data['In-app Purchases'] = data['In-app Purchases'].apply(lambda x: x.split(', '))  # turn string into list

    # apply function convert list of string into list of floating
    def convert_to_float(string_list):
        float_list = []
        for element in string_list:
            try:
                float_list.append(float(element))
            except ValueError:
                print(f"Warning: '{element}' cannot be converted to float and will be skipped.")
        return float_list

    # apply function given mean values for list of floating
    def mean_list(numbers):
        if not numbers:
            return 0.0

        total = sum(numbers)
        length = len(numbers)
        mean = total / length
        return mean

    # convert list of string into list of floating
    data['In-app Purchases'] = data['In-app Purchases'].apply(convert_to_float)
    # find mean values from list of floating
    data['In-app Purchases'] = data['In-app Purchases'].apply(mean_list)

    # apply cleaning on Languages

    data['Languages'] = data['Languages'].fillna(data['Languages'].mode()[0])  # No data will be taken as empty string
    # count number of words in text of Languages
    data['Languages'] = data['Languages'].apply(lambda x: len(x.split(', ')))

    # apply cleaning on URL
    # Define the regular expression pattern
    pattern = r"/([-\w]+)/id\d+|/([-\w]+-\d+)/id\d+"  # regex pattern to extract the string after the last slash and
    # before "/id"
    # Create an empty list to store the extracted game names
    game_names = []
    # Iterate over the rows of the DataFrame
    for url in data['URL']:
        # Apply the regular expression pattern to the URL string
        match = re.search(pattern, url)
        if match:
            game_names.append(match.group(1))
        else:
            game_names.append(np.nan)

    # Add the game names to a new column in the DataFrame
    data['URL'] = game_names
    data['URL'].dropna(how='any', inplace=True)
    le = LabelEncoder()
    data['URL'] = le.fit_transform(data['URL'])

    # Remove duplicates
    data = data.drop_duplicates(subset='ID')

    # Remove duplicates
    data = data.drop_duplicates(subset='Name')
    le = LabelEncoder()
    data['Name'] = le.fit_transform(data['Name'])
    # Check for uniqueness
    # print(data['Name'].unique()) # should print the number of unique values

    # apply cleaning on Description
    # count number of words in text of description
    data['no# of words'] = data['Description'].apply(lambda x: len(re.findall('(\w+)', str(x))))
    data['no# of words'] = data['no# of words'].fillna(data['no# of words'].mean())
    # apply cleaning on Age Rating
    le = LabelEncoder()
    data['Age Rating'] = le.fit_transform(data['Age Rating'])
    data['Age Rating'] = data['Age Rating'].fillna(data['Age Rating'].mean())

    # apply cleaning on Price
    le = LabelEncoder()
    data['Price'] = le.fit_transform(data['Price'])
    data['Price'] = data['Price'].fillna(data['Price'].mean())

    # apply cleaning on Size
    data['Size'] = data['Size'].fillna(data['Size'].mean())
    size_25_percentile = data['Size'].quantile(0.25)
    size_50_percentile = data['Size'].quantile(0.50)
    size_75_percentile = data['Size'].quantile(0.75)
    data['size_Q1'] = data['Size'].apply(lambda x: 1 if x < size_25_percentile else 0)
    data['size_Q2'] = data['Size'].apply(lambda x: 1 if size_25_percentile <= x < size_50_percentile else 0)
    data['size_Q3'] = data['Size'].apply(lambda x: 1 if size_50_percentile <= x < size_75_percentile else 0)
    data['size_Q4'] = data['Size'].apply(lambda x: 1 if x >= size_75_percentile else 0)

    # apply cleaning on Genres
    # convert string into list and count length of this list
    data['Genres'] = data['Genres'].fillna('0.0')
    # convert list of string into list of floating
    data['Genres'] = data['Genres'].apply(lambda x: len(x.split(' ')))
    # apply cleaning on {Original Release Date,Current Version Release Date}
    data['Original Release Date'] = data['Original Release Date'].fillna(0000)
    data['Current Version Release Date'] = data['Current Version Release Date'].fillna(0000)
    date_columns = ['Original Release Date', 'Current Version Release Date']

    # Define the regular expression pattern to extract the year from the 'Date' column
    pattern = r'([0-9]{4})'

    # Create empty lists to store the extracted years
    original_release_years = []
    current_version_years = []

    # Iterate over the rows of the DataFrame
    for date_column in date_columns:
        for date in data[date_column]:
            # Apply the regular expression pattern to the date string
            match = re.search(pattern, str(date))
            if match:
                # Extract the year from the match object and append it to the list
                year = int(match.group(1))
                if date_column == 'Original Release Date':
                    original_release_years.append(year)
                elif date_column == 'Current Version Release Date':
                    current_version_years.append(year)
            else:
                # If the pattern doesn't match, append NaN to the list
                if date_column == 'Original Release Date':
                    original_release_years.append(np.nan)
                elif date_column == 'Current Version Release Date':
                    current_version_years.append(np.nan)

    # Add the years to new columns in the DataFrame
    data['Original Release Year'] = original_release_years
    data['Current Version Release Year'] = current_version_years
    data['Original Release Year'] = data['Original Release Year'].fillna(data['Original Release Year'].mean())
    data['Current Version Release Year'] = data['Current Version Release Year'].fillna(data['Current Version Release Year'].mean())
    # Drop any rows with missing values
    data['Original Release Year'].dropna(inplace=True)
    data['Current Version Release Year'].dropna(inplace=True)

    # apply cleaning on Developer
    # data['Developer'].duplicated().sum() =1064

    data['Developer'] = le.fit_transform(data['Developer'])

    # apply cleaning on Primary Genre
    # data['Primary Genre'].duplicated().sum() = 2965
    # apply LabelEncoder
    le = LabelEncoder()
    data['Primary Genre'] = le.fit_transform(data['Primary Genre'])
    data['Primary Genre'] = data['Primary Genre'].fillna(data['Primary Genre'].mean())

    def apply_feature_scaling(df, column_name):
        scaler = MinMaxScaler()
        reshaped_feature = df[column_name].values.reshape(-1, 1)
        scaled_feature = scaler.fit_transform(reshaped_feature)
        df[column_name] = scaled_feature.flatten()
        return df

    # print(data['User Rating Count'].describe())
    # print(data.shape)
    if ch == 0:
        data = data[data['User Rating Count'] >= data['User Rating Count'].quantile(0.25)]

        # data = data[data['User Rating Count'] >= data['User Rating Count'].quantile(0.25)]

    # print(data.shape)
    if Type == 'classify':
        le = LabelEncoder()
        data['Rate'] = data['Rate'].fillna(data['Rate'].mode())
        data['Rate'] = le.fit_transform(data['Rate'])
    elif Type == 'regression':
        data['Average User Rating'] = data['Average User Rating'].fillna(data['Average User Rating'].mean())
        # Apply feature scaling to 'Average User Rating' column
        #data = apply_feature_scaling(data, 'Average User Rating')

    data = data.drop(columns='Icon URL')
    data = data.drop(columns='Name')
    data = data.drop(columns='URL')
    data = data.drop(columns='ID')
    data = data.drop(columns='Developer')
    data = data.drop(columns='Current Version Release Date')
    data = data.drop(columns='Original Release Date')
    data = data.drop(columns='Size')
    data = data.drop(columns='Description')
    print(data.info())

    return data
