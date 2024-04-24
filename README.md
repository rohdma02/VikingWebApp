## Documentation for the interface 

Viking is a web application built with Flask

## Prerequisites

- Python 3.x
- pip (Python package installer)

## Getting Started

1.Activate the virtual environment:

- For Windows:
  ```
  python -m venv virtualEnvironmentName
  virtualEnvironmentName\Scripts\activate
  ```

- For macOS and Linux:
  ```
  python -m venv virtualEnvironmentName
  source virtualEnvironmentName/bin/activate
  ```

Your command prompt should now show the name of the virtual environment.

2.Install the required dependencies:

```
pip install -r requirements.txt
```

This will install all the necessary packages specified in the `requirements.txt` file.

3.Run the Flask application:

```
flask run
```

This will start the development server, and you should see output similar to the following:

Running on http://127.0.0.1:5000/

Open your web browser and visit `http://127.0.0.1:5000` or `http://localhost:5000` to see the Viking web app in action.



## Project Structure

The project structure is as follows:

```
interface/
│
├── app/
│   ├── templates/
│   │   ├── includes/
│   │   │   ├── footer.html
│   │   │   └── nav.html
│   │   ├── about.html
│   │   ├── base.html
│   │   ├── doc.html
│   │   ├── index.html
│   │   └── recommendation.html
│   ├── static/
│   │   ├── css/
│   │        ├── about.css
│   │   │    └── main.css
│   │   └── data/
│   │        ├── bert_embeddings/
│   │        ├──demographics/
│   │        ├── images/
│   │        └── csv/
│   │   
│   ├── init.py
│   ├── config.py
│   └── views.py
│
├── README.md
├── requirements.txt
└── run.py
```

- The `app` directory contains the Flask application code.
  - The `templates` directory contains the HTML templates used for rendering the web pages.
    - The `includes` directory contains reusable HTML components such as the navigation bar and footer.
  - The `static` directory contains static assets such as CSS stylesheets, images, CSV files, and  files.
  - `__init__.py` initializes the Flask app and sets up the necessary configurations.
  - `config.py` contains configuration settings for the application.
  - `views.py` defines the routes and handles the requests.
- `README.md` is the file you are currently reading, providing an overview of the project.
- `requirements.txt` lists the required Python packages for the project.
- `run.py` is the entry point to run the Flask application.

## Contributing

Do not make changes in the main branch! If you'd like to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your forked repository.
5. Submit a pull request detailing your changes.

## Business Recommendation Feature

The Business Recommendation feature allows users to find the top restaurant in a specific business category and get recommendations for similar cities based on demographic data.
### How It Works

1. The user visits the index page (`/`) and is presented with a form to select a business type from a dropdown menu.

2. When the user submits the form, a POST request is sent to the `/recommend` route.

3. The `/recommend` route performs the following steps:
   - Retrieves the selected business type from the form data.
   - Calls the `find_top_restaurants_info` function to find the top restaurants in the selected category based on the category dictionary.
   - For each top restaurant, calls the `find_similar_cities_in_iowa` function to find the top 3 similar cities in Iowa based on the restaurant's city and demographic data.
   - Renders the `recommendation.html` template with the top restaurants' information and the list of similar cities for each restaurant.

4. The `recommendation.html` template displays a table with the top restaurants' information, including the restaurant name, location (city and state), and the top 3 similar cities in Iowa for each restaurant.

### Functions

The Business Recommendation feature relies on the following functions:

- `read_data(file_path)`: Reads data from a CSV file and returns a pandas DataFrame.
- `clean_numerical_data(data)`: Cleans the numerical data in a DataFrame by converting columns to numeric data type.
- `save_bert_embeddings(all_state_wiki_descriptions, all_state_bert_embeddings, folder_path)`: Saves the BERT embeddings and wiki descriptions to a folder.
- `load_bert_embeddings(folder_path)`: Loads the BERT embeddings and wiki descriptions from a folder.
- `get_bert_embeddings(sentences)`: Extracts BERT embeddings for a list of sentences.
- `combine_features(data, bert_embeddings)`: Combines numerical features and BERT embeddings.
- `apply_pca(combined_features)`: Applies PCA for dimensionality reduction.
- `create_category_dict(data_frame)`: Creates a dictionary of business categories and their corresponding restaurant information from a DataFrame.
- `find_top_restaurants_in_category(category_dict, category)`: Finds the top restaurants in a specific category based on the category dictionary.
- `find_top_restaurants_info(category_dict, category, top_n)`: Finds the top N restaurants' information in a specific category.
- `find_similar_cities_in_iowa(city_name, all_state_demo, iowa_indices, cos_sim_matrix, top_n)`: Finds the top N similar cities in Iowa based on a target city and cosine similarity matrix.

### Data Files

The feature uses the following data files:

- `all_state_rating_rev_common_cities.csv`: Contains restaurant rating and review data for all states.
- `all_state_demo_common_cities.csv`: Contains demographic data for cities in all states.

Make sure these files are located in the `app/static/data/restaurants/` and `app/static/data/demographics/` directories, respectively.

### BERT Embeddings

The feature uses BERT embeddings to represent the textual data (wiki descriptions) of cities. The BERT embeddings are stored in the `app/static/data/bert_embeddings/` directory. If the embeddings are not found, they will be generated and saved for future use.

### Template Files

The feature uses the following template files:

- `index.html`: Renders the form for selecting a business type.
- `recommendation.html`: Renders the recommendation results, including the top restaurants and similar cities for each restaurant.

### Future Improvements

- Enhance the algorithm to handle cases where no restaurants are found for the selected category.
- Improve the efficiency of the code by optimizing the data processing and similarity calculations.
- Expand the feature to include more states and cities beyond Iowa.