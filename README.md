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


# PROGRESS
## Business Recommendation Feature

The Business Recommendation feature allows users to find the top restaurant in a specific business category and get recommendations for similar cities based on demographic data.

### How It Works

1. The user visits the index page (`/`) and is presented with a form to select a business type from a dropdown menu.

2. When the user submits the form, a POST request is sent to the `/recommend` route.

3. The `/recommend` route performs the following steps:
   - Retrieves the selected business type from the form data.
   - Calls the `find_top_restaurant_in_category` function to find the top restaurant in the selected category based on ratings and reviews.
   - Retrieves the city of the top restaurant.
   - Calls the `finding_similar_cities` function to find the top 5 similar cities to the restaurant's city based on demographic data.
   - Renders the `recommendation.html` template with the top restaurant details and the list of similar cities.

4. The `recommendation.html` template displays the top restaurant information, including the city, business name, address, rating, and reviews. It also shows the list of top 5 similar cities.

### Functions

The Business Recommendation feature relies on the following functions:

- `create_category_dict(csv_file)`: Creates a dictionary of business categories and their corresponding restaurant information from a CSV file.
- `find_top_restaurant_in_category(category_dict, category)`: Finds the top restaurant in a specific category based on ratings and reviews.
- `finding_similar_cities(csv_file, target_city, exclude_restaurant=None)`: Finds the top 5 similar cities to a target city based on demographic data from a CSV file.

### Data Files

The feature uses the following data files:

- `rating_reviews.csv`: Contains restaurant rating and review data.
- `clean_merged_iowa_demo_with_description.csv`: Contains demographic data for cities.

Make sure these files are located in the `app/static/data/csv/` directory.

### Template Files

The feature uses the following template files:

- `index.html`: Renders the form for selecting a business type.
- `recommendation.html`: Renders the recommendation results, including the top restaurant and similar cities.

### Error Handling

If no demographic data is found for the top restaurant's city, an empty list of similar cities will be returned, and no similar cities will be displayed in the recommendation results.

### Future Improvements

- Enhance the algorithm to work for every type of business.