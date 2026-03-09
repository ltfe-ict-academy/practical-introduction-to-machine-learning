# Part 2: Data Gathering

Machine learning models can often feel like magic, but under the hood, they are entirely dependent on one thing: data. Before we even think about algorithms, training, or making predictions, we have to talk about where our knowledge actually comes from.

## Data Quality Over Everything

In the world of machine learning, there is a famous and inescapable rule: **"Garbage In, Garbage Out."** To be able to predict an Airbnb price with any kind of certainty, you need a good amount of data, and crucially, it must be of the right type. Algorithms are just engines; data is the fuel. If your data is heavily biased, incomplete, or simply incorrect, even the most advanced machine learning model in the world will give you terrible predictions.

Before building anything, a data scientist must carefully consider the sources of their data and document its origin to ensure the model is learning from reality, not from noise or errors.

## Where Does Data Come From?

Data doesn't just magically appear on our hard drives. In the real world, we have to go out and retrieve it. Generally, there are four main ways we gather this information:
- **Web Scraping**: This involves writing code to read the underlying HTML of a webpage and systematically extract the text or numbers you see on screen. It's powerful but can be tedious if the website changes its layout.
- **APIs (Application Programming Interfaces)**: This is the preferred method when available. An API allows your code to request data directly from a company's database in a clean, structured format. It's essentially how software talks to other software.
- **Open Datasets**: Often, the hard work of gathering data has already been done by researchers or communities. You can retrieve massive datasets from repositories like [Kaggle](https://www.kaggle.com/), government portals, or dedicated projects like [Inside Airbnb](https://insideairbnb.com/).
- **Manual Collection**: In some cases, especially when dealing with small datasets or specific niche topics, you might have to collect data manually. This could involve surveys, interviews, or even just copying and pasting information from various sources.

For our workshop, we are skipping the heavy lifting of scraping. We will be using a fantastic open dataset containing a comprehensive list of real Airbnb listings from Vienna.

## Practical example: Gathering Airbnb Data

Check the interactive notebook in the `02_part_Data_Gathering` folder to see how we can use Python to retrieve and load the Airbnb dataset for Vienna. We will be using the `pandas` library to read the CSV file.

> When opening the notebook for the first time press `Select Kernel` in the top right corner and select the Python environment you set up previously. This will allow you to run the code cells in the notebook.
