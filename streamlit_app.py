import streamlit as st
import pandas as pd
import contextily as ctx
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----- Map Creator -----

def main():
    # Page configuration
    st.set_page_config(page_title="Valencia Rentals")

    # Load the data file and cache it
    @st.cache_data
    def load_data():
        return pd.read_csv(r"listings.csv")

    # Data preprocessing
    df = load_data()

    # Create a selection widget for sections at the top
    st.header("Valencia Rentals")
    pages = ("Airbnb Distribution in Valencia", "Prices and Room Types in Valencia", "Use of Methods Related to DS")
    selected_page = st.selectbox(
        label="Choose the section you want to view:",
        options=pages)

    ### ---- Airbnb Distribution in Valencia ----

    if selected_page == "Airbnb Distribution in Valencia":
        st.header("Distribution of Rentals in Valencia")
        st.subheader("Distribution of Houses by Neighborhoods")
        st.write(
            "This chart shows the different houses available on Airbnb in Valencia. The color indicates the neighborhood where they are located.")

        # Map of houses by neighborhoods
        plt.figure(figsize=(10, 10))

        # Create the scatter plot using seaborn
        sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group', s=20, data=df)
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Add title and axis labels
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Airbnb Distribution in Valencia')

        # Add the base map from OpenStreetMap using contextily
        ctx.add_basemap(plt.gca(), crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)

        # Adjust the legend to make it more discreet
        plt.legend(title="Neighborhood Groups", loc='lower left', fontsize='small')

        # Display the plot in Streamlit
        st.pyplot()

        # Add houses by neighborhood
        st.subheader("Number of Houses by Neighborhood")
        # Count houses by neighborhood
        neight_count = df.groupby('neighbourhood_group').size().reset_index(name='count')
        quantities = {elem[0]: elem[1] for elem in neight_count.values}

        # Selection widget for neighborhoods
        neighborhoods = df['neighbourhood_group'].unique()
        hood = st.selectbox('Select a neighborhood:', neighborhoods)

        # Display the number of houses for the selected neighborhood
        if hood in quantities:
            st.write(f'The number of houses in {hood} is {quantities[hood]}')
        else:
            st.write(f'No data available for the neighborhood {hood}')
            
        # Bar chart of the number of houses by neighborhood
        st.subheader("Bar Chart of Number of Houses by Neighborhood")
        plt.figure(figsize=(10, 5))
        sns.barplot(x='neighbourhood_group', y='count', data=neight_count)
        plt.xlabel('Neighborhood')
        plt.ylabel('Number of Houses')
        plt.title('Number of Houses by Neighborhood')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        st.pyplot()

        # Map centered on the selected neighborhood
        st.subheader("Map of the Selected Neighborhood")
        # Create GeoDataFrame of houses
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        geo_df = gpd.GeoDataFrame(df, geometry=geometry)
        geo_df = geo_df.set_crs(epsg=4326)

        # Filter data for the selected neighborhood
        selected_geo_df = geo_df[geo_df['neighbourhood_group'] == hood]

        # Create the map
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        base = selected_geo_df.plot(ax=ax, marker='o', color='red', markersize=5)
        ctx.add_basemap(ax, crs=geo_df.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
        plt.title(f'Map of Houses in {hood}')
        st.pyplot(fig)

    ### ---- Prices and Room Types in Valencia ----

    if selected_page == "Prices and Room Types in Valencia":
        st.header("Analysis of Prices and Room Types in Valencia")
        st.subheader("Density and Distribution of Prices by Neighborhood")
    
        # List of neighborhoods of interest
        nei_group_list = ['POBLATS MARITIMS', 'RASCANYA', 'EXTRAMURS', 'CAMPANAR', 'QUATRE CARRERES',
                          'CAMINS AL GRAU', 'LA SAIDIA', 'BENICALAP', 'JESUS', 'CIUTAT VELLA', 
                          "L'OLIVERETA", 'ALGIROS', 'EL PLA DEL REAL', "L'EIXAMPLE", 'PATRAIX', 
                          'BENIMACLET', 'POBLATS DEL SUD', "POBLATS DE L'OEST", 'POBLATS DEL NORD']
    
        # List to store the DataFrames of prices by neighborhood group
        price_list_by_n = []
    
        # Get statistics on the price ranges for each neighborhood group
        for group in nei_group_list:
            sub_df = df.loc[df['neighbourhood_group'] == group, 'price']
            stats = sub_df.describe(percentiles=[.25, .50, .75])
            stats.loc['mean'] = sub_df.mean()
            stats = stats[['min', 'max', 'mean']]
            stats.name = group
            price_list_by_n.append(stats)
    
        # Concatenate all DataFrames into one to show the final table
        stat_df = pd.concat(price_list_by_n, axis=1)
    
        # Display the table with the minimum, maximum, and average prices for each neighborhood
        st.write(
            "As we can see below, the maximum price values for each neighborhood are very high. Therefore, we will set a limit of 500€ to better understand and represent the data.")
        st.dataframe(stat_df)
    
        # Creation of the violin plot
    
        # Create a sub-dataframe without extreme values / less than 500
        sub_6 = df[df['price'] <= 500]
        # Use violin plot to show the density and distribution of prices
        plt.figure(figsize=(12, 8))  # Adjust the figure size for better readability
        viz_2 = sns.violinplot(data=sub_6, x='neighbourhood_group', y='price')
        viz_2.set_title('Density and Distribution of Prices for Each Neighborhood')
        viz_2.set_xlabel('Neighborhood Name')
        viz_2.set_ylabel('Price in €')
        plt.xticks(rotation=45, ha='right')  # Rotate and align x-axis labels
        st.pyplot(plt.gcf())  # use plt.gcf() to get the current figure
        st.write(
            "With the statistical table and the violin plot, we can observe some things about the price distribution of Airbnb in Valencia districts. Firstly, we can affirm that some neighborhoods have a higher price range for listings, with a considerable average price. This price distribution and density can be influenced by factors such as tourist demand and available supply.")
    
        # Room Type

        st.subheader("Room Types by District")
        hood1 = st.selectbox("Select the neighborhood you want to view:", nei_group_list + ["All"])
        aggregated_price = sub_6.groupby(['neighbourhood_group', 'room_type']).agg({'price': 'mean'})
        aggregated_price1 = aggregated_price
        aggregated_price1 = aggregated_price1.reset_index()
        if hood1 != "All":
            sub_7 = df.loc[df["neighbourhood_group"] == hood1]
            viz_3 = sns.catplot(x='neighbourhood_group', col='room_type', data=sub_7, kind='count')
            viz_3.set_xlabels('')
            viz_3.set_ylabels('Number of Rooms')
            viz_3.set_xticklabels(rotation=90)
            st.pyplot(viz_3)
            st.write(f"The average prices for each room type in the {hood1} district are:")
            st.dataframe(aggregated_price1.loc[aggregated_price1["neighbourhood_group"] == hood1])
            st.write(
                "Note that this average takes into account only those rentals whose price is less than 500 euros.")
        else:
            st.pyplot(sns.catplot(x='neighbourhood_group', hue='neighbourhood_group', col='room_type', data=sub_6,
                                  kind="count"))
            st.write("These are the average prices for each room type by neighborhood:")
            st.dataframe(aggregated_price)
            st.write(
                "Note that this average takes into account only those rentals whose price is less than 500 euros.")

    ### ---- Use of Methods Related to DS ----
    if selected_page == "Use of Methods Related to DS":
        st.header("Machine Learning Models: Price Prediction")

        st.subheader("1. Data Preparation")
        # Selecting relevant features
        features = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
        X = df[features]
        y = df['price']

        st.write("Selected Features:")
        st.write(features)

        st.subheader("1.1 Data Cleaning")
        # Handling missing values
        X.fillna(0, inplace=True)
        y.fillna(0, inplace=True)

        # Checking for infinite values
        X.replace([np.inf, -np.inf], 0, inplace=True)
        y.replace([np.inf, -np.inf], 0, inplace=True)

        st.write("Data cleaned: missing and infinite values handled.")

        st.subheader("2. Data Splitting")
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.write(f"Training Set Size: {X_train.shape[0]} samples")
        st.write(f"Testing Set Size: {X_test.shape[0]} samples")

        st.subheader("3. Model Training")
        # Training the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        st.write("Model Coefficients:")
        coefficients = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
        st.write(coefficients)

        st.subheader("4. Model Evaluation")
        # Making predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluating the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"R-squared (R2): {r2}")

        st.subheader("5. Results Visualization")
        # Scatter plot of actual vs predicted prices
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, color='red')
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title('Actual vs Predicted Prices')
        st.pyplot(plt.gcf())

        st.subheader("Feature Importance")
        # Bar plot of feature importances
        plt.figure(figsize=(10, 5))
        sns.barplot(x=coefficients.index, y='Coefficient', data=coefficients)
        plt.title('Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Coefficient')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(plt.gcf())

if __name__ == "__main__":
    main()

