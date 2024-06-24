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
                                                # Seleccionar las columnas relevantes
        selected_columns = ['neighbourhood_group', 'neighbourhood', 'latitude', 'longitude',
                            'room_type', 'price', 'minimum_nights', 'number_of_reviews',
                            'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
        
        df = df[selected_columns]
        
        # Interfaz de Streamlit
        st.title("Predicción del Precio de Alquiler en Valencia")
        st.header("Características de la Vivienda")
        
        # Mostrar los primeros registros y las columnas disponibles
        st.write(df.head())
        st.write("Columnas disponibles en el DataFrame:")
        st.write(df.columns)
        
        # Preprocesamiento de datos
        st.subheader("Preprocesamiento de Datos")
        st.write("Transformación logarítmica de los precios para normalizar la distribución")
        df['log_price'] = np.log1p(df['price'])
        
        
        # Filtrado de valores atípicos basado en distribución logarítmica
        st.subheader("Filtrado de Valores Atípicos")
        st.write("Seleccionamos únicamente los precios que están entre 3 y 8 en la escala logarítmica")
        df_filtered = df[(df['log_price'] > 3) & (df['log_price'] < 8)]
        
        
        # División en conjunto de entrenamiento y prueba
        X = df_filtered.drop(['price', 'log_price'], axis=1)
        y = df_filtered['log_price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenamiento del modelo de Regresión Ridge
        model = Ridge(alpha=1.0)  # Puedes ajustar el parámetro alpha según sea necesario
        model.fit(X_train, y_train)
        
        # Evaluación del modelo
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        st.subheader("Resultados del Modelo de Regresión Ridge")
        st.write(f"Error Cuadrático Medio (MSE) - Conjunto de Entrenamiento: {mse_train}")
        st.write(f"Error Cuadrático Medio (MSE) - Conjunto de Prueba: {mse_test}")
        st.write(f"Coeficiente de Determinación (R^2) - Conjunto de Entrenamiento: {r2_train}")
        st.write(f"Coeficiente de Determinación (R^2) - Conjunto de Prueba: {r2_test}")
        
        # Interfaz para predicción de precio de alquiler
        st.subheader("Predicción de Precio de Alquiler")
        st.write("Ingrese las características de la vivienda para obtener una predicción")
        
        neighbourhood_group = st.selectbox("Distrito:", df['neighbourhood_group'].unique())
        neighbourhood = st.selectbox("Barrio:", df['neighbourhood'].unique())
        latitude = st.slider("Latitud:", min_value=df['latitude'].min(), max_value=df['latitude'].max())
        longitude = st.slider("Longitud:", min_value=df['longitude'].min(), max_value=df['longitude'].max())
        room_type = st.selectbox("Tipo de Habitación:", df['room_type'].unique())
        minimum_nights = st.slider("Número Mínimo de Noches:", min_value=df['minimum_nights'].min(), max_value=df['minimum_nights'].max())
        number_of_reviews = st.slider("Número de Reviews:", min_value=df['number_of_reviews'].min(), max_value=df['number_of_reviews'].max())
        reviews_per_month = st.slider("Reviews por Mes:", min_value=df['reviews_per_month'].min(), max_value=df['reviews_per_month'].max())
        calculated_host_listings_count = st.slider("Número de Casas Ofertadas:", min_value=df['calculated_host_listings_count'].min(), max_value=df['calculated_host_listings_count'].max())
        availability_365 = st.slider("Disponibilidad en el Año:", min_value=df['availability_365'].min(), max_value=df['availability_365'].max())
        
        # Crear una nueva instancia de entrada para la predicción
        input_data = {
            'neighbourhood_group': neighbourhood_group,
            'neighbourhood': neighbourhood,
            'latitude': latitude,
            'longitude': longitude,
            'room_type': room_type,
            'minimum_nights': minimum_nights,
            'number_of_reviews': number_of_reviews,
            'reviews_per_month': reviews_per_month,
            'calculated_host_listings_count': calculated_host_listings_count,
            'availability_365': availability_365
        }
        
        # Realizar predicción con el modelo entrenado
        input_df = pd.DataFrame([input_data])
        log_price_prediction = model.predict(input_df.drop(['neighbourhood_group', 'neighbourhood', 'room_type'], axis=1))
        
        # Transformar la predicción de vuelta a escala original y mostrarla
        price_prediction = np.expm1(log_price_prediction)[0]
        st.subheader("Precio Estimado de Alquiler")
        st.write(f"El precio estimado de alquiler es: {price_prediction:.2f} dólares")

if __name__ == "__main__":
    main()

