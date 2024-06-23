import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats



# ----- Creador de mapas -----

def main():
    # Configuración de la página
    st.set_page_config(page_title="Alquiler Valencia")

    ## Cargamos el fichero de datos y lo almacenamos en caché
    @st.cache_data
    def load_data():
        return pd.read_csv(r"listings.csv")

    # Pretratamiento del fichero de datos
    df = load_data()

    # Crear un widget de selección para las secciones
    with st.sidebar:
        st.header("Secciones")
        pages = ("Airbnb en NYC", "Precios y habitaciones en NYC")
        selected_page = st.selectbox(
            label="Elige la sección que deseas visualizar:",
            options=pages)

    ### ---- Airbnb en NYC ----

    if selected_page == "Airbnb en NYC":
        st.header("Distribución de los alquileres en NYC")
        st.subheader("Distribución de viviendas por barrios")
        st.write(
            "En este gráfico vemos representadas las diferentes viviendas disponibles en Airbnb Nueva York. El color hace referencia al barrio en donde se situan.")

        # Mapa de las viviendas por barrios
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x='longitude', y='latitude',
                        hue='neighbourhood_group', s=20, data=df)
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        plt.title('Distribución Airbnb NYC')
        plt.legend(title="Agrupaciones de Barrios")
        st.pyplot(plt, clear_figure=True)

        # Agregamos viviendas por barrio
        st.subheader("Número de viviendas por barrio")
        neight_count = df.groupby('neighbourhood_group').agg('count').reset_index()
        cantidades = {elem[0]: elem[1] for elem in neight_count.values}
        barrios = list(cantidades.keys())
        hood = st.selectbox('Selecciona un barrio:', barrios)
        if hood == barrios[0]:
            st.write(f'El número de viviendas en {barrios[0]} es de {cantidades[barrios[0]]}')
        elif hood == barrios[1]:
            st.write(f'El número de viviendas en {barrios[1]} es de {cantidades[barrios[1]]}')
        elif hood == barrios[2]:
            st.write(f'El número de viviendas en {barrios[2]} es de {cantidades[barrios[2]]}')
        elif hood == barrios[3]:
            st.write(f'El número de viviendas en {barrios[3]} es de {cantidades[barrios[3]]}')
        else:
            st.write(f'El número de viviendas en {barrios[4]} es de {cantidades[barrios[4]]}')
            

        # Usamos geopandas para construir una capa base de los barrios de NYC
        st.subheader("Representación de los vecindarios con geopandas")
        st.write(
            "Con el uso de geopandas, podemos obtener el área de cada uno de los barrios. El código es el siguiente:")
        st.code("gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))")
        nyc = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
        st.write("Que convertido a dataframe queda de la siguiente manera:")

        df_neight = pd.DataFrame(nyc.drop('geometry', axis=1))
        st.dataframe(df_neight.head(5))

        nyc.rename(columns={'BoroName': 'neighbourhood_group'}, inplace=True)
        bc_geo = nyc.merge(neight_count,
                           on='neighbourhood_group')  # DataFrame agregado con el número de viviendas por barrio y su área

        # Creamos el mapa
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10))
        bc_geo.plot(column='id', cmap='viridis_r', alpha=.5, ax=ax1, legend=True)
        bc_geo.apply(lambda x: ax1.annotate(text=x.neighbourhood_group, color='black', xy=x.geometry.centroid.coords[0],
                                            ha='center'), axis=1)
        plt.title("Número de alquileres por barrio en NYC")
        plt.axis('off')
        st.pyplot(fig1, clear_figure=True)

    ### ---- Precios en NYC ----

    if selected_page == "Precios y habitaciones en NYC":
        st.header("Análisis de los precios y tipos de habitación")
        st.subheader("Densidad y distribución de los precios por barrio")

        # Almacenamos la información sobre cada uno de los barrios

        neighbourhoods = ['LA MALVA-ROSA', 'SANT LLORENS', 'EL BOTANIC', 'CAMPANAR', 'MONT-OLIVET', 'LA CREU DEL GRAU', 
                  'CIUTAT DE LES ARTS I DE LES CIENCIES', 'MORVEDRE', 'EL CALVARI', 'CIUTAT FALLERA', 'CAMI REAL', 
                  'LA RAIOSA', 'EL PILAR', 'CABANYAL-CANYAMELAR', 'NOU MOLES', 'EN CORTS', 'CIUTAT JARDI', 'MESTALLA', 
                  'RUSSAFA', 'PATRAIX', 'PENYA-ROJA', 'NATZARET', 'SANT ANTONI', 'LA FONTETA S.LLUIS', 'BENIMACLET', 
                  'EL PERELLONET', 'EL CARME', 'JAUME ROIG', 'ARRANCAPINS', 'TORMOS', 'EL PLA DEL REMEI', "L'AMISTAT", 
                  'LES TENDETES', 'LA ROQUETA', 'NA ROVELLA', 'MALILLA', 'EL MERCAT', 'LA XEREA', 'LA PETXINA', 'PINEDO', 
                  'SANT MARCEL.LI', 'CIUTAT UNIVERSITARIA', 'BENIFERRI', 'MARXALENES', 'BENICALAP', 'BENIMAMET', 
                  'ELS ORRIOLS', 'SANT PAU', 'EL GRAU', 'MASSARROJOS', 'LA CREU COBERTA', 'TRES FORQUES', 'TORREFIEL', 
                  'ALBORS', 'LA CARRASCA', 'BETERO', 'AIORA', "L'HORT DE SENABRE", 'LA FONTSANTA', 'SOTERNES', 
                  'SANT FRANCESC', "L'ILLA PERDUDA", 'LA PUNTA', 'EL SALER', 'LA SEU', 'POBLE NOU', 'EXPOSICIO', 'FAVARA', 
                  'VARA DE QUART', 'CAMI FONDO', 'CAMI DE VERA', 'SANT ISIDRE', 'LA VEGA BAIXA', 'LA GRAN VIA', 
                  "CASTELLAR-L'OLIVERAL", 'TRINITAT', 'LA TORRE', "EL FORN D'ALCEDO", 'EL PALMAR', 'SAFRANAR', 'FAITANAR', 
                  'LA LLUM', 'MAHUELLA-TAULADELLA', 'CARPESA']
        # Crear una lista para almacenar los DataFrames de precios por barrio
        price_list_by_n = []
        
        # Iterar sobre cada barrio en la lista
        for neighbourhood in neighbourhoods:
            # Filtrar el DataFrame para obtener solo las filas correspondientes al barrio actual
            sub_df = df.loc[df['neighbourhood'] == neighbourhood]
            
            # Seleccionar solo la columna 'price' y agregar a la lista
            price_sub = sub_df[['price']]
            price_list_by_n.append(price_sub)
        
        # Ahora price_list_by_neighbourhood contiene un DataFrame de precios para cada barrio


        # # Brooklyn
        # sub_1 = df.loc[df['neighbourhood_group'] == 'Brooklyn']
        # price_sub1 = sub_1[['price']]
        # # Manhattan
        # sub_2 = df.loc[df['neighbourhood_group'] == 'Manhattan']
        # price_sub2 = sub_2[['price']]
        # # Queens
        # sub_3 = df.loc[df['neighbourhood_group'] == 'Queens']
        # price_sub3 = sub_3[['price']]
        # # Staten Island
        # sub_4 = df.loc[df['neighbourhood_group'] == 'Staten Island']
        # price_sub4 = sub_4[['price']]
        # # Bronx
        # sub_5 = df.loc[df['neighbourhood_group'] == 'Bronx']
        # price_sub5 = sub_5[['price']]
        # # Ponemos todos los df en una lista
        # price_list_by_n = [price_sub1, price_sub2, price_sub3, price_sub4, price_sub5]

        # creating an empty list that we will append later with price distributions for each neighbourhood_group
        p_l_b_n_2 = []
        # creating list with known values in neighbourhood_group column
        nei_list =['LA MALVA-ROSA', 'SANT LLORENS', 'EL BOTANIC', 'CAMPANAR', 'MONT-OLIVET', 'LA CREU DEL GRAU', 
                  'CIUTAT DE LES ARTS I DE LES CIENCIES', 'MORVEDRE', 'EL CALVARI', 'CIUTAT FALLERA', 'CAMI REAL', 
                  'LA RAIOSA', 'EL PILAR', 'CABANYAL-CANYAMELAR', 'NOU MOLES', 'EN CORTS', 'CIUTAT JARDI', 'MESTALLA', 
                  'RUSSAFA', 'PATRAIX', 'PENYA-ROJA', 'NATZARET', 'SANT ANTONI', 'LA FONTETA S.LLUIS', 'BENIMACLET', 
                  'EL PERELLONET', 'EL CARME', 'JAUME ROIG', 'ARRANCAPINS', 'TORMOS', 'EL PLA DEL REMEI', "L'AMISTAT", 
                  'LES TENDETES', 'LA ROQUETA', 'NA ROVELLA', 'MALILLA', 'EL MERCAT', 'LA XEREA', 'LA PETXINA', 'PINEDO', 
                  'SANT MARCEL.LI', 'CIUTAT UNIVERSITARIA', 'BENIFERRI', 'MARXALENES', 'BENICALAP', 'BENIMAMET', 
                  'ELS ORRIOLS', 'SANT PAU', 'EL GRAU', 'MASSARROJOS', 'LA CREU COBERTA', 'TRES FORQUES', 'TORREFIEL', 
                  'ALBORS', 'LA CARRASCA', 'BETERO', 'AIORA', "L'HORT DE SENABRE", 'LA FONTSANTA', 'SOTERNES', 
                  'SANT FRANCESC', "L'ILLA PERDUDA", 'LA PUNTA', 'EL SALER', 'LA SEU', 'POBLE NOU', 'EXPOSICIO', 'FAVARA', 
                  'VARA DE QUART', 'CAMI FONDO', 'CAMI DE VERA', 'SANT ISIDRE', 'LA VEGA BAIXA', 'LA GRAN VIA', 
                  "CASTELLAR-L'OLIVERAL", 'TRINITAT', 'LA TORRE', "EL FORN D'ALCEDO", 'EL PALMAR', 'SAFRANAR', 'FAITANAR', 
                  'LA LLUM', 'MAHUELLA-TAULADELLA', 'CARPESA']
        # creating a for loop to get statistics for price ranges and append it to our empty list
     
        
        for neighborhood in nei_list:
            sub_df = df.loc[df['neighbourhood'] == neighborhood]
            price_stats = sub_df[['price']].describe(percentiles=[.25, .50, .75])
            price_stats = price_stats.iloc[3:]  # Nos quedamos con las estadísticas relevantes
            price_stats.reset_index(inplace=True)
            price_stats.rename(columns={'index': 'Stats', 'price': neighborhood}, inplace=True)
            p_l_b_n_2.append(price_stats)
        
        # Finalizando el DataFrame con las estadísticas para todos los vecindarios
        stat_df = p_l_b_n_2[0].set_index('Stats')
        for df in p_l_b_n_2[1:]:
            stat_df = stat_df.join(df.set_index('Stats'))         
        
     
        # for x in price_list_by_n:
        #     i = x.describe(percentiles=[.25, .50, .75])
        #     i = i.iloc[3:]
        #     i.reset_index(inplace=True)
        #     i.rename(columns={'index': 'Stats'}, inplace=True)
        #     p_l_b_n_2.append(i)
        # # changing names of the price column to the area name for easier reading of the table
        # p_l_b_n_2[0].rename(columns={'price': nei_list[0]}, inplace=True)
        # p_l_b_n_2[1].rename(columns={'price': nei_list[1]}, inplace=True)
        # p_l_b_n_2[2].rename(columns={'price': nei_list[2]}, inplace=True)
        # p_l_b_n_2[3].rename(columns={'price': nei_list[3]}, inplace=True)
        # p_l_b_n_2[4].rename(columns={'price': nei_list[4]}, inplace=True)
        # # finilizing our dataframe for final view
        # stat_df = p_l_b_n_2
        # stat_df = [df.set_index('Stats') for df in stat_df]
        # stat_df = stat_df[0].join(stat_df[1:])
        st.write(
            "Como podemos observar a continuación, los valores máximos de los precios para cada uno de los barrios son muy altos. Por tanto, vamos a establecer un límite de 500$ para poder realizar un mejor entendimiento y representación.")
        st.dataframe(stat_df)

        # Creación del violinplot

        # creating a sub-dataframe with no extreme values / less than 500
        sub_6 = df[df.price < 500]
        # using violinplot to showcase density and distribtuion of prices
        viz_2 = sns.violinplot(data=sub_6, x='neighbourhood', y='price')
        viz_2.set_title('Densidad y distribución de los precios para cada barrio')
        viz_2.set_xlabel('Nombre del barrio')
        viz_2.set_ylabel('Precio en $')
        st.pyplot(plt.gcf())  # se utiliza plt.gcf() para obtener la figura actual
        st.write(
            "Con la tabla estadística y el gráfico de violín podemos observar algunas cosas sobre la distribución de precios de Airbnb en los distritos de la ciudad de Nueva York. En primer lugar, podemos afirmar que Manhattan tiene el rango de precios más alto para las publicaciones, con un precio promedio de 150 dólares por noche, seguido de Brooklyn con 90 dólares por noche. Queens y Staten Island parecen tener distribuciones muy similares, mientras que Bronx es el más económico de todos. Esta distribución y densidad de precios eran totalmente esperadas; por ejemplo, no es un secreto que Manhattan es uno de los lugares más caros del mundo para vivir, mientras que Bronx, por otro lado, parece tener un nivel de vida más bajo.")

        # Tipo de habitación

        st.subheader("Tipos de habitación por distrito")
        hood1 = st.selectbox("Selecciona el barrio que deseas visualizar:", nei_list + ["Todos"])
        agregado_price = sub_6.groupby(['neighbourhood_group', 'room_type']).agg({'price': 'mean'})
        agregado_price1 = agregado_price[agregado_price.price < 500]
        agregado_price1 = agregado_price1.reset_index()
        if hood1 != "Todos":
            sub_7 = df.loc[df["neighbourhood_group"] == hood1]
            viz_3 = sns.catplot(x='neighbourhood_group', col='room_type', data=sub_7, kind='count')
            viz_3.set_xlabels('')
            viz_3.set_ylabels('Nº de habitaciones')
            viz_3.set_xticklabels(rotation=90)
            st.pyplot(viz_3)
            st.write(f"Los precios promedios para cada tipo de habitación en el distrito {hood1} son:")
            st.dataframe(agregado_price1.loc[agregado_price1["neighbourhood_group"] == hood1])
            st.write(
                "Ten en cuenta que este promedio es teniendo en cuenta solo aquellos alquileres cuyo precio es inferior 500 dólares.")
        else:
            st.pyplot(sns.catplot(x='neighbourhood_group', hue='neighbourhood_group', col='room_type', data=sub_6,
                                  kind="count"))
            st.write("Estos son los precios promedio para cada habitación por barrio:")
            st.dataframe(agregado_price)
            st.write(
                "Ten en cuenta que este promedio es teniendo en cuenta solo aquellos alquileres cuyo precio es inferior a 500 dólares.")

if __name__ == "__main__":
    main()
