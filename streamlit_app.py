import streamlit as st
import pandas as pd
import contextily as ctx
import matplotlib.pyplot as plt
import seaborn as sns


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

        # Crear el gráfico de dispersión usando seaborn
        sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group', s=20, data=df)
        st.set_option('deprecation.showPyplotGlobalUse', False)


        # Añadir título y etiquetas de ejes
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        plt.title('Distribución Airbnb NYC')

        # Añadir el mapa base de OpenStreetMap utilizando contextily
        ctx.add_basemap(plt.gca(), crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)

        # Ajustar la leyenda para hacerla más discreta
        plt.legend(title="Agrupaciones de Barrios", loc='lower left', fontsize='small')

        # Mostrar el gráfico en Streamlit
        st.pyplot()

        # Agregamos viviendas por barrio
        st.subheader("Número de viviendas por barrio")
        # Conteo de viviendas por barrio
        neight_count = df.groupby('neighbourhood_group').size().reset_index(name='count')
        cantidades = {elem[0]: elem[1] for elem in neight_count.values}

        # Widget de selección para barrios
        barrios = df['neighbourhood_group'].unique()
        hood = st.selectbox('Selecciona un barrio:', barrios)

        # Mostrar el número de viviendas para el barrio seleccionado
        if hood in cantidades:
            st.write(f'El número de viviendas en {hood} es de {cantidades[hood]}')
        else:
            st.write(f'No hay datos disponibles para el barrio {hood}')

    ### ---- Precios en NYC ----

    if selected_page == "Precios y habitaciones en NYC":
        st.header("Análisis de los precios y tipos de habitación")
        st.subheader("Densidad y distribución de los precios por barrio")
    
        # Lista de barrios de interés
        nei_group_list = ['POBLATS MARITIMS', 'RASCANYA', 'EXTRAMURS', 'CAMPANAR', 'QUATRE CARRERES',
                          'CAMINS AL GRAU', 'LA SAIDIA', 'BENICALAP', 'JESUS', 'CIUTAT VELLA', 
                          "L'OLIVERETA", 'ALGIROS', 'EL PLA DEL REAL', "L'EIXAMPLE", 'PATRAIX', 
                          'BENIMACLET', 'POBLATS DEL SUD', "POBLATS DE L'OEST", 'POBLATS DEL NORD']
    
        # Lista para almacenar los DataFrames de precios por grupo de vecindarios
        price_list_by_n = []
    
        # Obtener estadísticas sobre los rangos de precios para cada grupo de vecindarios
        for group in nei_group_list:
            sub_df = df.loc[df['neighbourhood_group'] == group, 'price']
            stats = sub_df.describe(percentiles=[.25, .50, .75])
            stats.loc['mean'] = sub_df.mean()
            stats = stats[['min', 'max', 'mean']]
            stats.name = group
            price_list_by_n.append(stats)
    
        # Concatenar todos los DataFrames en uno solo para mostrar la tabla final
        stat_df = pd.concat(price_list_by_n, axis=1)
    
        # Mostrar la tabla con los precios mínimos, máximos y la media para cada barrio
        st.write(
            "Como podemos observar a continuación, los valores máximos de los precios para cada uno de los barrios son muy altos. Por tanto, vamos a establecer un límite de 500€ para poder realizar un mejor entendimiento y representación.")
        st.dataframe(stat_df)
    
        # Creación del violinplot
    
        # Crear un sub-dataframe sin valores extremos / menores de 500
        sub_6 = df[df.price < 1000]
        # Usar violinplot para mostrar la densidad y distribución de los precios
        plt.figure(figsize=(12, 8))  # Ajusta el tamaño de la figura para mayor legibilidad
        viz_2 = sns.violinplot(data=sub_6, x='neighbourhood_group', y='price')
        viz_2.set_title('Densidad y distribución de los precios para cada barrio')
        viz_2.set_xlabel('Nombre del barrio')
        viz_2.set_ylabel('Precio en €')
        plt.xticks(rotation=45, ha='right')  # Rotar y alinear etiquetas del eje X
        st.pyplot(plt.gcf())  # se utiliza plt.gcf() para obtener la figura actual
        st.write(
            "Con la tabla estadística y el gráfico de violín podemos observar algunas cosas sobre la distribución de precios de Airbnb en los distritos de Valencia. En primer lugar, podemos afirmar que algunos barrios tienen un rango de precios más alto para las publicaciones, con un precio promedio considerable. Esta distribución y densidad de precios pueden estar influenciadas por factores como la demanda turística y la oferta disponible.")
    
        # Tipo de habitación

        st.subheader("Tipos de habitación por distrito")
        hood1 = st.selectbox("Selecciona el barrio que deseas visualizar:", nei_group_list + ["Todos"])
        agregado_price = sub_6.groupby(['neighbourhood_group', 'room_type']).agg({'price': 'mean'})
        agregado_price1 = agregado_price
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
                "Ten en cuenta que este promedio es teniendo en cuenta solo aquellos alquileres cuyo precio es inferior a 500 euros.")
        else:
            st.pyplot(sns.catplot(x='neighbourhood_group', hue='neighbourhood_group', col='room_type', data=sub_6,
                                  kind="count"))
            st.write("Estos son los precios promedio para cada habitación por barrio:")
            st.dataframe(agregado_price)
            st.write(
                "Ten en cuenta que este promedio es teniendo en cuenta solo aquellos alquileres cuyo precio es inferior a 500 euros.")

if __name__ == "__main__":
    main()

