import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import folium
import json
import branca.colormap as cm
import streamlit as st
from streamlit_folium import st_folium
import streamlit.components.v1 as components
import difflib
import re
import ast
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from workalendar.europe import Spain
cal = Spain()
holidays_2024 = cal.holidays(2024)
holidays_2024_dates = [date for date, name in holidays_2024] 
holidays_2023 = cal.holidays(2023)
holidays_2023_dates = [date for date, name in holidays_2023] 
all_holidays = set(holidays_2023_dates + holidays_2024_dates)
st.set_page_config(layout="wide")
# Use the cache_data decorator to cache the loading of data
@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

@st.cache_data
def load_geocoded_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_geojson_data(file_path):
    with open(file_path, encoding='utf-8') as f:
        return json.load(f)
    
#CY DATA
file_path = 'salesdata1.xlsx'
sales_data = pd.read_excel(file_path)
# Select relevant columns for further analysis
columns_to_keep = [
    'Shop[Shop Code]', 'Shop[Shop Code - Descr]', 'Shop[Area Code]',
    'Shop[Area Code - Descr]', 'Shop[Area Manager]', 'Shop[D_ADDRESS_DS]',
    'Shop[Province Descr]',
    'Calendar[ISO Week]', 'Calendar[Date]','Shop[Shop Descr]', '[FP_OppTest__Heads__Month_]',
    '[Net_Trial_Activated__Units__Side_]',
    '[Net_Trial_Closed_Same_Month__Units__Side_]',
    '[HA_Sales_from_Trial_Same_Month__Units_]',
    '[Net_Trial_Closed_Previous_Month__Units__Side_]',
    '[HA_Sales_from_Trial_Previous_Month__Units_]',
    '[Agenda_Appointments__Heads_]',
    '[Appointments_Completed]',
    '[Appointments_Cancelled]', 
    '[Direct_Orders__Units_]', 
    '[HA_Sales__Units_]',
    '[HA_Sales__Value_]', 'Calendar[Working Day Spain]'
]

cleaned_data = sales_data[columns_to_keep].copy()

# Rename columns for easier reference
cleaned_data.columns = [
    'Shop_Code', 'Shop_Descr', 'Area_Code', 'Area_Descr', 'Area_Manager', 
    'Shop_Address', 'Province', 'ISO Week', 'Date',
    'Shop_Full_Descr', 'FP_OppTest_Heads_Month',
    'Trial_Activated',
    'Net_Trial_Closed_Same_Month_Units_Side',
    'HA_Sales_from_Trial_Same_Month_Units', 
    'Net_Trial_Closed_Previous_Month_Units_Side',
    'HA_Sales_from_Trial_Previous_Month_Units',
    'Agenda_Appointments_Heads',
    'Appointments_Completed', 
    'Appointments_Cancelled',
    'Direct_Orders_Units',
    'HA_Sales_Units', 
    'HA_Sales_Value', 'working day'
]

# Convert columns to appropriate data types
numeric_columns = [
    'HA_Sales_Units', 'HA_Sales_Value', 'Appointments_Completed', 
    'Appointments_Cancelled', 'Trial_Activated',
    'Net_Trial_Closed_Same_Month_Units_Side', 'HA_Sales_from_Trial_Same_Month_Units',
    'Direct_Orders_Units'
]
cleaned_data[numeric_columns] = cleaned_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Replace missing values with 0s in critical columns
cleaned_data[numeric_columns] = cleaned_data[numeric_columns].fillna(0)
cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'])
#PY DATA
# Previous Year (PY) Data
py_file_path = 'PYsalesdata1.xlsx'
py_sales_data = pd.read_excel(py_file_path)

py_sales = py_sales_data[columns_to_keep].copy()
py_sales.columns = cleaned_data.columns

# Convert columns to appropriate data types
py_sales[numeric_columns] = py_sales[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Replace missing values with 0s in critical columns
py_sales[numeric_columns] = py_sales[numeric_columns].fillna(0)

# Load regionmapping data
region_mapping_path = 'regionmapping.xlsx'
region_mapping = pd.read_excel(region_mapping_path)
# Merge cleaned_data with region_mapping to update 'Region_Descr'
cleaned_data = cleaned_data.merge(region_mapping[['CODE', 'REGION']], left_on='Shop_Code', right_on='CODE', how='left')
cleaned_data.rename(columns={'REGION': 'Region_Descr'}, inplace=True)
cleaned_data = cleaned_data.drop(columns=['CODE'])
py_sales = py_sales.merge(region_mapping[['CODE', 'REGION']], left_on='Shop_Code', right_on='CODE', how='left')
py_sales.rename(columns={'REGION': 'Region_Descr'}, inplace=True)
py_sales = py_sales.drop(columns=['CODE'])


# Convert the 'Date' column to datetime if it isn't already
cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'])
py_sales['Date'] = pd.to_datetime(py_sales['Date'])

today = datetime.now().date()
yesterday = today - timedelta(days=1)

today_working_day = cleaned_data.loc[cleaned_data['Date'] <= pd.Timestamp(today), 'working day'].max()
last_year_today = today.replace(year=today.year - 1)
py_working_day = py_sales.loc[py_sales['Date'] <= pd.Timestamp(last_year_today), 'working day'].max()
mtd_cleaned_data = cleaned_data[cleaned_data['working day'] <= today_working_day]
mtd_py_sales = py_sales[py_sales['working day'] <= py_working_day]

# Load the forecast data
forecast_path = 'fcst_aug.xlsx'
forecast = pd.read_excel(forecast_path, header=9)
forecast = forecast.loc[:, ~forecast.columns.str.startswith('Unnamed')]

string_columns = ['Region', 'Area', 'Shop', 'Code', 'SYM', 'ATG', 'Motivo del cambio']
forecast[string_columns] = forecast[string_columns].astype(str)
float_columns = forecast.columns.difference(string_columns)
forecast[float_columns] = forecast[float_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

fcst_rates_path = 'fcst_rates.xlsx' 
fcst_rates = pd.read_excel(fcst_rates_path, sheet_name='Forecast')
target_working_day_row = fcst_rates[fcst_rates['dia'] == today_working_day]
daily_target_working_day_row = fcst_rates[fcst_rates['dia'] == today_working_day]
if not target_working_day_row.empty:
    gen_rate = target_working_day_row['GEN'].values[0]
    entr_rate = target_working_day_row['ENTR'].values[0]

daily_gen_rate = daily_target_working_day_row['generado'].values[0]
daily_entr_rate = daily_target_working_day_row['entregado'].values[0]

columns_to_string = [
    'Order.GT_RegionCode__c', 'Order', 'Order.GT_AreaCode__c', 
    'Order.GT_ShopCode__c', 'Order.OrderNumber', 'Order.CustomerCode__c',
    'Order.Account_Name__c', 'Order.vlocity_cmt__OriginatingChannel__c', 
    'Order.OrderRecordTypeName__c'
]
sfdatapy =pd.read_excel('sfdatapy.xlsx', dtype={col: str for col in columns_to_string})
sfdatatoday = pd.read_excel('sfdatatoday.xlsx', dtype={col: str for col in columns_to_string})
# Convert relevant columns to numeric
sfdatapy['GT_OneTimePriceNoVAT__c'] = pd.to_numeric(sfdatapy['GT_OneTimePriceNoVAT__c'], errors='coerce')
sfdatapy['Order.GT_TotalAmountNoVAT__c'] = pd.to_numeric(sfdatapy['Order.GT_TotalAmountNoVAT__c'], errors='coerce')
sfdatatoday['GT_OneTimePriceNoVAT__c'] = pd.to_numeric(sfdatatoday['GT_OneTimePriceNoVAT__c'], errors='coerce')
sfdatatoday['Order.GT_TotalAmountNoVAT__c'] = pd.to_numeric(sfdatatoday['Order.GT_TotalAmountNoVAT__c'], errors='coerce')
sfdatatoday = sfdatatoday.merge(region_mapping[['CODE', 'REGION']], left_on='Order.GT_ShopCode__c', right_on='CODE', how='left')
sfdatatoday.rename(columns={'REGION': 'Region_Descr'}, inplace=True)
sfdatatoday = sfdatatoday.drop(columns=['CODE'])

sfdatapy = sfdatapy.merge(region_mapping[['CODE', 'REGION']], left_on='Order.GT_ShopCode__c', right_on='CODE', how='left')
sfdatapy.rename(columns={'REGION': 'Region_Descr'}, inplace=True)
sfdatapy = sfdatapy.drop(columns=['CODE'])
# Convert date fields to datetime format
sfdatapy['Order.GT_SoldDate__c'] = pd.to_datetime(sfdatapy['Order.GT_SoldDate__c'], errors='coerce')
sfdatapy['Order.GT_ReturnDate__c'] = pd.to_datetime(sfdatapy['Order.GT_ReturnDate__c'], errors='coerce')
sfdatapy['Order.EffectiveDate'] = pd.to_datetime(sfdatapy['Order.EffectiveDate'], errors='coerce')

# Convert to date object to match the comparison format
sfdatapy['Order.GT_SoldDate__c'] = sfdatapy['Order.GT_SoldDate__c'].dt.date
sfdatapy['Order.GT_ReturnDate__c'] = sfdatapy['Order.GT_ReturnDate__c'].dt.date
sfdatapy['Order.EffectiveDate'] = sfdatapy['Order.EffectiveDate'].dt.date

# Correctly fill NaT values with a specific placeholder date
sfdatapy['Order.GT_SoldDate__c'] = sfdatapy['Order.GT_SoldDate__c'].fillna(datetime(2000, 1, 1).date())
sfdatapy['Order.GT_ReturnDate__c'] = sfdatapy['Order.GT_ReturnDate__c'].fillna(datetime(2000, 1, 1).date())
sfdatapy['Order.EffectiveDate'] = sfdatapy['Order.EffectiveDate'].fillna(datetime(2000, 1, 1).date())

# Define the specific date for comparison
py_today = datetime.strptime('29.09.2023', '%d.%m.%Y').date()
ca_today = datetime.strptime('30.09.2024', '%d.%m.%Y').date()

# Calculate 'HA_Sales_Value'
sfdatapy['HA_Sales_Value'] = sfdatapy.apply(lambda row: 
    row['GT_OneTimePriceNoVAT__c'] if (
        row['Order.GT_SoldDate__c'] == py_today and row['Order.Status'] == 'Sold'
    ) else (
        -row['GT_OneTimePriceNoVAT__c'] if (
            row['Order.GT_ReturnDate__c'] == py_today and row['Order.Status'] == 'Returned'
        ) else 0
    ), axis=1)


# Calculate 'HA_Sales_Q'
sfdatapy['HA_Sales_Units'] = sfdatapy.apply(lambda row: 
    row['Quantity'] if (
        row['Order.GT_SoldDate__c'] == py_today and row['Order.Status'] == 'Sold'
    ) else (
        -row['Quantity'] if (
            row['Order.GT_ReturnDate__c'] == py_today and row['Order.Status'] == 'Returned'
        ) else 0
    ), axis=1)

# Calculate 'Trial_Activated'
sfdatapy['Trial_Activated'] = sfdatapy.apply(lambda row: 
    row['Quantity'] if (
        row['Order.Type'] in ['Trial'] and
        row['Order.EffectiveDate'] == py_today 
    ) else 0, axis=1)
sfdatatoday['Order.GT_SoldDate__c'] = pd.to_datetime(sfdatatoday['Order.GT_SoldDate__c'], errors='coerce').dt.date
sfdatatoday['Order.GT_ReturnDate__c'] = pd.to_datetime(sfdatatoday['Order.GT_ReturnDate__c'], errors='coerce').dt.date
sfdatatoday['Order.EffectiveDate'] = pd.to_datetime(sfdatatoday['Order.EffectiveDate'], errors='coerce').dt.date

sfdatatoday['Order.GT_SoldDate__c'] = sfdatatoday['Order.GT_SoldDate__c'].fillna(datetime(2000, 1, 1).date())
sfdatatoday['Order.GT_ReturnDate__c'] = sfdatatoday['Order.GT_ReturnDate__c'].fillna(datetime(2000, 1, 1).date())
sfdatatoday['Order.EffectiveDate'] = sfdatatoday['Order.EffectiveDate'].fillna(datetime(2000, 1, 1).date())

# Calculate 'HA_Sales_Value'
sfdatatoday['HA_Sales_Value'] = sfdatatoday.apply(lambda row: 
    row['GT_OneTimePriceNoVAT__c'] if (
        row['Order.GT_SoldDate__c'] == ca_today and row['Order.Status'] == 'Sold'
    ) else (
        -row['GT_OneTimePriceNoVAT__c'] if (
            row['Order.GT_ReturnDate__c'] == ca_today and row['Order.Status'] == 'Returned'
        ) else 0
    ), axis=1)


# Calculate 'HA_Sales_Q'
sfdatatoday['HA_Sales_Units'] = sfdatatoday.apply(lambda row: 
    row['Quantity'] if (
        row['Order.GT_SoldDate__c'] == ca_today and row['Order.Status'] == 'Sold'
    ) else (
        -row['Quantity'] if (
            row['Order.GT_ReturnDate__c'] == ca_today and row['Order.Status'] == 'Returned'
        ) else 0
    ), axis=1)

sfdatatoday['Trial_Activated'] = sfdatatoday.apply(lambda row: 
    row['Quantity'] if (
        row['Order.Type'] in ['Trial'] and
        row['Order.EffectiveDate'] == ca_today 
    ) else 0, axis=1)

# Load the pre-existing geocoded data from CSV
geocoded_file_path = 'geocoded_data.csv'
geocoded_data = pd.read_csv(geocoded_file_path)
# Ensure that the latitude and longitude columns are in the correct format
geocoded_data['Address.latitude'] = pd.to_numeric(geocoded_data['Address.latitude'], errors='coerce')
geocoded_data['Address.longitude'] = pd.to_numeric(geocoded_data['Address.longitude'], errors='coerce')

# Rename columns to match the format in the cleaned data
geocoded_data = geocoded_data.rename(columns={
    'Address.latitude': 'Latitude',
    'Address.longitude': 'Longitude',
    'GT_ShopCode__c': 'Shop_Code'  
})

# Merge the geocoded data with the sales data based on Shop Code
mtd_cleaned_data = mtd_cleaned_data.merge(geocoded_data[['Shop_Code', 'Latitude', 'Longitude']], on='Shop_Code', how='left')

# Identify shops missing address information
missing_address_info = mtd_cleaned_data[mtd_cleaned_data['Latitude'].isna() | mtd_cleaned_data['Longitude'].isna()]
missing_shops = missing_address_info[['Shop_Code', 'Shop_Descr', 'Shop_Address', 'Province', 'Region_Descr']]

# Load the GeoJSON file for provinces
geojson_file_path = 'spain-provinces.geojson'
with open(geojson_file_path, encoding='utf-8') as f:
    geojson_data = json.load(f)

# Function to match provinces using contains and fuzzy matching
def find_best_match(province_name, geojson_data, threshold=0.6):
    best_match = None
    highest_ratio = 0
    
    for feature in geojson_data['features']:
        geojson_province = feature['properties']['name']
        
        # Check if either name contains a significant part of the other
        if province_name.lower() in geojson_province.lower() or geojson_province.lower() in province_name.lower():
            return geojson_province  # Direct match using contains
        
        # Use fuzzy matching for partial matches
        match_ratio = difflib.SequenceMatcher(None, province_name.lower(), geojson_province.lower()).ratio()
        if match_ratio > highest_ratio and match_ratio >= threshold:
            highest_ratio = match_ratio
            best_match = geojson_province
    
    return best_match  # Return the best match found

# Add the best match province name to the sales data
mtd_cleaned_data['GeoJSON_Province'] = mtd_cleaned_data['Province'].apply(lambda x: find_best_match(x, geojson_data))

# Handle remaining unmatched provinces
def handle_remaining_unmatched(province_name):
    if province_name == 'Guipuzcoa':
        return 'Gipuzkoa'
    return province_name

# Apply the specific handling for remaining unmatched provinces
mtd_cleaned_data['GeoJSON_Province'] = mtd_cleaned_data['GeoJSON_Province'].fillna(mtd_cleaned_data['Province'].apply(handle_remaining_unmatched))
# Identify any provinces that still did not match
unmatched_provinces = mtd_cleaned_data[mtd_cleaned_data['GeoJSON_Province'].isna()]
# Group the sales data by the matched GeoJSON province names for mapping
province_data = mtd_cleaned_data.groupby('GeoJSON_Province').agg({
    'HA_Sales_Value': 'sum', 
    'Shop_Code': pd.Series.nunique
}).reset_index()
province_data.columns = ['Province', 'sale', 'shop_count']
province_data['average_sale'] = province_data['sale'] / province_data['shop_count']

# Calculate the sale quartiles for color scaling
min_sale = province_data['sale'].min()
max_sale = province_data['sale'].max()
province_q1 = province_data['sale'].quantile(0.25)
province_q2 = province_data['sale'].quantile(0.5)
province_q3 = province_data['sale'].quantile(0.75)

def get_province_color(sale, min_sale, province_q1, province_q2, province_q3, max_sale):
    if sale < province_q1:
        return cm.LinearColormap(['#ff0000', '#ffcccc'], vmin=min_sale, vmax=province_q1)(sale)
    elif sale <= province_q2:
        return cm.LinearColormap(['#ffffcc', 'yellow'], vmin=province_q1, vmax=province_q2)(sale)
    elif sale <= province_q3:
        return cm.LinearColormap(['#99ff99', 'green'], vmin=province_q2, vmax=province_q3)(sale)
    else:
        return cm.LinearColormap(['#006400', 'darkgreen'], vmin=province_q3, vmax=max_sale)(sale)

# Merge the province data with the GeoJSON data
for feature in geojson_data['features']:
    province_name = feature['properties']['name']
    matching_rows = province_data[province_data['Province'].apply(lambda x: province_name.lower() in x.lower() or x.lower() in province_name.lower())]
    
    if not matching_rows.empty:
        sale = int(matching_rows['sale'].values[0])
        shop_count = int(matching_rows['shop_count'].values[0])
        average_sale = matching_rows['average_sale'].values[0]
        feature['properties']['sale'] = sale
        feature['properties']['shop_count'] = shop_count
        feature['properties']['average_sale'] = average_sale
    else:
        feature['properties']['sale'] = None
        feature['properties']['shop_count'] = None
        feature['properties']['average_sale'] = None

# Create a base map centered on Spain
m = folium.Map(
    location=[40.4168, -3.7038],  # Centered on Spain
    zoom_start=6,  # Initial zoom level
    scrollWheelZoom=False,  # Disable zooming with the scroll wheel
    dragging=False  # Disable dragging the map
)
# Add province overlay
def style_function(feature):
    return {
        'fillColor': get_province_color(
            feature['properties']['sale'],
            min_sale,
            province_q1,
            province_q2,
            province_q3,
            max_sale
        ) if feature['properties']['sale'] is not None else '#ffffff',
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.3,
        'opacity': 0.3
    }

def highlight_function(feature):
    return {
        'fillColor': get_province_color(
            feature['properties']['sale'],
            min_sale,
            province_q1,
            province_q2,
            province_q3,
            max_sale
        ) if feature['properties']['sale'] is not None else '#ffffff',
        'color': 'black',
        'weight': 2,
        'fillOpacity': 0.6,
        'opacity': 0.6
    }

tooltip = folium.GeoJsonTooltip(
    fields=['name', 'average_sale', 'shop_count'],
    aliases=['Province:', 'Average Sales:', 'Shop Count:'],
    localize=True
)

# Add the province GeoJSON layer to the map
geojson_layer = folium.GeoJson(
    geojson_data,
    name='Provinces',
    style_function=style_function,
    highlight_function=highlight_function,
    tooltip=tooltip
).add_to(m)

# Add Layer Control
folium.LayerControl().add_to(m)
# Add a color scale legend
colormap = cm.LinearColormap(
    ['#ff0000', '#ff3333', '#ff6666', '#ffffcc', '#ccffcc', '#006400'],
    vmin=min_sale,
    vmax=max_sale,
    caption='Average Sales (€)'
)
colormap.add_to(m)

# Function to dynamically add shop circles to the existing map
def add_shop_circles_to_map(selected_province_data):
    for _, row in selected_province_data.iterrows():
        folium.Circle(
            location=(row['Latitude'], row['Longitude']),
            radius=100,  # Small fixed radius for visibility
            color=get_province_color(
                row['HA_Sales_Value'], min_sale, province_q1, province_q2, province_q3, max_sale
            ),
            fill=True,
            fill_color=get_province_color(
                row['HA_Sales_Value'], min_sale, province_q1, province_q2, province_q3, max_sale
            ),
            fill_opacity=0.7,
            tooltip=f"{row['Shop_Descr']}: €{row['HA_Sales_Value']:,.2f}"
        ).add_to(m)

# JavaScript to handle click event and pass the selected province to Streamlit
js_code = f"""
    <script>
    function addClickEventToMap(geojson) {{
        geojson.on('click', function(e) {{
            const clickedFeature = e.propagatedFrom.feature.properties;
            const provinceName = clickedFeature.name;

            // Send the province name to Streamlit
            const message = {{
                type: 'selected_province',
                province: provinceName
            }};
            window.parent.postMessage(message, "*");
        }});
    }}

    window.addEventListener('load', function() {{
        const mapElement = document.querySelector('.folium-map');
        if (mapElement && mapElement._leaflet_map) {{
            const map = mapElement._leaflet_map;
            const geojsonLayer = map._layers[Object.keys(map._layers)[0]];
            if (geojsonLayer) {{
                addClickEventToMap(geojsonLayer); 
            }}
        }}
    }});
    </script>
"""
components.html(js_code, height=0)

# Main Title and Introduction
st.title("📊 Spain Sales Dashboard")
st.markdown("Explore the data by interacting with the map and charts below.")

# Centered and colored "TODAY" section
st.markdown("""
    <style>
    .today-container {
        background-color: #cc0641;  /* Background color */
        color: white;  /* Text color */
        padding: 5px;
        text-align: center;  /* Center align text */
        border-radius: 10px;  /* Rounded corners */
        font-size: 24px;  /* Font size */
        font-weight: bold;  /* Bold font */
        margin-top: 20px;  /* Space above the box */
    }
    </style>
    <div class="today-container">
        TODAY
    </div>
    """, unsafe_allow_html=True)


grouped_sfdatapy = sfdatapy.groupby('Region_Descr').agg({
    'Trial_Activated': 'sum',
    'HA_Sales_Units': 'sum',
    'HA_Sales_Value': 'sum'
}).reset_index()

grouped_sfdatatoday = sfdatatoday.groupby('Region_Descr').agg({
    'Trial_Activated': 'sum',
    'HA_Sales_Units': 'sum',
    'HA_Sales_Value': 'sum'
}).reset_index()
sf_grouped_forecast = forecast.groupby('Region').agg({
    'TR Piezas': lambda x: x.sum() * daily_gen_rate,
    'Sales U': lambda x: x.sum() * daily_entr_rate,
    'Sales €': lambda x: x.sum() * daily_entr_rate
}).reset_index()

# Clean and prepare the region names
grouped_sfdatatoday = grouped_sfdatatoday[grouped_sfdatatoday['Region_Descr'] != 'N/A N/A']
grouped_sfdatatoday['Region_Descr'] = grouped_sfdatatoday['Region_Descr'].str.upper()

grouped_sfdatapy = grouped_sfdatapy[grouped_sfdatapy['Region_Descr'] != 'N/A N/A']
grouped_sfdatapy['Region_Descr'] = grouped_sfdatapy['Region_Descr'].str.upper()

sf_grouped_forecast = sf_grouped_forecast[sf_grouped_forecast['Region'] != 'N/A N/A']
sf_grouped_forecast['Region'] = sf_grouped_forecast['Region'].str.upper()

# Calculate PZ TOT for current year, forecast, and previous year
grouped_sfdatatoday['PZ_TOT_Act'] = grouped_sfdatatoday['Trial_Activated'] 
grouped_sfdatatoday['Sales_U_Act'] = grouped_sfdatatoday['HA_Sales_Units']
grouped_sfdatatoday['Sales_E_Act'] = grouped_sfdatatoday['HA_Sales_Value']

sf_grouped_forecast['PZ_TOT_FC'] = sf_grouped_forecast['TR Piezas'] 
sf_grouped_forecast['Sales_U_FC'] = sf_grouped_forecast['Sales U']
sf_grouped_forecast['Sales_E_FC'] = sf_grouped_forecast['Sales €']

grouped_sfdatapy['PZ_TOT_PY'] = grouped_sfdatapy['Trial_Activated'] 
grouped_sfdatapy['Sales_U_PY'] = grouped_sfdatapy['HA_Sales_Units']
grouped_sfdatapy['Sales_E_PY'] = grouped_sfdatapy['HA_Sales_Value']
# Merge the dataframes based on region
sf_merged_data = pd.merge(grouped_sfdatatoday, sf_grouped_forecast, left_on='Region_Descr', right_on='Region', how='left')
sf_merged_data = pd.merge(sf_merged_data, grouped_sfdatapy, on='Region_Descr', suffixes=('_today', '_yesterday'))
sf_merged_data.fillna(0, inplace=True)

# Function to safely calculate percentage changes
def calculate_safe_percentage_change(numerator, denominator):
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100

# Calculate deltas and percentage changes for PZ TOT
sf_merged_data['PZ_TOT_Delta_vs_FC'] = sf_merged_data['PZ_TOT_Act'] - sf_merged_data['PZ_TOT_FC']
sf_merged_data['PCT_PZ_TOT_Delta_vs_FC'] = sf_merged_data.apply(lambda x: calculate_safe_percentage_change(x['PZ_TOT_Delta_vs_FC'], x['PZ_TOT_FC']), axis=1)
sf_merged_data['PZ_TOT_Delta_vs_PY'] = sf_merged_data['PZ_TOT_Act'] - sf_merged_data['PZ_TOT_PY']
sf_merged_data['PCT_PZ_TOT_Delta_vs_PY'] = sf_merged_data.apply(lambda x: calculate_safe_percentage_change(x['PZ_TOT_Delta_vs_PY'], x['PZ_TOT_PY']), axis=1)

# Ensure that the correct column names are used for current year (CY), forecast (FC), and YESTERDAY (PY) data.
sf_merged_data['Sales_U_Delta_vs_FC'] = sf_merged_data['Sales_U_Act'] - sf_merged_data['Sales_U_FC']
sf_merged_data['PCT_Sales_U_Delta_vs_FC'] = sf_merged_data.apply(lambda x: calculate_safe_percentage_change(x['Sales_U_Delta_vs_FC'], x['Sales_U_FC']), axis=1)
sf_merged_data['Sales_U_Delta_vs_PY'] = sf_merged_data['Sales_U_Act'] - sf_merged_data['Sales_U_PY']
sf_merged_data['PCT_Sales_U_Delta_vs_PY'] = sf_merged_data.apply(lambda x: calculate_safe_percentage_change(x['Sales_U_Delta_vs_PY'], x['Sales_U_PY']), axis=1)

# Similarly, ensure correct references for Sales € (Euro) values
sf_merged_data['Sales_E_Delta_vs_FC'] = sf_merged_data['Sales_E_Act'] - sf_merged_data['Sales_E_FC']
sf_merged_data['PCT_Sales_E_Delta_vs_FC'] = sf_merged_data.apply(lambda x: calculate_safe_percentage_change(x['Sales_E_Delta_vs_FC'], x['Sales_E_FC']), axis=1)
sf_merged_data['Sales_E_Delta_vs_PY'] = sf_merged_data['Sales_E_Act'] - sf_merged_data['Sales_E_PY']
sf_merged_data['PCT_Sales_E_Delta_vs_PY'] = sf_merged_data.apply(lambda x: calculate_safe_percentage_change(x['Sales_E_Delta_vs_PY'], x['Sales_E_PY']), axis=1)

# Prepare data for each block using the specific columns you need
sf_data_pz = sf_merged_data[['Region_Descr', 'PZ_TOT_Act', 'PCT_PZ_TOT_Delta_vs_FC', 'PCT_PZ_TOT_Delta_vs_PY']].copy()
sf_data_pz.columns = ['Region', 'Actual', '%Delta vs FC', '%Delta vs PY']

sf_data_sales_u = sf_merged_data[['Region_Descr', 'Sales_U_Act', 'PCT_Sales_U_Delta_vs_FC', 'PCT_Sales_U_Delta_vs_PY']].copy()
sf_data_sales_u.columns = ['Region', 'Actual', '%Delta vs FC', '%Delta vs PY']

sf_data_sales_e = sf_merged_data[['Region_Descr', 'Sales_E_Act', 'PCT_Sales_E_Delta_vs_FC', 'PCT_Sales_E_Delta_vs_PY']].copy()
sf_data_sales_e.columns = ['Region', 'Actual', '%Delta vs FC', '%Delta vs PY']
# Calculate total absolute values for PZ TOT, Sales U, and Sales E
total_pz_tot_act = sf_merged_data['PZ_TOT_Act'].sum()
total_pz_tot_fc = sf_merged_data['PZ_TOT_FC'].sum()
total_pz_tot_py = sf_merged_data['PZ_TOT_PY'].sum()

total_sales_u_act = sf_merged_data['Sales_U_Act'].sum()
total_sales_u_fc = sf_merged_data['Sales_U_FC'].sum()
total_sales_u_py = sf_merged_data['Sales_U_PY'].sum()

total_sales_e_act = sf_merged_data['Sales_E_Act'].sum()
total_sales_e_fc = sf_merged_data['Sales_E_FC'].sum()
total_sales_e_py = sf_merged_data['Sales_E_PY'].sum()

# Recalculate percentage changes based on total values
total_pct_pz_tot_vs_fc = calculate_safe_percentage_change(total_pz_tot_act - total_pz_tot_fc, total_pz_tot_fc)
total_pct_pz_tot_vs_py = calculate_safe_percentage_change(total_pz_tot_act - total_pz_tot_py, total_pz_tot_py)

total_pct_sales_u_vs_fc = calculate_safe_percentage_change(total_sales_u_act - total_sales_u_fc, total_sales_u_fc)
total_pct_sales_u_vs_py = calculate_safe_percentage_change(total_sales_u_act - total_sales_u_py, total_sales_u_py)

total_pct_sales_e_vs_fc = calculate_safe_percentage_change(total_sales_e_act - total_sales_e_fc, total_sales_e_fc)
total_pct_sales_e_vs_py = calculate_safe_percentage_change(total_sales_e_act - total_sales_e_py, total_sales_e_py)

# Create the total rows for each category using the recalculated percentages
sf_total_pz_row = pd.DataFrame([[
    'Total',
    total_pz_tot_act,
    total_pct_pz_tot_vs_fc,
    total_pct_pz_tot_vs_py
]], columns=sf_data_pz.columns)

sf_total_sales_u_row = pd.DataFrame([[
    'Total',
    total_sales_u_act,
    total_pct_sales_u_vs_fc,
    total_pct_sales_u_vs_py
]], columns=sf_data_sales_u.columns)

sf_total_sales_e_row = pd.DataFrame([[
    'Total',
    total_sales_e_act,
    total_pct_sales_e_vs_fc,
    total_pct_sales_e_vs_py
]], columns=sf_data_sales_e.columns)

# Append totals as a new row at the end of each DataFrame
sf_data_pz = pd.concat([sf_data_pz, sf_total_pz_row], ignore_index=True)
sf_data_sales_u = pd.concat([sf_data_sales_u, sf_total_sales_u_row], ignore_index=True)
sf_data_sales_e = pd.concat([sf_data_sales_e, sf_total_sales_e_row], ignore_index=True)

# Function to apply styles to DataFrame
def style_df(df, is_currency=False):
    styled_df = df.style.format({
        'Actual': "€{:,.2f}" if is_currency else "{:,.2f}",
        '%Delta vs FC': "{:+.1f}%",
        '%Delta vs PY': "{:+.1f}%",
    }).applymap(lambda x: 'color: red;' if isinstance(x, (int, float)) and x < 0 else (
        'color: green;' if isinstance(x, (int, float)) and x > 0 else ''
    ), subset=['%Delta vs FC', '%Delta vs PY']
    ).set_properties(**{
        'text-align': 'right',
        'font-weight': 'bold',
        'padding': '6px',  # Reduced padding for compactness
    }, subset=['Actual', '%Delta vs FC', '%Delta vs PY']).set_properties(**{
        'text-align': 'left',
        'padding': '6px',  # Reduced padding for compactness
    }, subset=['Region']) \
    .set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#cc0641'), ('color', 'white'), ('font-weight', 'bold'), ('padding', '6px'), ('border-bottom', '2px solid #333')]},
        {'selector': 'td', 'props': [('padding', '6px'), ('border-bottom', '1px solid #ddd')]},
        {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '100%'), ('table-layout', 'fixed')]},
    ])
    
    return styled_df

# Displaying the blocks in columns using Streamlit's native table rendering
col1, col2, col3 = st.columns([1, 1, 1], gap="small")  # Equal column width, small gap between columns

with col1:
    st.markdown("<h3 style='text-align: left; color: #cc0641; font-size: 16px;'>PZ TOT Act</h3>", unsafe_allow_html=True)
    st.write(style_df(sf_data_pz).hide(axis="index").to_html(), unsafe_allow_html=True)

with col2:
    st.markdown("<h3 style='text-align: left; color: #cc0641; font-size: 16px;'>Sales U Act</h3>", unsafe_allow_html=True)
    st.write(style_df(sf_data_sales_u).hide(axis="index").to_html(), unsafe_allow_html=True)

with col3:
    st.markdown("<h3 style='text-align: left; color: #cc0641; font-size: 16px;'>Sales € Act</h3>", unsafe_allow_html=True)
    st.write(style_df(sf_data_sales_e, is_currency=True).hide(axis="index").to_html(), unsafe_allow_html=True)

# Add CSS to align the columns and make them of equal height
st.markdown(
    """
    <style>
    /* Ensure all containers have the same height */
    div[data-testid="column"] > div {
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    /* Ensure all tables have the same width */
    .block-container .stDataFrame, .block-container table {
        width: 100%;
    }

    /* Align headers and tables inside each block to top */
    .block-container h3 {
        margin-bottom: 10px;
    }
    
    .block-container .stDataFrame {
        flex-grow: 1;
    }

    /* Ensure table cells have a consistent padding */
    .block-container td, .block-container th {
        padding: 6px;
        font-size: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Centered and colored "MTD" section
st.markdown("""
    <style>
    .today-container {
        background-color: #cc0641;  /* Background color */
        color: white;  /* Text color */
        padding: 5px;
        text-align: center;  /* Center align text */
        border-radius: 10px;  /* Rounded corners */
        font-size: 24px;  /* Font size */
        font-weight: bold;  /* Bold font */
        margin-top: 20px;  /* Space above the box */
    }
    </style>
    <div class="today-container">
        MTD
    </div>
    """, unsafe_allow_html=True)

# Group by region in cleaned_data and py_sales
grouped_cleaned_data = mtd_cleaned_data.groupby('Region_Descr').agg({
    'Trial_Activated': 'sum',
    'Direct_Orders_Units': 'sum',
    'HA_Sales_Units': 'sum',
    'HA_Sales_Value': 'sum',
    'Appointments_Completed': 'sum'  # Include Appointments_Completed

}).reset_index()

grouped_py_sales = mtd_py_sales.groupby('Region_Descr').agg({
    'Trial_Activated': 'sum',
    'Direct_Orders_Units': 'sum',
    'HA_Sales_Units': 'sum',
    'HA_Sales_Value': 'sum',
    'Appointments_Completed': 'sum'  # Include Appointments_Completed
}).reset_index()

# Group by region in forecast
grouped_forecast = forecast.groupby('Region').agg({
    'TR Piezas': lambda x: x.sum() * gen_rate,
    'DO Piezas': lambda x: x.sum() * gen_rate,
    'Sales U': lambda x: x.sum() * entr_rate,
    'Sales €': lambda x: x.sum() * entr_rate
}).reset_index()

grouped_cleaned_data = grouped_cleaned_data[grouped_cleaned_data['Region_Descr'] != 'N/A N/A']
grouped_py_sales = grouped_py_sales[grouped_py_sales['Region_Descr'] != 'N/A N/A']
grouped_cleaned_data['Region_Descr'] = grouped_cleaned_data['Region_Descr'].str.upper()
grouped_forecast['Region'] = grouped_forecast['Region'].str.upper()
grouped_py_sales['Region_Descr'] = grouped_py_sales['Region_Descr'].str.upper()
# Calculate PZ TOT for current year, forecast, and previous year
grouped_cleaned_data['PZ_TOT_Act'] = grouped_cleaned_data['Trial_Activated'] + grouped_cleaned_data['Direct_Orders_Units']
grouped_cleaned_data['Sales_U_Act'] = grouped_cleaned_data['HA_Sales_Units']
grouped_cleaned_data['Sales_E_Act'] = grouped_cleaned_data['HA_Sales_Value']

grouped_forecast['PZ_TOT_FC'] = grouped_forecast['TR Piezas'] + grouped_forecast['DO Piezas']
grouped_forecast['Sales_U_FC'] = grouped_forecast['Sales U']
grouped_forecast['Sales_E_FC'] = grouped_forecast['Sales €']

grouped_py_sales['PZ_TOT_PY'] = grouped_py_sales['Trial_Activated'] + grouped_py_sales['Direct_Orders_Units']
grouped_py_sales['Sales_U_PY'] = grouped_py_sales['HA_Sales_Units']
grouped_py_sales['Sales_E_PY'] = grouped_py_sales['HA_Sales_Value']
# Merge the dataframes based on region
merged_data = pd.merge(grouped_cleaned_data, grouped_forecast, left_on='Region_Descr', right_on='Region', how='left')
merged_data = pd.merge(merged_data, grouped_py_sales, on='Region_Descr', suffixes=('_CY', '_PY'))
merged_data.fillna(0, inplace=True)
# Function to safely calculate percentage changes
def calculate_safe_percentage_change(numerator, denominator):
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100

# Calculate deltas and percentage changes for PZ TOT
merged_data['PZ_TOT_Delta_vs_FC'] = merged_data['PZ_TOT_Act'] - merged_data['PZ_TOT_FC']
merged_data['PCT_PZ_TOT_Delta_vs_FC'] = merged_data.apply(lambda x: calculate_safe_percentage_change(x['PZ_TOT_Delta_vs_FC'], x['PZ_TOT_FC']), axis=1)
merged_data['PZ_TOT_Delta_vs_PY'] = merged_data['PZ_TOT_Act'] - merged_data['PZ_TOT_PY']
merged_data['PCT_PZ_TOT_Delta_vs_PY'] = merged_data.apply(lambda x: calculate_safe_percentage_change(x['PZ_TOT_Delta_vs_PY'], x['PZ_TOT_PY']), axis=1)

# Ensure that the correct column names are used for current year (CY), forecast (FC), and previous year (PY) data.
merged_data['Sales_U_Delta_vs_FC'] = merged_data['Sales_U_Act'] - merged_data['Sales_U_FC']
merged_data['PCT_Sales_U_Delta_vs_FC'] = merged_data.apply(lambda x: calculate_safe_percentage_change(x['Sales_U_Delta_vs_FC'], x['Sales_U_FC']), axis=1)
merged_data['Sales_U_Delta_vs_PY'] = merged_data['Sales_U_Act'] - merged_data['Sales_U_PY']
merged_data['PCT_Sales_U_Delta_vs_PY'] = merged_data.apply(lambda x: calculate_safe_percentage_change(x['Sales_U_Delta_vs_PY'], x['Sales_U_PY']), axis=1)

# Similarly, ensure correct references for Sales € (Euro) values
merged_data['Sales_E_Delta_vs_FC'] = merged_data['Sales_E_Act'] - merged_data['Sales_E_FC']
merged_data['PCT_Sales_E_Delta_vs_FC'] = merged_data.apply(lambda x: calculate_safe_percentage_change(x['Sales_E_Delta_vs_FC'], x['Sales_E_FC']), axis=1)
merged_data['Sales_E_Delta_vs_PY'] = merged_data['Sales_E_Act'] - merged_data['Sales_E_PY']
merged_data['PCT_Sales_E_Delta_vs_PY'] = merged_data.apply(lambda x: calculate_safe_percentage_change(x['Sales_E_Delta_vs_PY'], x['Sales_E_PY']), axis=1)

# Prepare data for each block using the specific columns you need
data_pz = merged_data[['Region_Descr', 'PZ_TOT_Act', 'PCT_PZ_TOT_Delta_vs_FC', 'PCT_PZ_TOT_Delta_vs_PY']].copy()
data_pz.columns = ['Region', 'Actual', '%Delta vs FC', '%Delta vs PY']

data_sales_u = merged_data[['Region_Descr', 'Sales_U_Act', 'PCT_Sales_U_Delta_vs_FC', 'PCT_Sales_U_Delta_vs_PY']].copy()
data_sales_u.columns = ['Region', 'Actual', '%Delta vs FC', '%Delta vs PY']

data_sales_e = merged_data[['Region_Descr', 'Sales_E_Act', 'PCT_Sales_E_Delta_vs_FC', 'PCT_Sales_E_Delta_vs_PY']].copy()
data_sales_e.columns = ['Region', 'Actual', '%Delta vs FC', '%Delta vs PY']

# Calculate column totals
# Calculate total absolute values for PZ TOT, Sales U, and Sales E
total_pz_tot_act = merged_data['PZ_TOT_Act'].sum()
total_pz_tot_fc = merged_data['PZ_TOT_FC'].sum()
total_pz_tot_py = merged_data['PZ_TOT_PY'].sum()

total_sales_u_act = merged_data['Sales_U_Act'].sum()
total_sales_u_fc = merged_data['Sales_U_FC'].sum()
total_sales_u_py = merged_data['Sales_U_PY'].sum()

total_sales_e_act = merged_data['Sales_E_Act'].sum()
total_sales_e_fc = merged_data['Sales_E_FC'].sum()
total_sales_e_py = merged_data['Sales_E_PY'].sum()

# Recalculate percentage changes based on total values
total_pct_pz_tot_vs_fc = calculate_safe_percentage_change(total_pz_tot_act - total_pz_tot_fc, total_pz_tot_fc)
total_pct_pz_tot_vs_py = calculate_safe_percentage_change(total_pz_tot_act - total_pz_tot_py, total_pz_tot_py)

total_pct_sales_u_vs_fc = calculate_safe_percentage_change(total_sales_u_act - total_sales_u_fc, total_sales_u_fc)
total_pct_sales_u_vs_py = calculate_safe_percentage_change(total_sales_u_act - total_sales_u_py, total_sales_u_py)

total_pct_sales_e_vs_fc = calculate_safe_percentage_change(total_sales_e_act - total_sales_e_fc, total_sales_e_fc)
total_pct_sales_e_vs_py = calculate_safe_percentage_change(total_sales_e_act - total_sales_e_py, total_sales_e_py)

# Create the total rows for each category using the recalculated percentages

# For PZ TOT
total_pz_row = pd.DataFrame([[
    'Total',
    total_pz_tot_act,
    total_pct_pz_tot_vs_fc,
    total_pct_pz_tot_vs_py
]], columns=data_pz.columns)

# For Sales U
total_sales_u_row = pd.DataFrame([[
    'Total',
    total_sales_u_act,
    total_pct_sales_u_vs_fc,
    total_pct_sales_u_vs_py
]], columns=data_sales_u.columns)

# For Sales E
total_sales_e_row = pd.DataFrame([[
    'Total',
    total_sales_e_act,
    total_pct_sales_e_vs_fc,
    total_pct_sales_e_vs_py
]], columns=data_sales_e.columns)

# Append totals as a new row at the end of each DataFrame
data_pz = pd.concat([data_pz, total_pz_row], ignore_index=True)
data_sales_u = pd.concat([data_sales_u, total_sales_u_row], ignore_index=True)
data_sales_e = pd.concat([data_sales_e, total_sales_e_row], ignore_index=True)

# Function to apply styles to DataFrame
def style_df(df, is_currency=False):
    styled_df = df.style.format({
        'Actual': "€{:,.0f}" if is_currency else "{:,.0f}",
        '%Delta vs FC': "{:+.1f}%",
        '%Delta vs PY': "{:+.1f}%",
    }).applymap(lambda x: 'color: red;' if isinstance(x, (int, float)) and x < 0 else (
        'color: green;' if isinstance(x, (int, float)) and x > 0 else ''
    ), subset=['%Delta vs FC', '%Delta vs PY']
    ).set_properties(**{
        'text-align': 'right',
        'font-weight': 'bold',
        'padding': '6px',  # Reduced padding for compactness
    }, subset=['Actual', '%Delta vs FC', '%Delta vs PY']).set_properties(**{
        'text-align': 'left',
        'padding': '6px',  # Reduced padding for compactness
    }, subset=['Region']) \
    .set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#cc0641'), ('color', 'white'), ('font-weight', 'bold'), ('padding', '6px'), ('border-bottom', '2px solid #333')]},
        {'selector': 'td', 'props': [('padding', '6px'), ('border-bottom', '1px solid #ddd')]},
        {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '100%'), ('table-layout', 'fixed')]},
    ])
    
    return styled_df

# Displaying the blocks in columns using Streamlit's native table rendering
col1, col2, col3 = st.columns([1, 1, 1], gap="small")  # Equal column width, small gap between columns

with col1:
    st.markdown("<h3 style='text-align: left; color: #cc0641; font-size: 16px;'>PZ TOT Act</h3>", unsafe_allow_html=True)
    st.write(style_df(data_pz).hide(axis="index").to_html(), unsafe_allow_html=True)

with col2:
    st.markdown("<h3 style='text-align: left; color: #cc0641; font-size: 16px;'>Sales U Act</h3>", unsafe_allow_html=True)
    st.write(style_df(data_sales_u).hide(axis="index").to_html(), unsafe_allow_html=True)

with col3:
    st.markdown("<h3 style='text-align: left; color: #cc0641; font-size: 16px;'>Sales € Act</h3>", unsafe_allow_html=True)
    st.write(style_df(data_sales_e, is_currency=True).hide(axis="index").to_html(), unsafe_allow_html=True)

# Add CSS to align the columns and make them of equal height
st.markdown(
    """
    <style>
    /* Ensure all containers have the same height */
    div[data-testid="column"] > div {
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    /* Ensure all tables have the same width */
    .block-container .stDataFrame, .block-container table {
        width: 100%;
    }

    /* Align headers and tables inside each block to top */
    .block-container h3 {
        margin-bottom: 5px;
    }
    
    .block-container .stDataFrame {
        flex-grow: 1;
    }

    /* Ensure table cells have a consistent padding */
    .block-container td, .block-container th {
        padding: 6px;
        font-size: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Function to calculate percentage change handling division by zero cases
def calculate_percentage_change(cy_value, py_value):
    if py_value == 0:
        if cy_value == 0:
            return 0.0  # No change
        elif cy_value > 0:
            return 100.0  # +100% increase when PY is 0 and CY is positive
        else:
            return -100.0  # -100% decrease when PY is 0 and CY is negative
    else:
        return ((cy_value - py_value) / py_value) * 100

# Function to add arrows and percentage changes in the details section
def add_trend_arrow(change_value):
    if pd.isna(change_value):
        return "N/A"
    elif change_value > 0:
        return f"🟢 +{change_value:.2f}%"
    elif change_value < 0:
        return f"🔴 {change_value:.2f}%"
    else:
        return "No Change"

# Create funnel chart function
def create_funnel_chart(data, title="Sales Funnel"):
    scaled_data = [np.log1p(x) for x in data]  # Logarithmic scaling
    conversion_rates = [100.0]
    for i in range(1, len(data)):
        conversion_rate = (data[i] / data[i-1]) * 100 if data[i-1] != 0 else 0
        conversion_rates.append(conversion_rate)
    
    funnel_data = {
        "Stage": [
            "Agenda Appointments (Heads)", 
            "Opportunity Tests (Heads)", 
            "Trial Activated (Units)", 
            "Trials closed (Units)",
            "Sales from Trials (Units)"
        ],
        "Value": scaled_data,
        "RealValue": data,
        "ConversionRate": conversion_rates
    }

    fig = px.funnel_area(
        names=funnel_data["Stage"],
        values=funnel_data["Value"],
        title=title,
        color_discrete_sequence=["#cc0641", "#ff66b2", "#660033", "#cc0641", "#ff66b2"],
    )

    fig.update_layout(
        title_font=dict(size=24, color="#333"),
        margin=dict(l=20, r=20, t=60, b=20),
        height=450,
        width=700,
        funnelmode="stack",
        showlegend=False,
    )
   
    fig.update_traces(
        textinfo="label+text",
        insidetextfont=dict(size=18, color="white"),
        texttemplate='%{label}<br>%{text}',
        text=[f"{rate:.1f}% ({value:,.0f})" for rate, value in zip(funnel_data["ConversionRate"], funnel_data["RealValue"])],
    )

    return fig


# Columns to calculate percentage change for
comparison_columns = [
    'HA_Sales_Units', 'HA_Sales_Value', 'Appointments_Completed', 
    'Appointments_Cancelled', 'Trial_Activated', 
    'Net_Trial_Closed_Same_Month_Units_Side', 'HA_Sales_from_Trial_Same_Month_Units'
]

# Step 1: Identify duplicates in mtd_cleaned_data and mtd_py_sales
cleaned_data_duplicates = mtd_cleaned_data.duplicated(subset=['Shop_Code', 'ISO Week'])
py_sales_duplicates = mtd_py_sales.duplicated(subset=['Shop_Code', 'ISO Week'])

# Step 2: Remove duplicates in both dataframes
mtd_cleaned_data_unique = mtd_cleaned_data.drop_duplicates(subset=['Shop_Code', 'ISO Week'], keep='first')
mtd_py_sales_unique = mtd_py_sales.drop_duplicates(subset=['Shop_Code', 'ISO Week'], keep='first')

# Step 3: Calculate total sales before merging (from cleaned unique data)
total_sales_before_merge = mtd_cleaned_data_unique['HA_Sales_Value'].sum()

# Step 4: Merge the two datasets on common keys without introducing duplicates
merged_data1 = pd.merge(
    mtd_cleaned_data_unique,
    mtd_py_sales_unique,
    on=['Shop_Code', 'ISO Week'],  # Adjust keys if necessary
    suffixes=('_CY', '_PY'),
    how='inner',  # Ensure only matching records are merged
    indicator=True  # Add indicator to track which rows matched
)

# Step 5: Ensure no duplicate rows were introduced after the merge
post_merge_sales_sum = merged_data1['HA_Sales_Value_CY'].sum()

# Step 6: Continue with analysis using the correct total from the cleaned data
correct_total_sales = mtd_cleaned_data_unique['HA_Sales_Value'].sum()


# After merging, calculate the total again and compare
post_merge_sales_sum = merged_data1['HA_Sales_Value_CY'].sum()
# Apply the calculation to each relevant column
for col in comparison_columns:
    merged_data1[f'{col}_Change'] = merged_data1.apply(
        lambda row: calculate_percentage_change(row[f'{col}_CY'], row[f'{col}_PY']),
        axis=1
    )

# Merge the changes back into mtd_cleaned_data
mtd_cleaned_data = mtd_cleaned_data.merge(
    merged_data1[['Shop_Code', 'ISO Week'] + [f'{col}_Change' for col in comparison_columns]],
    on=['Shop_Code', 'ISO Week'],
    how='left'
)

# Aggregate the key KPIs for the entire dataset
total_sales_all = mtd_cleaned_data['HA_Sales_Value'].sum()
total_sales_all
total_units_sold_all = mtd_cleaned_data['HA_Sales_Units'].sum()
total_shops_all = mtd_cleaned_data['Shop_Code'].nunique()
total_appointments_completed_all = mtd_cleaned_data['Appointments_Completed'].sum()

# Define missing total variables for funnel and trendline analysis
total_agenda_appointments_all = mtd_cleaned_data['Agenda_Appointments_Heads'].sum()
total_opportunity_tests_all = mtd_cleaned_data['FP_OppTest_Heads_Month'].sum()
total_trials_activated_all = mtd_cleaned_data['Trial_Activated'].sum()
total_trials_closed_all = mtd_cleaned_data['Net_Trial_Closed_Same_Month_Units_Side'].sum()
total_trials_sales_all = mtd_cleaned_data['HA_Sales_from_Trial_Same_Month_Units'].sum()


# Map rendering
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Sales Map")
    map_data = st_folium(m, width=1000, height=600, key="main_map")

    # Initialize session state for selected_province
    if 'selected_province' not in st.session_state:
        st.session_state['selected_province'] = None

    if map_data is not None and 'last_object_clicked_tooltip' in map_data:
        tooltip_content = map_data['last_object_clicked_tooltip']
        if tooltip_content:
            try:
                # Extract province name
                match = re.search(r"Province:\s*([^\n]+)", tooltip_content)
                if match:
                    province_name = match.group(1).strip()
                    # Only update the province if it matches one in the dataset
                    if province_name in mtd_cleaned_data['GeoJSON_Province'].unique():
                        st.session_state['selected_province'] = province_name

                        # Filter and add shop circles (functionality assumed to exist)
                        selected_province_data = mtd_cleaned_data[mtd_cleaned_data['GeoJSON_Province'] == province_name]
                        add_shop_circles_to_map(selected_province_data)

                        # Update the map
                        map_data = st_folium(m, width=1000, height=600, key="main_map_updated")
                    else:
                        st.session_state['selected_province'] = None  # Reset if invalid province
            except Exception as e:
                st.write(f"Error parsing tooltip content: {e}")

# Display Province Details based on the selected province
with col2:
    selected_province = st.session_state.get('selected_province')

    if selected_province:
        # Province-specific details
        province_details = mtd_cleaned_data[mtd_cleaned_data['GeoJSON_Province'] == selected_province]

        if not province_details.empty:
            st.subheader(f"Details for {selected_province}")
            total_sales = province_details['HA_Sales_Value'].sum()
            total_units_sold = province_details['HA_Sales_Units'].sum()
            total_shops = province_details['Shop_Code'].nunique()
            total_appointments_completed = province_details['Appointments_Completed'].sum()

            # Updated province details styling with background color and better padding
            st.markdown(
                f"""
                <div style="background-color:#f7f7f7; padding:5px; border-radius:5px;">
                    <p><b>Total Sales (€):</b> {total_sales:,.2f} {add_trend_arrow(province_details['HA_Sales_Value_Change'].mean())}</p>
                    <p><b>Total Units Sold:</b> {total_units_sold:,.2f} {add_trend_arrow(province_details['HA_Sales_Units_Change'].mean())}</p>
                    <p><b>Total Shops:</b> {total_shops:,.2f}</p>
                    <p><b>Total Appointments Completed:</b> {total_appointments_completed:,.2f} {add_trend_arrow(province_details['Appointments_Completed_Change'].mean())}</p>
                </div>
                """, unsafe_allow_html=True)

            st.subheader(f"Worst Performing Shops in {selected_province}")
            # Group the data by Shop_Code (or Shop_Descr) and sum the sales
            worst_shops_grouped = province_details.groupby(['Shop_Code', 'Shop_Descr']).agg({
                'HA_Sales_Value': 'sum'
            }).reset_index()
            worst_shops_grouped['HA_Sales_Value'] = worst_shops_grouped['HA_Sales_Value'].round(2)
            worst_shops_grouped = worst_shops_grouped.sort_values(by='HA_Sales_Value', ascending=True).drop_duplicates(subset=['Shop_Code']).head(8)
            st.write(worst_shops_grouped[['Shop_Code', 'Shop_Descr', 'HA_Sales_Value']], use_container_width=True)
    else:
        # Default to showing details of all regions if no province is selected
        st.subheader("Details of All Regions")
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6; padding:5px; border-radius:5px;">
                <p><b>Total Sales (€):</b> {total_sales_all:,.2f} {add_trend_arrow(mtd_cleaned_data['HA_Sales_Value_Change'].mean())}</p>
                <p><b>Total Units Sold:</b> {total_units_sold_all:,.2f} {add_trend_arrow(mtd_cleaned_data['HA_Sales_Units_Change'].mean())}</p>
                <p><b>Total Shops:</b> {total_shops_all:,.2f}</p>
                <p><b>Total Appointments Completed:</b> {total_appointments_completed_all:,.2f} {add_trend_arrow(mtd_cleaned_data['Appointments_Completed_Change'].mean())}</p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Worst Performing Shops in All Regions")
        worst_shops = mtd_cleaned_data.sort_values(by='HA_Sales_Value', ascending=True).drop_duplicates(subset=['Shop_Code']).head(8)
        st.write(worst_shops[['Shop_Code', 'Shop_Descr', 'HA_Sales_Value']], use_container_width=True)

st.markdown("---") 
st.subheader("Sales Funnel and Interactive Trendline")

# Create columns with the ratio 3:2
col1, col2 = st.columns([2, 3])

# Funnel Chart in the first column
with col1:
    if selected_province:
        province_details = mtd_cleaned_data[mtd_cleaned_data['GeoJSON_Province'] == selected_province]

        if not province_details.empty:
            total_agenda_appointments = province_details['Agenda_Appointments_Heads'].sum()
            total_opportunity_tests = province_details['FP_OppTest_Heads_Month'].sum()
            total_trials_activated = province_details['Trial_Activated'].sum()
            total_trials_closed = province_details['Net_Trial_Closed_Same_Month_Units_Side'].sum()
            total_trials_sales = province_details['HA_Sales_from_Trial_Same_Month_Units'].sum()
            # Create the funnel chart for the selected province
            funnel_fig = create_funnel_chart([
                total_agenda_appointments, total_opportunity_tests, 
                total_trials_activated, 
                total_trials_closed, 
                total_trials_sales
            ], title=f"Sales Funnel for {selected_province}")

            st.plotly_chart(funnel_fig, use_container_width=True)
    else:
        # Create the funnel chart for all regions
        funnel_fig_all = create_funnel_chart([
            total_agenda_appointments_all, total_opportunity_tests_all, 
            total_trials_activated_all, total_trials_closed_all, 
            total_trials_sales_all
        ], title="Total Sales Funnel for All Regions")

        st.plotly_chart(funnel_fig_all, use_container_width=True)

# Trendline Chart in the second column
with col2:
    metric_options = {
        "Agenda Appointments (Heads)": 'Agenda_Appointments_Heads',
        "Opportunity Tests (Heads)": 'FP_OppTest_Heads_Month',
        "Trial Activated (Units)": 'Trial_Activated',
        "Trials closed (Units)": 'Net_Trial_Closed_Same_Month__Units__Side',
        "Sales from Trials (Units)": 'HA_Sales_from_Trial_Same_Month_Units'
    }
    
    selected_metric = st.selectbox("Select a Metric to View Trend Over Weeks", list(metric_options.keys()))
    selected_column_cy = metric_options[selected_metric] + "_CY"
    selected_column_py = metric_options[selected_metric] + "_PY"

    # Prepare data for CY and PY trends
    metric_trend_cy = merged_data1.groupby('ISO Week')[selected_column_cy].sum().reset_index()
    metric_trend_py = merged_data1.groupby('ISO Week')[selected_column_py].sum().reset_index()

    # Ensure both datasets have the full range of weeks
    full_weeks = pd.DataFrame({'ISO Week': range(min(metric_trend_cy['ISO Week'].min(), metric_trend_py['ISO Week'].min()), 
                                                 max(metric_trend_cy['ISO Week'].max(), metric_trend_py['ISO Week'].max()) + 1)})
    
    metric_trend_cy = pd.merge(full_weeks, metric_trend_cy, on='ISO Week', how='left').fillna(0)
    metric_trend_py = pd.merge(full_weeks, metric_trend_py, on='ISO Week', how='left').fillna(0)

    # Plot both CY and PY trends on the same figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=metric_trend_cy['ISO Week'],
        y=metric_trend_cy[selected_column_cy],
        mode='lines+markers',
        line=dict(color='#17BECF', width=3),
        marker=dict(size=8, color='#17BECF'),
        name=f"Current Year {selected_metric}"
    ))

    fig.add_trace(go.Scatter(
        x=metric_trend_py['ISO Week'],
        y=metric_trend_py[selected_column_py],
        mode='lines+markers',
        line=dict(color='#FF6699', width=3),
        marker=dict(size=8, color='#FF6699'),
        name=f"Previous Year {selected_metric}"
    ))

    # Trendline Chart enhancement with improved layout and color scheme
    fig.update_layout(
        title=f'Trend of {selected_metric} Over ISO Weeks (CY vs PY)',
        xaxis_title='ISO Week',
        yaxis_title=selected_metric,
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        template='plotly_white',
        height=450,
        margin=dict(l=10, r=10, t=50, b=50),
        hovermode='x unified',
        font=dict(size=12)
    )

    st.plotly_chart(fig, use_container_width=True)

