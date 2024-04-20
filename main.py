import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def encode_features(df):
    feature_dicts = {
        'Country of Origin': {v: k for k, v in enumerate(df['Country of Origin'].unique())},
        'Altitude': {v: k for k, v in enumerate(df['Altitude'].unique())},
        'Region': {v: k for k, v in enumerate(df['Region'].unique())},
        'Variety': {v: k for k, v in enumerate(df['Variety'].unique())},
        'Processing Method': {v: k for k, v in enumerate(df['Processing Method'].unique())},
        'Color': {v: k for k, v in enumerate(df['Color'].unique())}
    }

    for col, mapping in feature_dicts.items():
        df[col + ' Label'] = df[col].map(mapping)
    
    df.drop(list(feature_dicts.keys()), axis=1, inplace=True)

country_encodings = {
            'Colombia': 0,
            'Taiwan': 1,
            'Laos': 2,
            'Costa Rica': 3,
            'Guatemala': 4,
            'Tanzania, United Republic Of': 5,
            'Ethiopia': 6,
            'Thailand': 7,
            'Brazil': 8,
            'United States (Hawaii)': 9,
            'Kenya': 10,
            'Uganda': 11,
            'Indonesia': 12,
            'Peru': 13,
            'Panama': 14,
            'Nicaragua': 15,
            'Vietnam': 16,
            'Honduras': 17,
            'El Salvador': 18,
            'Madagascar': 19,
            'Mexico': 20,
            'Myanmar': 21
        }

region_options = {
            0: 'Piendamo,Cauca', 1: 'Chiayi', 2: 'Laos Borofen Plateau', 3: 'Los Santos,Tarrazu', 4: 'Popayan,Cauca', 5: 'Chimaltenango', 6: 'KILIMANJARO', 7: 'Guji', 8: 'Acatenango', 9: 'Yunlin', 10: 'tolima', 11: 'Gedeb,Yirgacheffe,Sidamo', 12: 'Shibi, Gukeng Township, Yunlin County 郵遞區號 , Taiwan (R.O.C.)', 13: 'Gukeng Township, Yunlin County', 14: 'Arusha', 15: 'Guatemala, Fraijanes, Santa Rosa', 16: '卓溪鄉Zhuoxi Township', 17: 'Chiang Mai', 18: 'Quindio', 19: 'Região Vulcânica', 20: 'Kona', 21: '壽豐鄉Shoufeng Township', 22: 'Dongshan Dist., Tainan City', 23: 'Oromia', 24: 'Southern Ethiopia Guji', 25: 'OROMIA', 26: 'Central', 27: 'Caoling , Gukeng Township, Yunlin County', 28: '秀林鄉Show Linxia Township', 29: '台灣屏東', 30: '苗栗縣', 31: 'Rwenzori', 32: 'Antigua', 33: 'Santa Rosa', 34: 'quiche', 35: '新竹縣', 36: 'Aceh Tengah', 37: 'Villa Rica', 38: 'Mbeya', 39: 'Nantou', 40: 'Campo das Vertentes', 41: 'Boquete', 42: 'Huehuetenango', 43: 'ANTIGUA GUATEMALA', 44: 'Tarrazu', 45: '( Dongshan Dist., Tainan City)', 46: 'ESTELI', 47: 'Quang Tri', 48: 'Centro, Lagunetillas-Ajuterique, Comayagua', 49: 'Ethiopia', 50: '玉里鎮Yuli Township', 51: "Ka'u district of Big Island", 52: 'Addis Ababa', 53: 'Chalatenango', 54: 'Sierra Nevada de Santa Marta', 55: 'Lintong Nihuta/Dolok Sanggul,Sumatera Utara', 56: 'Atitlán', 57: 'Itasy', 58: '新北市', 59: 'Oriente Santa rosa', 60: 'Sierra de las minas', 61: 'San Andrés, Lempira', 62: 'ARUSHA', 63: 'Sidama', 64: 'Huila', 65: 'New Oriente', 66: 'Nan', 67: 'Pereira', 68: 'HUEHUETENANGO', 69: 'Cauca', 70: "Ka'u", 71: 'Tolima', 72: 'Chiapas', 73: 'not known', 74: 'South Shan State', 75: 'San Jose, La Paz', 76: 'QUICHE', 77: 'Marcala', 78: 'Popayán Cauca', 79: 'North of Thailand', 80: 'Suan Ya Lung', 81: 'Mt Elgon', 82: 'NEW ORIENTE & HUEHUETENANGO', 83: '桃園市', 84: 'Sul de Minas', 85: 'Marcala, La Paz', 86: 'Nongluang Bolaven Plateau, Champasack, Lao PDR', 87: 'Centro, Lagunetillas - Ajuterique, Comayagua', 88: 'Los Planes de Santa Maria, La Paz', 89: 'HUILA', 90: 'ZONGOLICA, VERACRUZ', 91: 'Eatan Commune, Krong Nang District, Krong Nang Province', 92: 'Matagalpa', 93: 'west Villege', 94: 'los planes de santa maria la paz', 95: 'Apaneca - Ilamatepec', 96: 'MANTIQUEIRA / SUL DE MINAS', 97: 'Veracruz', 98: 'Matagalpa, Nicaragua', 99: 'Dalat', 100: 'Chanchamayo, La Merced', 101: 'Kona district of Big Island', 102: 'Kericho', 103: 'Corralillo Tarrazu', 104: 'Oriente', 105: 'Sumatra', 106: 'Lam Dong Province', 107: 'Mantiquira de minas', 108: 'Santander', 109: 'Coatepec, Veracruz', 110: 'occidente', 111: '臺北市', 112: 'Chiang Rai', 113: 'Alta Mogiana-Ibiraci', 114: 'Volcan Chinchontepek, San Vicente, El Salvador', 115: 'Jinotega', 116: 'Caicedonia,Valle del Cauca', 117: 'Chanchamayo, Lamerced', 118: 'Bolaven Plateau', 119: 'Volcan de San Vicente, La Paz, El Salvador'
        }

variety_encodings = {
            0: 'Castillo', 1: 'Gesha', 2: 'Java', 3: 'Red Bourbon', 4: 'Sl34+Gesha', 5: 'SL34',
            6: 'Bourbon', 7: 'Ethiopian Heirlooms', 8: 'Caturra', 9: 'Wolishalo,Kurume,Dega',
            10: 'Typica', 11: 'Catimor', 12: 'Castillo Paraguaycito', 13: 'nan', 14: 'SL28',
            15: 'SL14', 16: 'Catuai', 17: 'Yellow Bourbon', 18: 'Catrenic', 19: 'unknown',
            20: 'Pacamara', 21: 'Castillo and Colombia blend', 22: 'Jember,TIM-TIM,Ateng',
            23: 'BOURBON, CATURRA Y CATIMOR', 24: 'Bourbon Sidra', 25: 'Sarchimor',
            26: 'Catimor,Catuai,Caturra,Bourbon', 27: 'Parainema', 28: 'SHG', 29: 'Typica + SL34',
            30: 'MARSELLESA, CATUAI, CATURRA & MARSELLESA, ANACAFE 14, CATUAI', 31: 'Mundo Novo',
            32: 'Red Bourbon,Caturra', 33: 'Lempira', 34: 'Typica Gesha', 35: 'Gayo',
            36: 'Bourbon, Catimor, Caturra, Typica', 37: 'unknow', 38: 'Maragogype',
            39: 'Caturra-Catuai', 40: 'SL28,SL34,Ruiru11', 41: 'Yellow Catuai', 42: 'Catucai',
            43: 'Santander', 44: 'Typica Bourbon Caturra Catimor', 45: 'Caturra,Colombia,Castillo',
            46: 'Castillo,Caturra,Bourbon', 47: 'Pacas'
        }

processing_method_encoding={
        0: 'Double Anaerobic Washed', 1: 'Washed / Wet', 2: 'Semi Washed', 3: 'Honey,Mossto', 4: 'Natural / Dry', 5: 'Pulped natural / honey', 6: 'nan', 7: 'Double Carbonic Maceration / Natural', 8: 'Wet Hulling', 9: 'Anaerobico 1000h'}

color_encoding={0: 'green', 1: 'blue-green', 2: 'yellowish', 3: 'yellow-green', 4: 'yellow green', 5: 'greenish', 6: 'brownish', 7: 'yellow- green', 8: 'browish-green', 9: 'bluish-green', 10: 'pale yellow', 11: 'yello-green'}

def preprocess_input_data(input_data):
    # Create a DataFrame from the input data
    df = pd.DataFrame(input_data, index=[0])
    
    # Encode categorical variables
    df['Country of Origin Label'] = df['Country of Origin'].map(country_encodings)
    df['Region Label'] = df['Region'].map(region_options)
    df['Variety Label'] = df['Variety'].map(variety_encodings)
    df['Processing Method Label'] = df['Processing Method'].map(processing_method_encoding)
    df['Color Label'] = df['Color'].map(color_encoding)
    
    # Encode numerical variables
    df['Altitude Label'] = df['Altitude']  # Include Altitude Label
    
    # Set the column names explicitly to match those used during model training
    feature_names = ['Country of Origin Label', 'Altitude Label', 'Region Label', 'Variety Label',
                     'Processing Method Label', 'Color Label', 'Aroma', 'Flavor', 'Aftertaste',
                     'Acidity', 'Body', 'Balance', 'Uniformity', 'Overall', 'Moisture Percentage', 'Quakers']
    df = df.reindex(columns=feature_names)
    
    # Return the preprocessed DataFrame
    return df


def predict_with_model(input_data, model):
    # Preprocess input data
    input_df = preprocess_input_data(input_data)

    # Make predictions
    predictions = model.predict(input_df)

    return predictions

if __name__ == "__main__":
    # Load pre-trained model
    model = joblib.load(r'C:\\Users\\Lenovo\\Desktop\\Visual Studio Codes\\Coffee-Quality_Prediction\\model.pkl')

    # Example input data (replace with actual user input)
    input_data = {
        'Country of Origin': 'Colombia',
        'Altitude': 1200,
        'Region': 'Antioquia',
        'Variety': 'Caturra',
        'Processing Method': 'Washed / Wet',
        'Color': 'Green',
        'Aroma': 7.5,
        'Flavor': 8.2,
        'Aftertaste': 8.0,
        'Acidity': 8.5,
        'Body': 7.8,
        'Balance': 8.0,
        'Uniformity': 9.0,
        'Overall': 8.2,
        'Moisture Percentage': 10.0,
        'Quakers': 0,
        'Color Label': 2  # Example label, adjust according to your encoding
    }

    # Predict
    predictions = predict_with_model(input_data, model)
    print("Predictions:", predictions)