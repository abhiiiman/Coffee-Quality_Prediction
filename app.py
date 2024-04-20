from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        region = request.form.get('region')
        # Process the form data as needed
        return f"Selected Region: {region}"
    else:
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

        processing_method_encoding={0: 'Double Anaerobic Washed', 1: 'Washed / Wet', 2: 'Semi Washed', 3: 'Honey,Mossto', 4: 'Natural / Dry', 5: 'Pulped natural / honey', 6: 'nan', 7: 'Double Carbonic Maceration / Natural', 8: 'Wet Hulling', 9: 'Anaerobico 1000h'}
        color_encoding={0: 'green', 1: 'blue-green', 2: 'yellowish', 3: 'yellow-green', 4: 'yellow green', 5: 'greenish', 6: 'brownish', 7: 'yellow- green', 8: 'browish-green', 9: 'bluish-green', 10: 'pale yellow', 11: 'yello-green'}

        return render_template('submit.html', country_encodings=country_encodings, region_options=region_options, variety_encodings=variety_encodings, processing_method_encoding=processing_method_encoding, color_encoding=color_encoding)
    

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':

        # Load encodings here, ensure they are consistent
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

        region_options = {0: 'Piendamo,Cauca', 1: 'Chiayi', 2: 'Laos Borofen Plateau', 3: 'Los Santos,Tarrazu', 4: 'Popayan,Cauca', 5: 'Chimaltenango', 6: 'KILIMANJARO', 7: 'Guji', 8: 'Acatenango', 9: 'Yunlin', 10: 'tolima', 11: 'Gedeb,Yirgacheffe,Sidamo', 12: 'Shibi, Gukeng Township, Yunlin County 郵遞區號 , Taiwan (R.O.C.)', 13: 'Gukeng Township, Yunlin County', 14: 'Arusha', 15: 'Guatemala, Fraijanes, Santa Rosa', 16: '卓溪鄉Zhuoxi Township', 17: 'Chiang Mai', 18: 'Quindio', 19: 'Região Vulcânica', 20: 'Kona', 21: '壽豐鄉Shoufeng Township', 22: 'Dongshan Dist., Tainan City', 23: 'Oromia', 24: 'Southern Ethiopia Guji', 25: 'OROMIA', 26: 'Central', 27: 'Caoling , Gukeng Township, Yunlin County', 28: '秀林鄉Show Linxia Township', 29: '台灣屏東', 30: '苗栗縣', 31: 'Rwenzori', 32: 'Antigua', 33: 'Santa Rosa', 34: 'quiche', 35: '新竹縣', 36: 'Aceh Tengah', 37: 'Villa Rica', 38: 'Mbeya', 39: 'Nantou', 40: 'Campo das Vertentes', 41: 'Boquete', 42: 'Huehuetenango', 43: 'ANTIGUA GUATEMALA', 44: 'Tarrazu', 45: '( Dongshan Dist., Tainan City)', 46: 'ESTELI', 47: 'Quang Tri', 48: 'Centro, Lagunetillas-Ajuterique, Comayagua', 49: 'Ethiopia', 50: '玉里鎮Yuli Township', 51: "Ka'u district of Big Island", 52: 'Addis Ababa', 53: 'Chalatenango', 54: 'Sierra Nevada de Santa Marta', 55: 'Lintong Nihuta/Dolok Sanggul,Sumatera Utara', 56: 'Atitlán', 57: 'Itasy', 58: '新北市', 59: 'Oriente Santa rosa', 60: 'Sierra de las minas', 61: 'San Andrés, Lempira', 62: 'ARUSHA', 63: 'Sidama', 64: 'Huila', 65: 'New Oriente', 66: 'Nan', 67: 'Pereira', 68: 'HUEHUETENANGO', 69: 'Cauca', 70: "Ka'u", 71: 'Tolima', 72: 'Chiapas', 73: 'not known', 74: 'South Shan State', 75: 'San Jose, La Paz', 76: 'QUICHE', 77: 'Marcala', 78: 'Popayán Cauca', 79: 'North of Thailand', 80: 'Suan Ya Lung', 81: 'Mt Elgon', 82: 'NEW ORIENTE & HUEHUETENANGO', 83: '桃園市', 84: 'Sul de Minas', 85: 'Marcala, La Paz', 86: 'Nongluang Bolaven Plateau, Champasack, Lao PDR', 87: 'Centro, Lagunetillas - Ajuterique, Comayagua', 88: 'Los Planes de Santa Maria, La Paz', 89: 'HUILA', 90: 'ZONGOLICA, VERACRUZ', 91: 'Eatan Commune, Krong Nang District, Krong Nang Province', 92: 'Matagalpa', 93: 'west Villege', 94: 'los planes de santa maria la paz', 95: 'Apaneca - Ilamatepec', 96: 'MANTIQUEIRA / SUL DE MINAS', 97: 'Veracruz', 98: 'Matagalpa, Nicaragua', 99: 'Dalat', 100: 'Chanchamayo, La Merced', 101: 'Kona district of Big Island', 102: 'Kericho', 103: 'Corralillo Tarrazu', 104: 'Oriente', 105: 'Sumatra', 106: 'Lam Dong Province', 107: 'Mantiquira de minas', 108: 'Santander', 109: 'Coatepec, Veracruz', 110: 'occidente', 111: '臺北市', 112: 'Chiang Rai', 113: 'Alta Mogiana-Ibiraci', 114: 'Volcan Chinchontepek, San Vicente, El Salvador', 115: 'Jinotega', 116: 'Caicedonia,Valle del Cauca', 117: 'Chanchamayo, Lamerced', 118: 'Bolaven Plateau', 119: 'Volcan de San Vicente, La Paz, El Salvador'}
        
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
        
        processing_method_encoding = {0: 'Double Anaerobic Washed', 1: 'Washed / Wet', 2: 'Semi Washed', 3: 'Honey,Mossto', 4: 'Natural / Dry', 5: 'Pulped natural / honey', 6: 'nan', 7: 'Double Carbonic Maceration / Natural', 8: 'Wet Hulling', 9: 'Anaerobico 1000h'}
        color_encoding = {0: 'green', 1: 'blue-green', 2: 'yellowish', 3: 'yellow-green', 4: 'yellow green', 5: 'greenish', 6: 'brownish', 7: 'yellow- green', 8: 'browish-green', 9: 'bluish-green', 10: 'pale yellow', 11: 'yello-green'}

        # Store the values of the features in a dictionary
        form_data = {
            'country': request.form['country'],
            'altitude': float(request.form['altitude']),
            'region': int(request.form['region']),  # Convert to integer
            'variety': request.form['variety'],
            'processingMethod': request.form['processingMethod'],
            'aroma': float(request.form['aroma']),
            'flavor': float(request.form['flavor']),
            'aftertaste': float(request.form['aftertaste']),
            'acidity': float(request.form['acidity']),
            'body': float(request.form['body']),
            'balance': float(request.form['balance']),
            'uniformity': float(request.form['uniformity']),
            'overall': float(request.form['overall']),
            'moisturePercentage': float(request.form['moisturePercentage']),
            'quakers': int(request.form['quakers']),
            'color': request.form['color']
        }
        print(form_data)

        # Map categorical values to numerical encodings
        country_code = country_encodings.get(form_data['country'], 0)  # Default to 0 if not found
        region_code = form_data['region']  # No need to index region_options here
        variety_code = variety_encodings.get(form_data['variety'], 0)  # Default to 0 if not found
        processing_method_code = processing_method_encoding.get(form_data['processingMethod'], 0)  # Default to 0 if not found
        color_code = color_encoding.get(form_data['color'], 0)  # Default to 0 if not found
        
        # Create a 2D array with the input features
        input_data = np.array([[country_code,
                                form_data['altitude'],
                                region_code,
                                variety_code,
                                processing_method_code,
                                form_data['aroma'],
                                form_data['flavor'],
                                form_data['aftertaste'],
                                form_data['acidity'],
                                form_data['body'],
                                form_data['balance'],
                                form_data['uniformity'],
                                form_data['overall'],
                                form_data['moisturePercentage'],
                                form_data['quakers'],
                                color_code]])

        # Load the saved model
        model = pickle.load(open('Coffee_Quality_Model.pkl', 'rb'))
        
        # Perform prediction
        prediction = model.predict(input_data)[0]
        
        return render_template('result.html', prediction=prediction)
        
    else:
        
        return render_template('result.html', 
                               country_encodings=country_encodings,
                               region_options=region_options,
                               variety_encodings=variety_encodings,
                               processing_method_encoding=processing_method_encoding,
                               color_encoding=color_encoding)
    

if __name__ == "__main__":
    app.run(debug=True)