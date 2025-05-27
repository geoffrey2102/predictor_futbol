import os
import logging
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, flash
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import requests
import time

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicializar Flask
app = Flask(__name__)
app.secret_key = 'football_prediction_secret_key_2024'

# Variables globales
model = None
team_encoder = None
liga_encoder = None
scaler = None
team_embeddings = None

# Las cinco grandes ligas europeas
LEAGUES = {
    'Premier League': {'id': '4328', 'country': 'England'},
    'La Liga': {'id': '4335', 'country': 'Spain'},
    'Serie A': {'id': '4332', 'country': 'Italy'},
    'Bundesliga': {'id': '4331', 'country': 'Germany'},
    'Ligue 1': {'id': '4334', 'country': 'France'}
}

# Equipos por liga (datos de ejemplo actualizables)
TEAMS_BY_LEAGUE = {
    'Premier League': [
        'Arsenal', 'Manchester City', 'Liverpool', 'Chelsea', 'Tottenham Hotspur',
        'Manchester United', 'Newcastle United', 'Aston Villa', 'West Ham United',
        'Brighton & Hove Albion', 'Wolverhampton Wanderers', 'Fulham', 'Crystal Palace',
        'Everton', 'Brentford', 'Nottingham Forest', 'AFC Bournemouth', 'Leicester City',
        'Southampton', 'Ipswich Town'
    ],
    'La Liga': [
        'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Athletic Bilbao', 'Real Sociedad',
        'Villarreal', 'Valencia', 'Sevilla', 'Real Betis', 'Girona FC',
        'Celta Vigo', 'Osasuna', 'Getafe', 'Mallorca', 'Las Palmas',
        'Alaves', 'Espanyol', 'Leganes', 'Rayo Vallecano', 'Valladolid'
    ],
    'Serie A': [
        'Inter Milan', 'AC Milan', 'Juventus', 'Napoli', 'AS Roma',
        'Lazio', 'Atalanta', 'Fiorentina', 'Bologna', 'Torino',
        'Udinese', 'Sassuolo', 'Empoli', 'Hellas Verona', 'Cagliari',
        'Genoa', 'Lecce', 'Frosinone', 'Salernitana', 'Monza'
    ],
    'Bundesliga': [
        'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Union Berlin', 'SC Freiburg',
        'Bayer Leverkusen', 'Eintracht Frankfurt', 'Wolfsburg', 'Borussia Monchengladbach', 'Mainz 05',
        'FC Koln', 'Augsburg', 'Werder Bremen', 'VfB Stuttgart', 'Hoffenheim',
        'VfL Bochum', 'Hertha Berlin', 'Schalke 04'
    ],
    'Ligue 1': [
        'Paris Saint-Germain', 'Marseille', 'AS Monaco', 'Lyon', 'Lille',
        'Rennes', 'Nice', 'Lens', 'Nantes', 'Montpellier',
        'Strasbourg', 'Brest', 'Reims', 'Toulouse', 'Le Havre',
        'Clermont Foot', 'Lorient', 'Metz'
    ]
}

def initialize_model_and_encoders():
    """Inicializar modelo y encoders"""
    global model, team_encoder, liga_encoder, scaler, team_embeddings
    
    try:
        # Cargar modelo
        if os.path.exists('football_prediction_model.h5'):
            model = tf.keras.models.load_model('football_prediction_model.h5')
            logging.info("Modelo cargado correctamente")
        else:
            logging.warning("Modelo no encontrado, creando modelo dummy")
            model = create_dummy_model()
            
        # Cargar encoders y otros componentes
        if os.path.exists('model_encoders.pkl'):
            with open('model_encoders.pkl', 'rb') as f:
                data = pickle.load(f)
                team_encoder = data.get('team_encoder')
                liga_encoder = data.get('liga_encoder')
                scaler = data.get('scaler')
                team_embeddings = data.get('team_embeddings')
            logging.info("Encoders cargados correctamente")
        else:
            logging.warning("Encoders no encontrados, creando encoders dummy")
            create_dummy_encoders()
            
    except Exception as e:
        logging.error(f"Error inicializando sistema: {e}")
        model = create_dummy_model()
        create_dummy_encoders()

def create_dummy_model():
    """Crear modelo dummy para demostración"""
    try:
        # Modelo simple para demostración
        input_seq = tf.keras.layers.Input(shape=(5, 12))
        lstm = tf.keras.layers.LSTM(64, return_sequences=True)(input_seq)
        lstm = tf.keras.layers.LSTM(32)(lstm)
        
        local_team = tf.keras.layers.Input(shape=(1,))
        visit_team = tf.keras.layers.Input(shape=(1,))
        
        local_embed = tf.keras.layers.Embedding(100, 8)(local_team)
        visit_embed = tf.keras.layers.Embedding(100, 8)(visit_team)
        local_embed = tf.keras.layers.Flatten()(local_embed)
        visit_embed = tf.keras.layers.Flatten()(visit_embed)
        
        concat = tf.keras.layers.Concatenate()([lstm, local_embed, visit_embed])
        dense = tf.keras.layers.Dense(64, activation='relu')(concat)
        dropout = tf.keras.layers.Dropout(0.3)(dense)
        
        output_result = tf.keras.layers.Dense(3, activation='softmax', name='result')(dropout)
        output_diff = tf.keras.layers.Dense(1, activation='linear', name='difference')(dropout)
        
        dummy_model = tf.keras.Model([input_seq, local_team, visit_team], [output_result, output_diff])
        dummy_model.compile(
            optimizer='adam',
            loss={'result': 'categorical_crossentropy', 'difference': 'mse'}
        )
        
        logging.info("Modelo dummy creado")
        return dummy_model
        
    except Exception as e:
        logging.error(f"Error creando modelo dummy: {e}")
        return None

def create_dummy_encoders():
    """Crear encoders dummy"""
    global team_encoder, liga_encoder, scaler, team_embeddings
    
    # Crear lista de todos los equipos
    all_teams = []
    for teams in TEAMS_BY_LEAGUE.values():
        all_teams.extend(teams)
    
    team_encoder = LabelEncoder()
    team_encoder.fit(all_teams)
    
    liga_encoder = LabelEncoder()
    liga_encoder.fit(list(LEAGUES.keys()))
    
    scaler = StandardScaler()
    # Datos dummy para el scaler
    dummy_data = np.random.randn(100, 11)
    scaler.fit(dummy_data)
    
    # Embeddings dummy
    team_embeddings = np.random.randn(len(all_teams), 8)
    
    logging.info("Encoders dummy creados")

def get_team_stats_api(team_name, league_name):
    """Obtener estadísticas reales del equipo desde API"""
    try:
        # Aquí implementarías la llamada real a la API
        # Por ahora, devolvemos estadísticas simuladas realistas
        base_stats = {
            # Premier League teams
            'Arsenal': {'forma': 4.2, 'posesion': 58.5, 'xg': 2.1, 'def_rating': 7.8},
            'Manchester City': {'forma': 4.5, 'posesion': 68.2, 'xg': 2.8, 'def_rating': 8.5},
            'Liverpool': {'forma': 4.3, 'posesion': 61.8, 'xg': 2.5, 'def_rating': 7.9},
            'Chelsea': {'forma': 3.8, 'posesion': 57.3, 'xg': 2.0, 'def_rating': 7.2},
            
            # La Liga teams
            'Real Madrid': {'forma': 4.4, 'posesion': 62.1, 'xg': 2.6, 'def_rating': 8.2},
            'Barcelona': {'forma': 4.2, 'posesion': 68.9, 'xg': 2.4, 'def_rating': 7.6},
            'Atletico Madrid': {'forma': 3.9, 'posesion': 54.2, 'xg': 1.8, 'def_rating': 8.8},
            
            # Serie A teams
            'Inter Milan': {'forma': 4.1, 'posesion': 58.7, 'xg': 2.2, 'def_rating': 8.3},
            'AC Milan': {'forma': 3.7, 'posesion': 56.4, 'xg': 1.9, 'def_rating': 7.4},
            'Juventus': {'forma': 3.8, 'posesion': 57.8, 'xg': 1.8, 'def_rating': 8.1},
            
            # Bundesliga teams
            'Bayern Munich': {'forma': 4.6, 'posesion': 65.3, 'xg': 2.9, 'def_rating': 8.0},
            'Borussia Dortmund': {'forma': 4.0, 'posesion': 58.9, 'xg': 2.3, 'def_rating': 7.1},
            
            # Ligue 1 teams
            'Paris Saint-Germain': {'forma': 4.4, 'posesion': 64.7, 'xg': 2.7, 'def_rating': 7.8}
        }
        
        if team_name in base_stats:
            return base_stats[team_name]
        else:
            # Estadísticas por defecto basadas en la liga
            league_defaults = {
                'Premier League': {'forma': 3.5, 'posesion': 52.0, 'xg': 1.6, 'def_rating': 7.0},
                'La Liga': {'forma': 3.3, 'posesion': 55.0, 'xg': 1.5, 'def_rating': 7.2},
                'Serie A': {'forma': 3.2, 'posesion': 53.0, 'xg': 1.4, 'def_rating': 7.5},
                'Bundesliga': {'forma': 3.4, 'posesion': 54.0, 'xg': 1.7, 'def_rating': 6.8},
                'Ligue 1': {'forma': 3.1, 'posesion': 51.0, 'xg': 1.3, 'def_rating': 7.1}
            }
            return league_defaults.get(league_name, {'forma': 3.0, 'posesion': 50.0, 'xg': 1.2, 'def_rating': 7.0})
            
    except Exception as e:
        logging.error(f"Error obteniendo estadísticas para {team_name}: {e}")
        return {'forma': 3.0, 'posesion': 50.0, 'xg': 1.2, 'def_rating': 7.0}

def safe_team_encode(team_name):
    """Codificar equipo de forma segura"""
    try:
        if team_encoder and team_name in team_encoder.classes_:
            return team_encoder.transform([team_name])[0]
        else:
            # Buscar equipo similar o usar hash
            return hash(team_name) % 100
    except Exception as e:
        logging.error(f"Error codificando equipo {team_name}: {e}")
        return 0

def make_prediction(local_team, visit_team, league_name):
    """Hacer predicción usando el modelo"""
    try:
        if model is None:
            raise ValueError("Modelo no disponible")
        
        # Obtener estadísticas de equipos
        local_stats = get_team_stats_api(local_team, league_name)
        visit_stats = get_team_stats_api(visit_team, league_name)
        
        # Preparar datos de entrada
        local_idx = safe_team_encode(local_team)
        visit_idx = safe_team_encode(visit_team)
        
        # Crear secuencia dummy (en producción, usarías datos históricos reales)
        sequence = np.zeros((1, 5, 12))
        for i in range(5):
            sequence[0, i, :] = [
                local_idx, visit_idx,
                local_stats['forma'] + np.random.normal(0, 0.2),
                visit_stats['forma'] + np.random.normal(0, 0.2),
                local_stats['forma'], visit_stats['forma'],
                local_stats['posesion'], visit_stats['posesion'],
                local_stats['xg'], visit_stats['xg'],
                local_stats['def_rating'], visit_stats['def_rating']
            ]
        
        # Normalizar si tenemos scaler
        if scaler:
            sequence_flat = sequence.reshape(-1, 12)
            try:
                sequence_normalized = scaler.transform(sequence_flat[:, 2:])  # Skip team indices
                sequence[0, :, 2:] = sequence_normalized.reshape(5, 10)
            except:
                pass  # Usar datos sin normalizar si falla
        
        local_team_input = np.array([[local_idx]])
        visit_team_input = np.array([[visit_idx]])
        
        # Hacer predicción
        predictions = model.predict([sequence, local_team_input, visit_team_input], verbose=0)
        
        if isinstance(predictions, list) and len(predictions) >= 2:
            result_pred = predictions[0][0]
            diff_pred = predictions[1][0][0] if len(predictions[1].shape) > 1 else predictions[1][0]
        else:
            result_pred = predictions[0] if hasattr(predictions, '__len__') else [0.4, 0.3, 0.3]
            diff_pred = 0.0
        
        # Asegurar que tenemos 3 probabilidades
        if len(result_pred) >= 3:
            victoria_local = float(result_pred[0]) * 100
            empate = float(result_pred[1]) * 100
            victoria_visitante = float(result_pred[2]) * 100
        else:
            # Valores por defecto basados en estadísticas
            home_advantage = 5.0
            strength_diff = (local_stats['forma'] - visit_stats['forma']) * 10
            
            victoria_local = max(20, min(70, 40 + home_advantage + strength_diff))
            empate = max(15, min(40, 30 - abs(strength_diff) * 0.5))
            victoria_visitante = 100 - victoria_local - empate
        
        # Normalizar probabilidades
        total = victoria_local + empate + victoria_visitante
        if total > 0:
            victoria_local = (victoria_local / total) * 100
            empate = (empate / total) * 100
            victoria_visitante = (victoria_visitante / total) * 100
        
        return victoria_local, empate, victoria_visitante, float(diff_pred)
        
    except Exception as e:
        logging.error(f"Error en predicción: {e}")
        # Predicción por defecto basada en estadísticas
        local_stats = get_team_stats_api(local_team, league_name)
        visit_stats = get_team_stats_api(visit_team, league_name)
        
        strength_diff = local_stats['forma'] - visit_stats['forma']
        victoria_local = max(25, min(65, 40 + strength_diff * 8))
        empate = max(20, min(35, 30))
        victoria_visitante = 100 - victoria_local - empate
        
        return victoria_local, empate, victoria_visitante, strength_diff * 0.5

def get_top_players_by_team(team_name, league_name):
    """Obtener jugadores destacados por equipo"""
    players_db = {
    # Premier League (20 teams)
    'Arsenal': ['Bukayo Saka', 'Gabriel Martinelli', 'Leandro Trossard', 'Kai Havertz'],
    'Manchester City': ['Erling Haaland', 'Phil Foden', 'Josko Gvardiol', 'Rodri'],
    'Liverpool': ['Mohamed Salah', 'Darwin Núñez', 'Luis Díaz', 'Diogo Jota'],
    'Chelsea': ['Cole Palmer', 'Nicolas Jackson', 'Noni Madueke', 'Christopher Nkunku'],
    'Tottenham Hotspur': ['Son Heung-min', 'Dominic Solanke', 'Brennan Johnson', 'James Maddison'],
    'Manchester United': ['Marcus Rashford', 'Rasmus Højlund', 'Alejandro Garnacho', 'Bruno Fernandes'],
    'Aston Villa': ['Ollie Watkins', 'Jhon Durán', 'Morgan Rogers', 'Leon Bailey'],
    'Newcastle United': ['Alexander Isak', 'Anthony Gordon', 'Valentino Livramento', 'Bruno Guimarães'],
    'Brighton & Hove Albion': ['Danny Welbeck', 'João Pedro', 'Kaoru Mitoma', 'Georginio Rutter'],
    'West Ham United': ['Jarrod Bowen', 'Tomas Soucek', 'Mohammed Kudus', 'Lucas Paquetá'],
    'Everton': ['Dwight McNeil', 'Dominic Calvert-Lewin', 'Iliman Ndiaye', 'Abdoulaye Doucouré'],
    'Nottingham Forest': ['Chris Wood', 'Morgan Gibbs-White', 'Callum Hudson-Odoi', 'Anthony Elanga'],
    'Fulham': ['Raúl Jiménez', 'Adama Traoré', 'Emile Smith Rowe', 'Sasa Lukić'],
    'Brentford': ['Bryan Mbeumo', 'Yoane Wissa', 'Nathan Collins', 'Kevin Schade'],
    'Bournemouth': ['Antoine Semenyo', 'Justin Kluivert', 'Evanilson', 'Dango Ouattara'],
    'Crystal Palace': ['Jean-Philippe Mateta', 'Eberechi Eze', 'Ismaïla Sarr', 'Daniel Muñoz'],
    'Wolverhampton Wanderers': ['Matheus Cunha', 'Jørgen Strand Larsen', 'Rayan Aït-Nouri', 'Pablo Sarabia'],
    'Leicester City': ['Jamie Vardy', 'Abdul Fatawu', 'Harry Winks', 'Wilfred Ndidi'],
    'Ipswich Town': ['Liam Delap', 'Sammie Szmodics', 'Omari Hutchinson', 'Leif Davis'],
    'Southampton': ['Cameron Archer', 'Tyler Dibling', 'Joe Aribo', 'Adam Armstrong'],

    # La Liga (20 teams)
    'Real Madrid': ['Kylian Mbappé', 'Vinicius Jr.', 'Rodrygo', 'Jude Bellingham'],
    'Barcelona': ['Robert Lewandowski', 'Raphinha', 'Lamine Yamal', 'Dani Olmo'],
    'Atletico Madrid': ['Antoine Griezmann', 'Álvaro Morata', 'Alexander Sørloth', 'Julian Alvarez'],
    'Athletic Bilbao': ['Iñaki Williams', 'Gorka Guruzeta', 'Oihan Sancet', 'Álex Berenguer'],
    'Villarreal': ['Ayoze Pérez', 'Thierno Barry', 'Gerard Moreno', 'Dani Parejo'],
    'Real Betis': ['Johnny Cardoso', 'Giovani Lo Celso', 'Chimy Ávila', 'Pablo Fornals'],
    'Girona': ['Cristhian Stuani', 'Yaser Asprilla', 'Bryan Gil', 'Miguel Gutiérrez'],
    'Osasuna': ['Ante Budimir', 'Rubén García', 'Aimar Oroz', 'Lucas Torró'],
    'Celta Vigo': ['Iago Aspas', 'Borja Iglesias', 'Óscar Mingueza', 'Anastasios Douvikas'],
    'Rayo Vallecano': ['Álvaro García', 'Randy Nteka', 'Isi Palazón', 'Jorge de Frutos'],
    'Real Sociedad': ['Mikel Oyarzabal', 'Brais Méndez', 'Takefusa Kubo', 'Ander Barrenetxea'],
    'Mallorca': ['Dani Rodríguez', 'Vedat Muriqi', 'Cyle Larin', 'Antonio Sánchez'],
    'Sevilla': ['Dodi Lukébakio', 'Jesús Navas', 'Isaac Romero', 'Chidera Ejuke'],
    'Valencia': ['Hugo Duro', 'Dani Gómez', 'Rafa Mir', 'Pepelu'],
    'Getafe': ['Álvaro Rodríguez', 'Mauro Arambarri', 'Borja Mayoral', 'Chrisantus Uche'],
    'Alavés': ['Kike García', 'Toni Martínez', 'Carlos Vicente', 'Santiago Mourino'],
    'Espanyol': ['Javi Puado', 'Alejo Véliz', 'Marash Kumbulla', 'Carlos Romero'],
    'Leganés': ['Juan Cruz', 'Miguel de la Fuente', 'Seydouba Cissé', 'Dani Raba'],
    'Valladolid': ['Raúl Moro', 'Mamadou Sylla', 'Iván Sánchez', 'Selim Amallah'],
    'Las Palmas': ['Sandro Ramírez', 'Alberto Moleiro', 'Oli McBurnie', 'Fábio Silva'],

    # Serie A (20 teams)
    'Inter Milan': ['Lautaro Martínez', 'Marcus Thuram', 'Hakan Çalhanoğlu', 'Nicolò Barella'],
    'AC Milan': ['Christian Pulisic', 'Rafael Leão', 'Tijjani Reijnders', 'Álvaro Morata'],
    'Juventus': ['Dušan Vlahović', 'Kenan Yıldız', 'Weston McKennie', 'Andrea Cambiaso'],
    'Napoli': ['Romelu Lukaku', 'Khvicha Kvaratskhelia', 'Giovanni Simeone', 'David Neres'],
    'Atalanta': ['Ademola Lookman', 'Charles De Ketelaere', 'Mateo Retegui', 'Lazar Samardžić'],
    'Lazio': ['Valentín Castellanos', 'Mattia Zaccagni', 'Boulaye Dia', 'Matteo Guendouzi'],
    'Fiorentina': ['Moise Kean', 'Albert Guðmundsson', 'Danilo Cataldi', 'Edoardo Bove'],
    'AS Roma': ['Artem Dovbyk', 'Lorenzo Pellegrini', 'Tommaso Baldanzi', 'Daniele de Rossi'],
    'Bologna': ['Riccardo Orsolini', 'Santiago Castro', 'Jens Odgaard', 'Giovanni Fabbian'],
    'Torino': ['Duván Zapata', 'Ché Adams', 'Antonio Sanabria', 'Valentino Lazaro'],
    'Udinese': ['Florian Thauvin', 'Lorenzo Lucca', 'Keinan Davis', 'Isaac Success'],
    'Empoli': ['Pietro Pellegri', 'Sebastiano Esposito', 'Lorenzo Colombo', 'Emmanuel Gyasi'],
    'Genoa': ['Andrea Pinamonti', 'Albert Guðmundsson', 'Mario Balotelli', 'Alessandro Vogliacco'],
    'Parma': ['Dennis Man', 'Valentin Mihăilă', 'Matteo Cancellieri', 'Ange-Yoan Bonny'],
    'Cagliari': ['Zito Luvumbo', 'Roberto Piccoli', 'Gianluca Gaetano', 'Nicola Nandez'],
    'Hellas Verona': ['Casper Tengstedt', 'Daniel Mosquera', 'Darko Lazović', 'Tomas Suslov'],
    'Monza': ['Daniel Maldini', 'Dany Mota', 'Gianluca Caprari', 'Alessandro Bianco'],
    'Lecce': ['Nikola Krstović', 'Ante Rebić', 'Lameck Banda', 'Patrick Dorgu'],
    'Venezia': ['Joel Pohjanpalo', 'Gaetano Oristanio', 'Hans Nicolussi Caviglia', 'Jay Idzes'],
    'Como': ['Patrick Cutrone', 'Andrea Belotti', 'Nico Paz', 'Alessandro Gabrielloni'],

    # Bundesliga (18 teams)
    'Bayern Munich': ['Harry Kane', 'Jamal Musiala', 'Leroy Sané', 'Thomas Müller'],
    'Borussia Dortmund': ['Serhou Guirassy', 'Donyell Malen', 'Julian Brandt', 'Karim Adeyemi'],
    'Bayer Leverkusen': ['Florian Wirtz', 'Patrik Schick', 'Victor Boniface', 'Jonathan Tah'],
    'RB Leipzig': ['Loïs Openda', 'Benjamin Šeško', 'Antonio Nusa', 'Xavi Simons'],
    'Eintracht Frankfurt': ['Omar Marmoush', 'Hugo Ekitiké', 'Fares Chaibi', 'Ansgar Knauff'],
    'VfB Stuttgart': ['Deniz Undav', 'Ermedin Demirović', 'Chris Führich', 'Josha Vagnoman'],
    'Wolfsburg': ['Jonas Wind', 'Lukas Nmecha', 'Lovro Majer', 'Mohammed Amoura'],
    'Freiburg': ['Vincenzo Grifo', 'Michael Gregoritsch', 'Ritsu Dōan', 'Junior Adamu'],
    'Werder Bremen': ['Marvin Ducksch', 'Marco Grüll', 'Jens Stage', 'Romano Schmid'],
    'Borussia Mönchengladbach': ['Tim Kleindienst', 'Alassane Pléa', 'Robin Hack', 'Franck Honorat'],
    'Augsburg': ['Phillip Tietz', 'Samuel Essende', 'Arne Engels', 'Keven Schlotterbeck'],
    'Mainz 05': ['Jonathan Burkardt', 'Paul Nebel', 'Moritz Jenz', 'Maxim Leitsch'],
    'Union Berlin': ['Benedict Hollerbach', 'Jordan Pefok', 'Tom Rothe', 'Danilho Doekhi'],
    'Hoffenheim': ['Andrej Kramarić', 'Mërgim Berisha', 'Marius Bülter', 'Valentin Gendrey'],
    'Heidenheim': ['Paul Wanner', 'Marvin Pieringer', 'Niklas Dorsch', 'Lennard Maloney'],
    'St. Pauli': ['Guido Burgstaller', 'Elias Saad', 'Morgan Guilavogui', 'Dapo Afolayan'],
    'Bochum': ['Matúš Bero', 'Dani de Wit', 'Moritz Broschinski', 'Philipp Hofmann'],
    'Holstein Kiel': ['Shuto Machino', 'Fiete Arp', 'Benedikt Pichler', 'Armin Gigović'],

    # Ligue 1 (18 teams)
    'Paris Saint-Germain': ['Bradley Barcola', 'Ousmane Dembélé', 'Randal Kolo Muani', 'Gonçalo Ramos'],
    'Olympique de Marseille': ['Mason Greenwood', 'Wesley Teixeira', 'Valentin Carboni', 'Pierre-Emile Højbjerg'],
    'Monaco': ['Folarin Balogun', 'Lamine Camara', 'Breel Embolo', 'Aleksandr Golovin'],
    'Lille': ['Jonathan David', 'Edon Zhegrova', 'Thomas Meunier', 'Osame Sahraoui'],
    'Lens': ['Adrien Thomasson', 'Przemysław Frankowski', 'Wesley Saïd', 'Florian Sotoca'],
    'Nice': ['Evann Guessand', 'Tanguy Ndombele', 'Sofiane Diop', 'Youssoufa Moukoko'],
    'Lyon': ['Malick Fofana', 'Alexandre Lacazette', 'Saïd Benrahma', 'Gift Orban'],
    'Rennes': ['Arnaud Kalimuendo', 'Amine Gouiri', 'Ludovic Blas', 'Andrey Santos'],
    'Strasbourg': ['Emanuel Emegha', 'Diego Moreira', 'Andrey Santos', 'Sekkou Dilane'],
    'Toulouse': ['Shavy Babicka', 'Zakaria Aboukhlal', 'Yann Gboho', 'César Gelabert'],
    'Nantes': ['Moses Simon', 'Mostafa Mohamed', 'Bahereba Guirassy', 'Sorba Thomas'],
    'Saint-Étienne': ['Zuriko Davitashvili', 'Mathieu Cafaro', 'Ibrahim Sissoko', 'Lucas Stassin'],
    'Reims': ['Keito Nakamura', 'Cédric Kipré', 'Amine Salama', 'Valentin Atangana'],
    'Montpellier': ['Akor Adams', 'Téji Savanier', 'Arnaud Nordin', 'Modibo Sagnan'],
    'Brest': ['Mama Baldé', 'Romain Del Castillo', 'Mahdi Camara', 'Ludovic Ajorque'],
    'Le Havre': ['Yassine Kechta', 'Daler Kuziaev', 'Gautier Lloris', 'Arouna Sangante'],
    'Angers': ['Himad Abdelli', 'Jean-Eudes Aholou', 'Zinedine Ferhat', 'Ibrahima Niane'],
    'Auxerre': ['Lassine Sinayoko', 'Jubal', 'Elisha Owusu', 'Hamed Traoré']
}
    return players_db.get(team_name, ['Jugador 1', 'Jugador 2'])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    selected_league = 'Premier League'
    
    if request.method == 'POST':
        try:
            selected_league = request.form.get('league', 'Premier League')
            local_team = request.form.get('local_team')
            visit_team = request.form.get('visit_team')
            
            if not local_team or not visit_team:
                flash('Por favor selecciona ambos equipos', 'error')
                return render_template('index.html', 
                                     leagues=LEAGUES, 
                                     teams_by_league=TEAMS_BY_LEAGUE,
                                     selected_league=selected_league)
            
            if local_team == visit_team:
                flash('Los equipos deben ser diferentes', 'error')
                return render_template('index.html', 
                                     leagues=LEAGUES, 
                                     teams_by_league=TEAMS_BY_LEAGUE,
                                     selected_league=selected_league)
            
            # Hacer predicción
            victoria_local, empate, victoria_visitante, diff_pred = make_prediction(
                local_team, visit_team, selected_league
            )
            
            # Obtener estadísticas adicionales
            local_stats = get_team_stats_api(local_team, selected_league)
            visit_stats = get_team_stats_api(visit_team, selected_league)
            
            prediction = {
                'league': selected_league,
                'local_team': local_team,
                'visit_team': visit_team,
                'victoria_local': round(victoria_local, 1),
                'empate': round(empate, 1),
                'victoria_visitante': round(victoria_visitante, 1),
                'diferencia_goles': round(diff_pred, 1),
                'goleadores_local': get_top_players_by_team(local_team, selected_league),
                'goleadores_visitante': get_top_players_by_team(visit_team, selected_league),
                'forma_local': local_stats['forma'],
                'forma_visitante': visit_stats['forma'],
                'posesion_local': local_stats['posesion'],
                'posesion_visitante': round(100 - local_stats['posesion'], 1),
                'xg_local': local_stats['xg'],
                'xg_visitante': visit_stats['xg'],
                'def_rating_local': local_stats['def_rating'],
                'def_rating_visitante': visit_stats['def_rating']
            }
            
            logging.info(f"Predicción generada: {local_team} vs {visit_team} en {selected_league}")
            
        except Exception as e:
            logging.error(f"Error procesando predicción: {e}")
            flash(f'Error al generar predicción: {str(e)}', 'error')
    
    return render_template('index.html', 
                         prediction=prediction,
                         leagues=LEAGUES,
                         teams_by_league=TEAMS_BY_LEAGUE,
                         selected_league=selected_league)

@app.route('/api/teams/<league>')
def get_teams_by_league(league):
    """API endpoint para obtener equipos por liga"""
    teams = TEAMS_BY_LEAGUE.get(league, [])
    return jsonify({'teams': teams})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint para predicciones"""
    try:
        data = request.get_json()
        league = data.get('league', 'Premier League')
        local_team = data.get('local_team')
        visit_team = data.get('visit_team')
        
        if not local_team or not visit_team:
            return jsonify({'error': 'Equipos requeridos'}), 400
            
        if local_team == visit_team:
            return jsonify({'error': 'Los equipos deben ser diferentes'}), 400
        
        victoria_local, empate, victoria_visitante, diff_pred = make_prediction(
            local_team, visit_team, league
        )
        
        return jsonify({
            'victoria_local': round(victoria_local, 1),
            'empate': round(empate, 1),
            'victoria_visitante': round(victoria_visitante, 1),
            'diferencia_goles': round(diff_pred, 1)
        })
        
    except Exception as e:
        logging.error(f"Error en API predict: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Endpoint de salud"""
    status = {
        'status': 'OK',
        'model_loaded': model is not None,
        'encoders_loaded': team_encoder is not None,
        'leagues_available': len(LEAGUES),
        'total_teams': sum(len(teams) for teams in TEAMS_BY_LEAGUE.values())
    }
    return jsonify(status)

@app.route('/retrain')
def retrain_model():
    """Endpoint para reentrenar el modelo (solo para desarrollo)"""
    try:
        # Aquí puedes agregar lógica para reentrenar el modelo
        # Por seguridad, esto debería estar protegido en producción
        flash('Funcionalidad de reentrenamiento no implementada en esta versión', 'info')
        return jsonify({'status': 'not_implemented'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Inicializar sistema al cargar la aplicación
initialize_model_and_encoders()

if __name__ == '__main__':
    logging.info("Iniciando aplicación Flask de predicción de fútbol...")
    logging.info(f"Ligas disponibles: {list(LEAGUES.keys())}")
    logging.info(f"Total de equipos: {sum(len(teams) for teams in TEAMS_BY_LEAGUE.values())}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)