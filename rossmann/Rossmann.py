import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann(object):
    def __init__(self):       
        self.home_path = ''
        self.competition_distance_scaler   = pickle.load(open(self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(open(self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb'))
        self.promo2_time_week_scaler       = pickle.load(open(self.home_path + 'parameter/promo2_time_week_scaler.pkl', 'rb'))
        self.year_scaler                   = pickle.load(open(self.home_path + 'parameter/year_scaler.pkl', 'rb'))
        self.store_type_scaler             = pickle.load(open(self.home_path + 'parameter/store_type_scaler.pkl', 'rb'))
    
    def data_cleaning(self, df1):

        ## 1.1. Rename Columns

        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 
                    'SchoolHoliday', 'StoreType', 'Assortment','CompetitionDistance', 'CompetitionOpenSinceMonth', 
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

        # snake_case
        snakecase = lambda x: inflection.underscore(x)

        # creates new columns from old columns in snakecase 
        cols_new = list(map(snakecase, cols_old))

        # renames the old columns
        df1.columns = cols_new

        ## 1.5. Fillout NA's

        # Alterar o tipo 'date' para datetime para usar na função lambda abaixo
        df1['date'] = pd.to_datetime( df1['date'] )

        # 'competition_distance': distância em metros até a loja concorrente mais próxima
        df1['competition_distance'] = df1['competition_distance'].apply( lambda x: 100000.0 if math.isnan(x) else x )

        # 'competition_open_since_month': o mês aproximado da hora em que o concorrente mais próximo foi aberto
        df1['competition_open_since_month'] = df1.apply( lambda x: x['date'].month 
                                                         if math.isnan(x['competition_open_since_month']) 
                                                         else x['competition_open_since_month'], axis=1 )

        # 'competition_open_since_year': o ano aproximado da hora em que o concorrente mais próximo foi aberto
        df1['competition_open_since_year'] = df1.apply( lambda x: x['date'].year 
                                                        if math.isnan(x['competition_open_since_year']) 
                                                        else x['competition_open_since_year'], axis=1 )

        # 'promo2_since_week': descreve a semana do calendário em que a loja começou a participar do Promo2
        df1['promo2_since_week'] = df1.apply ( lambda x: x['date'].week
                                               if math.isnan(x['promo2_since_week'])
                                               else x['promo2_since_week'], axis=1 ) 

        # 'promo2_since_year': descreve o ano em que a loja começou a participar do Promo2
        df1['promo2_since_year'] = df1.apply( lambda x: x['date'].year
                                              if math.isnan(x['promo2_since_year'])
                                              else x['promo2_since_year'], axis=1)

        # 'promo_interval': descreve os intervalos consecutivos em que o Promo2 é iniciado, nomeando os meses em que a promoção é iniciada novamente.
        df1['promo_interval'].fillna( 0, inplace=True )

        ## 1.6. Change Data Types

        # Alterar os tipos de float para int
        # competition
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)

        # promo2
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)
        
        return df1
    
    def feature_engineering(self, df2):
        
        ### 2.4.1. Converter

        # Converter os códigos de 'state_holiday' em nome (a: public_holiday, b: easter_holiday, c: christmas, 0: regular_day)
        state_holiday_rename = {'a': 'public_holiday', 'b': 'easter_holiday', 'c': 'christmas', '0': 'regular_day'}
        df2['state_holiday'] = df2['state_holiday'].map(state_holiday_rename)

        # Converter os códigos de 'assortment' em nível de variedades de produtos (a: basic, b: extra, c: extended)
        assortment_rename = {'a': 'basic', 'b': 'extra', 'c': 'extended'}
        df2['assortment'] = df2['assortment'].map(assortment_rename)

        ### 2.4.2. Criar

        # 'year': ano de venda
        df2['year'] = df2['date'].dt.year

        # 'month': mês de venda
        df2['month'] = df2['date'].dt.month

        # 'day': day de venda
        df2['day'] = df2['date'].dt.day

        # 'week_of_year': semana de venda do ano
        df2['week_of_year'] = df2['date'].dt.weekofyear

        # 'year_week': semana de venda do ano, no formato string 'YYYY-WW'
        df2['year_week'] = df2['date'].dt.strftime( '%Y-%W' )

        # 'competition_time_month': Quantidade de meses desde que a concorrência abriu até a data de venda
        df2['competition_since'] = df2.apply( lambda x: datetime.datetime( year=x['competition_open_since_year'], 
                                                                           month=x['competition_open_since_month'],
                                                                           day=1 ), axis=1 )
        df2['competition_time_month'] = ( (df2['date'] - df2['competition_since'])/30 ).apply( lambda x: x.days ).astype( int )

        # promo2_since: Data de início da promo2
        df2['promo2_since'] = df2['promo2_since_year'].astype( str ) + '-' + df2['promo2_since_week'].astype( str )
        df2['promo2_since'] = df2['promo2_since'].apply( lambda x: datetime.datetime.strptime( x + '-1', '%Y-%W-%w' ) - datetime.timedelta( days=7 ) )

        # promo2_time_week: Quantidade de semanas desde o início da promo2 até a data de venda
        df2['promo2_time_week'] = ( ( df2['date'] - df2['promo2_since'] )/7 ).apply( lambda x: x.days ).astype( int )

        # 3. PASSO 03 - FILTRAGEM DE VARIÁVEIS

        ## 3.1. Filtragem das Linhas

        # Filtrar apenas lojas abertas e com vendas
        df2 = df2.loc[df2['open'] != 0, :].reset_index(drop=True)

        ## 3.2. Seleção das Colunas

        # Remover colunas que não serão utilizadas
        cols_drop = ['open', 'promo_interval']
        df2 = df2.drop( cols_drop, axis=1 )

        return df2

    def data_preparation(self, df5):
        
        ## 5.2. Rescaling
        # 'competition_distance'
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform(df5[['competition_distance']].values)

        # 'competition_time_month'
        df5['competition_time_month'] = self.competition_time_month_scaler.fit_transform(df5[['competition_distance']].values)

        # 'promo2_time_week'
        df5['promo2_time_week'] = self.promo2_time_week_scaler.fit_transform(df5[['year']].values)

        # 'year'
        df5['year'] = self.year_scaler.fit_transform(df5[['year']].values)

        ## 5.3. Transformação

        ### 5.3.1. Encoding

        # store_type - Label Encoding
        df5['store_type'] = self.store_type_scaler.fit_transform(df5['store_type'])

        # assortment - Ordinal Encoding
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)

        # state_holiday - One Hot Encoding
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])

        ### 5.3.3. Transformações de natureza

        # day_of_week
        df5['day_of_week_sin'] = df5['day_of_week'].apply( lambda x: np.sin( x * (2 * np.pi/7) ) )
        df5['day_of_week_cos'] = df5['day_of_week'].apply( lambda x: np.cos( x * (2 * np.pi/7) ) )

        # month
        df5['month_sin'] = df5['month'].apply( lambda x: np.sin( x * (2 * np.pi/12) ) )
        df5['month_cos'] = df5['month'].apply( lambda x: np.cos( x * (2 * np.pi/12) ) )

        # week_of_year
        df5['week_of_year_sin'] = df5['week_of_year'].apply( lambda x: np.sin( x * (2 * np.pi/52) ) )
        df5['week_of_year_cos'] = df5['week_of_year'].apply( lambda x: np.cos( x * (2 * np.pi/52) ) )

        # day
        df5['day_sin'] = df5['day'].apply( lambda x: np.sin( x * (2 * np.pi/30) ) )
        df5['day_cos'] = df5['day'].apply( lambda x: np.cos( x * (2 * np.pi/30) ) )
        
        cols_selected = ['store', 'store_type', 'assortment', 'competition_distance',
                        'competition_open_since_month', 'competition_open_since_year', 'promo2',
                        'promo2_since_week', 'promo2_since_year', 'promo',
                        'competition_time_month', 'day_of_week_sin', 'day_of_week_cos',
                        'month_cos', 'week_of_year_cos', 'day_sin', 'day_cos']
        
        return df5[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict(test_data)
        
        # join pred into the original data
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient='records', date_format='iso')