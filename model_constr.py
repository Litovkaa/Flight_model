import pandas as pd
import numpy as np
import json
import datetime
import math
import scipy.stats as st
from statsmodels.stats.weightstats import _tconfint_generic
import xgboost as xgb
from sklearn.model_selection import train_test_split
#import timezonefinder

# ##############################################################################################################
# ##########################################Подготовка данных###################################################
# ##############################################################################################################
def train_data_preparation(dat):
    # Очистка от строк с пустыми значениями во всех стобцах, кроме aircraft_model ,real_departure_time,real_arrival_time
    dat = dat.dropna(subset = ['flight_number', 'scheduled_departure_time', 'scheduled_arrival_time', 'airport_origin', 'airport_destination', 'airline_name', 'flight_status'])
    # Oчистка от неугодных данных (Scheduled, Estimated, Unknown)
    dat = dat.loc[(~dat[["real_departure_time", "real_arrival_time"]].isna().sum(axis=1) & dat.flight_status.str.contains("Landed")) | (dat.flight_status.str.contains("Canceled") | dat.flight_status.str.contains("Diverted"))]
        
    # Чтение csv файла c АП
    airport_dat = pd.read_csv(u"C:/Users/sam/Desktop/reysy/AP.txt", sep = ",", encoding = "ISO-8859-1")
    
    # джоин данных об аэропортах к основной таблице
    dat = pd.merge(dat, airport_dat[['city', 'country', 'iata', 'latitude', 'longitude', 'elevation', 'class',  'timezone']], how = "left", left_on = "airport_origin", right_on = "iata")
    dat = pd.merge(dat, airport_dat[['city', 'country', 'iata', 'latitude', 'longitude', 'elevation', 'class',  'timezone']], how = "left", left_on = "airport_destination", right_on = "iata", suffixes = ("_airport_origin", "_airport_destination"))
    
    # очистка от неверно записанных аэропрортов и прочей нечисти, вдруг такого АП нет в списке (скорее всего это закрытый аэропорт, как в Ростове например)
    dat = dat.dropna(subset = ['city_airport_origin', 'country_airport_origin', 'iata_airport_origin', 'latitude_airport_origin', 'longitude_airport_origin', 'elevation_airport_origin', 'class_airport_origin',  'timezone_airport_origin',
                               'city_airport_destination', 'country_airport_destination', 'iata_airport_destination', 'latitude_airport_destination', 'longitude_airport_destination', 'elevation_airport_destination', 'class_airport_destination',  'timezone_airport_destination'])
    
    # ####Приведение дат к datetime и пересчет всех дат на МСК время
    dat.scheduled_arrival_time = dat.apply(lambda row: pd.to_datetime(pd.Timestamp(row["scheduled_arrival_time"],unit="s").tz_localize(row["timezone_airport_destination"])), axis = 1)
    dat.scheduled_departure_time = dat.apply(lambda row: pd.to_datetime(pd.Timestamp(row["scheduled_departure_time"],unit="s").tz_localize(row["timezone_airport_origin"])), axis = 1)
    dat.real_arrival_time = dat.apply(lambda row: pd.to_datetime(pd.Timestamp(row["real_arrival_time"],unit="s").tz_localize(row["timezone_airport_destination"])), axis = 1)
    dat.real_departure_time = dat.apply(lambda row: pd.to_datetime(pd.Timestamp(row["real_departure_time"],unit="s").tz_localize(row["timezone_airport_origin"])), axis = 1)
    
    # перевод на единое московское время, как на вокзале ей богу
    dat["scheduled_arrival_time_msk"] = dat.apply(lambda row: row["scheduled_arrival_time"].tz_convert("Europe/Moscow"), axis = 1)
    dat["scheduled_departure_time_msk"] = dat.apply(lambda row: row["scheduled_departure_time"].tz_convert("Europe/Moscow"), axis = 1)
    dat["real_arrival_time_msk"] = dat.apply(lambda row: row["real_arrival_time"].tz_convert("Europe/Moscow"), axis = 1)
    dat["real_departure_time_msk"] = dat.apply(lambda row: row["real_departure_time"].tz_convert("Europe/Moscow"), axis = 1)
    
    # ####Расчет доп столбцов и очистка от заблудших строк
    # Расчет задержек
    dat["departure_delay"] = ((dat.real_departure_time_msk - dat.scheduled_departure_time_msk)/np.timedelta64(1,'s'))//60
    dat["arrival_delay"] = ((dat.real_arrival_time_msk - dat.scheduled_arrival_time_msk)/np.timedelta64(1,'s'))//60
    dat["delays_diff"] = np.abs(dat.departure_delay - dat.arrival_delay)
    
    # очистка от рейсов, в которых разница в задержке вылета и задержке прилета больше 2 часов,
    # типа ну не может самолет больше двух часов кружить над городом, тут где-то кракозябра в данных - такие зачищаем
    dat = dat.loc[dat.delays_diff < 120]
    return dat

def test_data_preparation(dat):
    # Чтение csv файла c АП
    airport_dat = pd.read_csv(u"C:/Users/sam/Desktop/reysy/AP.txt", sep = ",", encoding = "ISO-8859-1")
    
    # джоин данных об аэропортах к основной таблице
    dat = pd.merge(dat, airport_dat[['city', 'country', 'iata', 'latitude', 'longitude', 'elevation', 'class',  'timezone']], how = "left", left_on = "airport_origin", right_on = "iata")
    dat = pd.merge(dat, airport_dat[['city', 'country', 'iata', 'latitude', 'longitude', 'elevation', 'class',  'timezone']], how = "left", left_on = "airport_destination", right_on = "iata", suffixes = ("_airport_origin", "_airport_destination"))
        
    # ####Приведение дат к datetime и пересчет всех дат на МСК время
    dat.scheduled_arrival_time = dat.apply(lambda row: pd.to_datetime(pd.Timestamp(row["scheduled_arrival_time"],unit="s").tz_localize(row["timezone_airport_destination"])), axis = 1)
    dat.scheduled_departure_time = dat.apply(lambda row: pd.to_datetime(pd.Timestamp(row["scheduled_departure_time"],unit="s").tz_localize(row["timezone_airport_origin"])), axis = 1)
    
    # перевод на единое московское время, как на вокзале ей богу
    dat["scheduled_arrival_time_msk"] = dat.apply(lambda row: row["scheduled_arrival_time"].tz_convert("Europe/Moscow"), axis = 1)
    dat["scheduled_departure_time_msk"] = dat.apply(lambda row: row["scheduled_departure_time"].tz_convert("Europe/Moscow"), axis = 1)
    return(dat)


# ##############ФИЧ ИНЖИНИРИНГ###############
# Функция, вычисляющая дамми переменные для входной категориальной и отсеивающая те столбцы,
# в которых крайне низкая дисперсия, выбивающаяся из общего ряда категорий, хотя мб эта очистка и не нужна вовсе
def clean_from_rare_dummies (ser, prefix, typ = "train", clean = True, count_na = False, other_name = "other", alpha = 0.03, train_feature_names = []):
      dummies = pd.get_dummies(ser, prefix = prefix, dummy_na = count_na)
      if clean: # если нужно очищать от редких
          if typ == "train": # если мы берем дамми по образу трейна, то нужно выбирать те фичи, которые были в трейне
              col_mean = dummies.mean(axis = 0)
              minimal_share_of_category = _tconfint_generic(np.mean(col_mean), np.std(col_mean, ddof = 1)/np.sqrt(len(col_mean)), len(col_mean) - 1, alpha, 'two-sided')[0]
              colnames_to_save = dummies.columns[col_mean >= minimal_share_of_category]
              colnames_to_drop = dummies.columns[col_mean < minimal_share_of_category]
          else:
              colnames_to_save = dummies.columns[dummies.columns[dummies.columns in train_feature_names]]
              colnames_to_drop = dummies.columns[dummies.columns[dummies.columns not in train_feature_names]]
          print("=====FEATURE "+prefix+"=====")
          print("total_categories:"+str(len(col_mean)))
          print("categories_to_save:"+str(len(colnames_to_save)))
          print("categories_to_drop:"+str(len(colnames_to_drop)))
          print("list_to_drop:", list(colnames_to_drop))
          if len(colnames_to_drop) > 0:
              dummies[prefix + "_" + other_name] = dummies[colnames_to_drop].sum(axis = 1)
              return pd.concat([dummies[colnames_to_save], dummies[prefix + "_" + other_name]], axis = 1)
          else:
              return dummies[colnames_to_save]
      else:
            return dummies

# функция, возвращающая фичи из сырых данных для трейна и теста
def create_features(dat, typ = "train", train_feature_names = []):
    # каждый полет может быть определен по номеру рейса и дате вылета
    X = dat[["flight_number", "scheduled_departure_time_msk"]]
    # ####Временные фичис
    # Длительность полета
    X["scheduled_flight_duration"] = ((dat.scheduled_arrival_time_msk - dat.scheduled_departure_time_msk)/np.timedelta64(1,'s'))//60
    
    # синус и косинус номера дня в году по местному времени!!!
    X["cos_scheduled_departure_dayofyear"] = np.cos(dat.apply(lambda row: row["scheduled_departure_time"].dayofyear / (365 * (1-row["scheduled_departure_time"].is_leap_year) + 366 * row["scheduled_departure_time"].is_leap_year) * 2 * np.pi , axis = 1))
    X["sin_scheduled_departure_dayofyear"] = np.sin(dat.apply(lambda row: row["scheduled_departure_time"].dayofyear / (365 * (1-row["scheduled_departure_time"].is_leap_year) + 366 * row["scheduled_departure_time"].is_leap_year) * 2 * np.pi , axis = 1))
    
    # синус и косинус номера минуты вылета и прилета в сутках по местному времени!!!
    X["cos_scheduled_departure_minuteofday"] = np.cos(dat.apply(lambda row: (row["scheduled_departure_time"].hour * 60 + row["scheduled_departure_time"].minute) / 1440 * 2 * np.pi , axis = 1))
    X["sin_scheduled_departure_minuteofday"] = np.sin(dat.apply(lambda row: (row["scheduled_departure_time"].hour * 60 + row["scheduled_departure_time"].minute) / 1440 * 2 * np.pi , axis = 1))
    X["cos_scheduled_arrival_minuteofday"] = np.cos(dat.apply(lambda row: (row["scheduled_arrival_time"].hour * 60 + row["scheduled_arrival_time"].minute) / 1440 * 2 * np.pi , axis = 1))
    X["sin_scheduled_arrival_minuteofday"] = np.sin(dat.apply(lambda row: (row["scheduled_arrival_time"].hour * 60 + row["scheduled_arrival_time"].minute) / 1440 * 2 * np.pi , axis = 1))
    
    # дамми день недели (без очистки)
    temp_dat = clean_from_rare_dummies(dat.scheduled_departure_time_msk.dt.dayofweek, "is_weekday", clean = False, typ = typ, train_feature_names = train_feature_names)
    X = pd.concat([X, temp_dat], axis = 1)
    
    # ####Фичис АК, АП и географические
    # дамми АК (без очистки)
    temp_dat = clean_from_rare_dummies(dat.airline_name, "is_airline", clean = False, typ = typ, train_feature_names = train_feature_names)
    X = pd.concat([X, temp_dat], axis = 1)
    
    # дамми страна вылета (без очистки)
    temp_dat = clean_from_rare_dummies(dat.country_airport_origin, "is_origin_country", clean = False, typ = typ, train_feature_names = train_feature_names)
    X = pd.concat([X, temp_dat], axis = 1)
    
    # дамми страна прилета (без очистки)
    temp_dat = clean_from_rare_dummies(dat.country_airport_destination, "is_destination_country", clean = False, typ = typ, train_feature_names = train_feature_names)
    X = pd.concat([X, temp_dat], axis = 1)
    
    # дамми АП вылета (без очистки)
    temp_dat = clean_from_rare_dummies(dat.airport_origin, "is_origin_airport", clean = False, typ = typ, train_feature_names = train_feature_names)
    X = pd.concat([X, temp_dat], axis = 1)
    
    # дамми АП прилета (без очистки)
    temp_dat = clean_from_rare_dummies(dat.airport_destination, "is_destination_airport", clean = False, typ = typ, train_feature_names = train_feature_names)
    X = pd.concat([X, temp_dat], axis = 1)
    
    # координаты и высота над уровнем моря, АП
    X['latitude_airport_origin'] = dat['latitude_airport_origin']
    X['longitude_airport_origin'] = dat['longitude_airport_origin']
    X['elevation_airport_origin'] = dat['elevation_airport_origin']
    X['latitude_airport_destination'] = dat['latitude_airport_destination']
    X['longitude_airport_destination'] = dat['longitude_airport_destination']
    X['elevation_airport_destination'] = dat['elevation_airport_destination']
    
    # дамми класс аэропорта (что это такое я не понял, ну да и пох, это ж информация))) (без очистки)
    temp_dat = clean_from_rare_dummies(dat.class_airport_origin, "is_class_origin_airport", clean = False, typ = typ, train_feature_names = train_feature_names)
    X = pd.concat([X, temp_dat], axis = 1)
    temp_dat = clean_from_rare_dummies(dat.class_airport_destination, "is_class_destination_airport", clean = False, typ = typ, train_feature_names = train_feature_names)
    X = pd.concat([X, temp_dat], axis = 1)
    
    # ####Фичис самолета
    # дамми модель самолета (без очистки)
    temp_dat = clean_from_rare_dummies(dat.aircraft_model, "is_aircraft_model", clean = False, count_na = True, typ = typ, train_feature_names = train_feature_names)
    X = pd.concat([X, temp_dat], axis = 1)
    
    # сохранение имен фич
    feature_names = X.columns[2:]
    # результат - матрица X, имена фич X
    return [X, feature_names]
    
def create_labels(dat):
    # сохранение имен откликов
    y_names = (np.char.array("y") + np.char.array(np.array(range(12))+1)); y_names = np.append(y_names, "y_canceled").tolist()
    
    # ####Расчет отклика моделей
    Y = dat[["flight_number", "scheduled_departure_time_msk"]]
    Y["y_canceled"] = (dat.flight_status.str.contains("Canceled") | dat.flight_status.str.contains("Diverted")).astype(int)
    for i in (np.array(range(12))+1):
        Y["y"+str(i)] = (dat.arrival_delay >= 60*i | Y["y_canceled"]).astype(int)
    
    # результат - матрица X, матрица Y, имена фич X и названия всех Y
    return [Y, y_names]


# ##############################################################################
# ################################# MODEL ######################################
# ##############################################################################

# Разбиение на трейн и тест и вал. Тест и вал = все рейсы 2018 года - ровно пополам
def data_split(X, Y):
    X_train = X.loc[X.apply(lambda row: row["scheduled_departure_time_msk"].year, axis = 1) != 2018]
    Y_train = Y.loc[Y.apply(lambda row: row["scheduled_departure_time_msk"].year, axis = 1) != 2018]
    
    X_test, X_val, Y_test, Y_val  = train_test_split(X.loc[X.apply(lambda row: row["scheduled_departure_time_msk"].year, axis = 1) == 2018],
                                                                      Y.loc[Y.apply(lambda row: row["scheduled_departure_time_msk"].year, axis = 1) == 2018],
                                                                      test_size = 0.5)
    return [X_train, Y_train, X_test, Y_test, X_val, Y_val]

# обучение 13 моделей и запись их в DF
def create_models(X, Y, feature_names, y_names):
    # создание DF, где будут храниться модели
    models = pd.DataFrame(index = range(13), columns = ["label_name", "model"]); models.label_name = y_names
    # собсна обучение всех 13ти моделей
    for n in y_names:
        XGB_train = xgb.DMatrix(X_train[feature_names], label=Y_train[n])
        XGB_val = xgb.DMatrix(X_val[feature_names], label=Y_val[n]) 
        print("start_training")
        param = {'objective' : "binary:logistic",
                 'booster' : "gblinear",
                 'eval_metric' : "logloss",
                 'eta' : 0.05,
                 'gamma': 2,
                 'data' : XGB_train,
                 'max_depth' : 3,
                 'subsample' : 0.8,
                 'colsample_bytree' : 0.8,
                 'min_child_weight' : 5,
                 'verbose': 1,
                 'print_every_n' : 1,
                 'early_stopping_rounds' : 5,
                 'verbose_eval': 1}
        models.model.loc[models.label_name == n]= xgb.train(param,
                                                            XGB_train,
                                                            300,
                                                            [(XGB_train, "train"), (XGB_val, "val")],
                                                            early_stopping_rounds = 5)
        print("finish_training "+n)
    return models

# def check_model_metrics_test(models, X_test, Y_test):
    


# ##############################################################################
# ################################# MAIN #######################################
# ##############################################################################
# ####Чтение и джоин таблиц
# Чтение csv файла c рейсами
dat = pd.read_csv("C:/Users/sam/Desktop/reysy/roma_data.csv")
dat.columns = np.array(['flight_number', 
                        'scheduled_departure_time', 
                        'scheduled_arrival_time', 
                        'airport_origin', 
                        'airport_destination',
                        'airline_name', 
                        'real_departure_time', 
                        'real_arrival_time',
                        'aircraft_model',
                        'flight_status'])
dat = train_data_preparation(dat)
[X, feature_names] = create_features(dat)
[Y, y_names] = create_labels(dat)
models = create_models(X, Y, feature_names, y_names)














