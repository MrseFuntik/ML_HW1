from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.linear_model import LinearRegression


df_train = pd.read_csv('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_train.csv')
df_train = df_train.drop(labels=list(df_train[df_train.duplicated(
    ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power',
     'torque', 'seats'])].index))
df_train = df_train.reset_index(drop=True)


def normal_values(value):
    try:
        return float(str(value).split()[0])
    except:
        return None


df_train['mileage'] = df_train['mileage'].apply(normal_values)
df_train['engine'] = df_train['engine'].apply(normal_values)
df_train['max_power'] = df_train['max_power'].apply(normal_values)

for column in ['mileage', 'engine', 'max_power', 'seats']:
    df_train[column] = df_train[column].fillna(df_train[column].median())

df_train['engine'] = df_train['engine'].apply(lambda x: int(x))
df_train['seats'] = df_train['seats'].apply(lambda x: int(x))

X_train = df_train[['year', 'km_driven','mileage','engine','max_power','seats']].copy()
y_train = df_train['selling_price'].copy()

model = LinearRegression()

model.fit(X_train.values, y_train.values)

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(itemm: Item):
    global model
    json_dict = dict(itemm)
    a = pd.DataFrame()
    df_dictionary = pd.DataFrame([json_dict])
    output = pd.concat([a, df_dictionary], ignore_index=True)
    output = output.drop(columns=['name', 'selling_price', 'torque', 'fuel', 'seller_type', 'transmission', 'owner'])

    def normal_values(value):
        try:
            return float(str(value).split()[0])
        except:
            return None

    output['mileage'] = output['mileage'].apply(normal_values)
    output['engine'] = output['engine'].apply(normal_values)
    output['max_power'] = output['max_power'].apply(normal_values)

    return abs(list(model.predict(output.values))[0])


@app.post("/predict_items")
def predict_items(items: List[Item]):
    global model
    predictions = []
    for json_obj in items:
        json_dict = dict(json_obj)
        a = pd.DataFrame()
        df_dictionary = pd.DataFrame([json_dict])
        output = pd.concat([a, df_dictionary], ignore_index=True)
        output = output.drop(
            columns=['name', 'selling_price', 'torque', 'fuel', 'seller_type', 'transmission', 'owner'])

        def normal_values(value):
            try:
                return float(str(value).split()[0])
            except:
                return None

        output['mileage'] = output['mileage'].apply(normal_values)
        output['engine'] = output['engine'].apply(normal_values)
        output['max_power'] = output['max_power'].apply(normal_values)
        json_dict['prediction'] = abs(list(model.predict(output.values))[0])
        predictions.append(json_dict)
    return predictions
