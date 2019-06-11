import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import LabelEncoder, OneHotEncoder;

def cleanBoolean(Data,attr):
    Data[attr].fillna(0,inplace = True);
    Data.loc[cleanData[attr] == True , attr ] =  1.0;
    Data.loc[cleanData[attr] == False, attr ] = -1.0;

def isEmpty(any_structure):
    if any_structure:
        return False
    else:
        return True

def cleanDict(Data,attr):
    print(attr);
    Data[attr].fillna(0,inplace = True);
    # number of data points
    n = Data.shape[0];

    # create list of new attributes
    new_att_list = [];
    for i in range(n):
        data_entry = Data.iloc[i].at[attr];
        if (not isEmpty(data_entry)):
            possibleAttr = ast.literal_eval(data_entry);
            for key in possibleAttr.keys():
                if key not in new_att_list:
                    new_att_list.append(key);
    # create new data with size n x len(new_att_list) of 1/0:
    newData = np.zeros((n,len(new_att_list)),dtype=np.float64);
    for i in range(n):
        data_entry = Data.iloc[i].at[attr];
        if (not isEmpty(data_entry)):
            data_entry = ast.literal_eval(data_entry);
            for key in data_entry:
                if key in new_att_list:
                    if (data_entry.get(key)):
                        newData[i,new_att_list.index(key)] =  1.0;
                    else:
                        newData[i,new_att_list.index(key)] = -1.0;
    print("finish");
    # create Dataframe with new_att_list and data
    for i in range(len(new_att_list)):
        new_att_list[i] = attr + "_" + new_att_list[i];
    newAttr = pd.DataFrame(newData,columns = new_att_list);
    cleanedData = Data.drop(attr,axis = 1);
    cleanedData = pd.concat([cleanedData,newAttr],axis=1);
    return cleanedData;    
        

if __name__ == '__main__':
    rawData = pd.read_csv("Data/business.csv");
    
    # Drop all attr unused
    unused_attr = ["address","attributes","categories",
                    "hours","hours_Monday","hours_Tuesday",
                    "hours_Wednesday", "hours_Thursday",
                    "hours_Friday", "hours_Saturday", "hours_Sunday",
                    "postal_code", "latitude", "longitude",
                    "name","neighborhood","state","city"];
    cleanData = rawData.drop(unused_attr,axis = 1);
    
    # Change all bool attr to 1 and 0
    boolean_attr = ["attributes_AcceptsInsurance", "attributes_BikeParking", 
                       "attributes_BusinessAcceptsBitcoin", "attributes_BusinessAcceptsCreditCards",
                       "attributes_BYOB","attributes_ByAppointmentOnly", "attributes_Caters", 
                       "attributes_CoatCheck",  
                       "attributes_DogsAllowed", "attributes_DriveThru","attributes_Corkage",
                       "attributes_GoodForDancing", "attributes_GoodForKids",
                       "attributes_HappyHour", "attributes_HasTV", 
                       "attributes_Open24Hours", "attributes_OutdoorSeating", 
                       "attributes_RestaurantsCounterService", "attributes_RestaurantsDelivery", 
                       "attributes_RestaurantsGoodForGroups", "attributes_RestaurantsReservations", 
                       "attributes_RestaurantsTableService", "attributes_RestaurantsTakeOut",
                       "attributes_WheelchairAccessible"];
    for attrs in boolean_attr:
        cleanBoolean(cleanData,attrs);
    
    # Encode data that are categorical data
    categorical_attr = ["attributes_AgesAllowed", "attributes_Alcohol",
                        "attributes_BYOBCorkage","attributes_NoiseLevel",
                        "attributes_WiFi","attributes_Smoking","attributes_RestaurantsAttire"]
    label_Attributes = {"attributes_AgesAllowed": ["0","18plus","19plus","21plus","allages"],
                       "attributes_Alcohol": ["0","none","beer_and_wine","full_bar"],
                        "attributes_BYOBCorkage": ["0","no","yes_corkage","yes_free"],
                        "attributes_NoiseLevel":["0","very_loud","loud","average","quiet"],     
                        "attributes_WiFi":["0","no","paid","free"],
                        "attributes_Smoking": ["0","no","yes","outdoor"],
                       "attributes_RestaurantsAttire": ["0","casual","dressy","formal"]}
    for attrs in categorical_attr:
        encoder = LabelEncoder();
        cleanData[attrs].fillna("0",inplace = True);
        encoder.fit(label_Attributes.get(attrs));
        cleanData[attrs] = encoder.transform(cleanData[attrs].astype(str));
    
    # Clean data with value is dictionary
    list_attr = ["attributes_Ambience","attributes_BestNights","attributes_BusinessParking",
                 "attributes_DietaryRestrictions","attributes_GoodForMeal",
                 "attributes_HairSpecializesIn","attributes_Music"]
    for attrs in list_attr:
        cleanData = cleanDict(cleanData,attrs);
    
    # write to Cleaned Data
    cleanData.to_csv("CleanedData/business.csv", index = False)
    print("Finish clean Bussiness")