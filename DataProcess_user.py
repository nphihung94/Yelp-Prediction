
import pandas as pd


if __name__ == '__main__':
    rawData = pd.read_csv("Data/users.csv");
    
    # Drop unused attributes:
    unused_attrs = {"elite","friends","name","yelping_since","review_count"}
    cleanData = rawData.drop(unused_attrs,axis = 1);
    
    # Create dictionary of works care in text
    # write to Cleaned Data
    cleanData.to_csv("CleanedData/user.csv", index = False)
    print("Finish clean users")
    
    