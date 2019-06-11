import numpy as np
import pandas as pd
import os, time, pickle
from collections import Counter

        
def populateCombinedReview(users_df, business_df, reviews_df):
    sub_df = reviews_df.drop(["cool", "funny", "useful"], axis=1, errors="ignore")
    output_df = sub_df.merge(users_df, how="left", on="user_id").drop(["user_id"], axis=1)
    output_df = output_df.merge(business_df, how="left", on="business_id").drop(["business_id"], axis=1)

    return output_df

def analysisText(train_review_df,users_df):    
    print("Start with analysis text")
    # Create a set of vocabulary care about
    vocab = list({"wifi","parking","service","friendly","music","noise","food","good","terrible","average"})
    # Number of user 
    n = users_df.shape[0]
    
    # Set threshold
    threshold = 0
    
    # New dataframe for user data
    newData = np.zeros((n,len(vocab)),dtype=np.float64);

    for index in range(len(train_review_df)):
        text = train_review_df.iloc[index].at["text"]
        words = dict(Counter(text))
        for vocab_index in range(len(vocab)):
            if (words.get(vocab[vocab_index],0) >= threshold):
                    user_ID = train_review_df.iloc[index].at["user_id"]
                    user_index = users_df.index[users_df["user_id"] == user_ID]
                    newData[user_index-1,vocab_index] = 1;
    # Merge together with old user data
    newFeature = pd.DataFrame(newData,columns = vocab)
    cleanedData = users_df
    cleanedData = pd.concat([cleanedData,newFeature],axis=1);
    
    print("Finish with analysisText")
    return cleanedData;    
                    

if __name__ == '__main__':
    start = time.time()
    print("### Load csv")
    users_df = pd.read_csv("CleanedData/user.csv")
    business_df = pd.read_csv("CleanedData/business.csv")
    
    train_reviews_df = pd.read_csv("Data/train_reviews.csv").drop(["date", "review_id"], axis=1)
    train_reviews_stars = train_reviews_df["stars"]
    train_reviews_df = train_reviews_df.drop(["stars"], axis=1)
    
    train_test_reviews_df = pd.read_csv("Data/validate_queries.csv")
    train_test_reviews_stars = train_test_reviews_df["stars"]
    train_test_reviews_df = train_test_reviews_df.drop(["stars"], axis=1)
    
    test_reviews_df = pd.read_csv("Data/test_queries.csv")
    

    print("### Populate combined train reviews")
    new_users_df = analysisText(train_reviews_df,users_df)
    combined_train_review = populateCombinedReview(new_users_df, business_df, train_reviews_df)
    combined_train_review["review_stars"] = train_reviews_stars
    print("### Populate combined train test reviews")
    combined_train_test_review = populateCombinedReview(new_users_df, business_df, train_test_reviews_df)
    combined_train_test_review["review_stars"] = train_test_reviews_stars
    print("### Populate combined test reviews")
    combined_test_review = populateCombinedReview(new_users_df, business_df, test_reviews_df)
    
    print("### Write to csv")
#    train_test_review.to_csv("validate_queries_hl.csv")
#    test_review.to_csv("test_queries_hl.csv")
    
    combined_train_review.to_csv("CleanedData/train_reviews_combined_hl.csv", index=False)
    combined_train_test_review.to_csv("CleanedData/validate_queries_combined_hl.csv", index=False)
    combined_test_review.to_csv("CleanedData/test_queries_combined_hl.csv", index=False)
    
    end = time.time()
    print(f"# Took: {end - start} sec")
    print("### All done")