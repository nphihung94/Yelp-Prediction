Data Link:
https://www.kaggle.com/c/yelpratingprediction/data
Projects:
- Use provided data to make prediction about rating of a user to a business

Given Data:
- Users.csv
- Business.csv
- Train, Test and validation queries

Files in Project:
- DataProcess_business.py:
	Remove columns not used for model:
		Address
		Attributes
		Categories
		Hours: hours_Monday,...., hours_Sunday
		Postal_code
		Latitude; Longitude
		Name
		Neighborhood
		State; City
	Clean boolean columns into number with:
		True: 1.0 
		False: -1.0
		Empty cell: 0
	Change categorical data in to number using a list of value for each columns:
		attributes_AgesAllowed: ["0": 0, "18plus": 1, "19plus" : 2, "21plus": 3 , "allages": 4],
		attributes_Alcohol: ["0" :0 ,"none": 1,"beer_and_wine" :2,"full_bar" :3],
		attributes_BYOBCorkage: ["0" : 0,"no" :1,"yes_corkage" :2,"yes_free":3],
		attributes_NoiseLevel:["0": 1,"very_loud" : 2,"loud" : 3,"average" : 4,"quiet" :4],     
		attributes_WiFi:["0": 0,"no" : 1,"paid" :2 ,"free" :3],
		attributes_Smoking: ["0" :0,"no" : 1,"yes" :2,"outdoor":3],
		attributes_RestaurantsAttire: ["0":1,"casual":2,"dressy":3,"formal":4]
	# For the list of value: “0” is used to make all empty cell to 0.
	Make all the dictionary kind of data into one hot style:
		Ambience
		BestNights
		Business Parking
		Dietary Restriction
		Good For Meal
		Hair Specializes 
		Music

- DataProcess_user.py:
	Remove columns not used for model:
		Elite
		Friends
		Name
		Yelping_since
		Review_count
	Changed all empty cell to 0. 

- dataHandling:
	Merge user data and business data with rating base on the train, test and validation queries
- Yelp_predict.py:
	Use sklearn library to make predictions
	Use metrics library to calculate accuracy