import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def car_mileage_predictor(horsepower, weight):
    # the data set is available at the url below.
    URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/mpg.csv"

    # using the read_csv function in the pandas library, we load the data into a dataframe.

    df = pd.read_csv(URL)

    # Clean the data
    df["MPG"].fillna(df["MPG"].mean())
    df["Horsepower"].fillna(df["Horsepower"].mean())
    df["Weight"].fillna(df["Weight"].mean())
    df["Engine Disp"].fillna(df["Engine Disp"].mean())
    df["Cylinders"].fillna(df["Cylinders"].mean())


    # plot the data
    df.plot.scatter(x = "Horsepower", y = "MPG")
    plt.savefig("car_mileage.png")

    # define features(X) and target(Y)
    X = df[["Horsepower", "Weight"]]  
    Y = df["MPG"]
    
    # split the data into train(80%) and test(20%) and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

    # create a linear regression object
    model=LinearRegression()

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    # List of individual R2 scores from different runs or datasets
    individual_r2_scores = []

    # Calculate MSE and R2 score
    mse = mean_squared_error(Y_test, Y_pred)
    for i in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        individual_r2_scores.append(r2_score(Y_test, Y_pred))
    
    # r2 = r2_score(Y_test, Y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared (R2) Score Maximum : {max(individual_r2_scores)}")
    print(f"R-squared (R2) Score Minimum : {min(individual_r2_scores)}")
    
    # Calculate the average R2 score
    average_r2_score = sum(individual_r2_scores) / len(individual_r2_scores)

    # Print the average R2 score
    print("Average R-squared (R2) Score:", average_r2_score)

    # print the predictions value 
    print("Predicted MPG :", model.predict([[horsepower, weight]])[0])




def diamond_price_predictor(caret, depth):
    URL="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/diamonds.csv"

    # using the read_csv function in the pandas library, we load the data into a dataframe.
    df = pd.read_csv(URL)
    # df.to_csv("diamonds.csv")
    # print(df.shape)
    # print(df.head())
    # print(df.dtypes)
    # Clean the data
    df["price"].fillna(df["price"].mean())
    df["carat"].fillna(df["carat"].mean())
    df["depth"].fillna(df["depth"].mean())

    # plot the data
    df.plot.scatter(x = "carat", y = "price")
    plt.savefig("diamond_price.png")

    # define features(X) and target(Y)
    X = df[["carat", "depth"]]
    Y = df["price"]
    # split the data into train(80%) and test(20%) and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    # create a linear regression object
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    # List of individual R2 scores from different runs or datasets
    individual_r2_scores = []
    # Calculate MSE and R2 score
    mse = mean_squared_error(Y_test, Y_pred)
    for i in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        individual_r2_scores.append(r2_score(Y_test, Y_pred))
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared (R2) Score Maximum : {max(individual_r2_scores)}")
    print(f"R-squared (R2) Score Minimum : {min(individual_r2_scores)}")
    # Calculate the average R2 score
    average_r2_score = sum(individual_r2_scores) / len(individual_r2_scores)
    # Print the average R2 score
    print(f"Average R2 Score: {average_r2_score}")
    print("Predicted Price :", model.predict([[caret, depth]])[0])





if __name__ == "__main__":
    # Call the car_mileage_predictor function
    # the function takes two arguments: horsepower and weight
    # horsepower is the horsepower of the car
    # weight is the weight of the car
    # horsepower is the integer value
    # weight is the integer value

    #------------------------------------------------------------------
    
    # Call the diamond_price_predictor function
    # the function takes two arguments: caret and depth
    # caret is the carat of the diamond
    # depth is the depth of the diamond
    # caret is the float value
    # depth is the float value
    #------------------------------------------------------------------
    print("####### Welcome to the car mileage predictor and diamond price predictor #######")
    print("Example of input values:\nhorsepower=198, weight=3850\ncaret=0.28, depth=55.5")
    while True:
        print("\nChoose any one of the following options:")
        print("1. Car Mileage Predictor")
        print("2. Diamond Price Predictor")
        print("3. Exit")
        option = int(input("Enter your choice: "))
        if option == 1: 
            horsepower = int(input("Enter the horsepower of the car: "))
            weight = int(input("Enter the weight of the car: "))
            print("\n")
            # horsepower = 200
            # weight = 400
            car_mileage_predictor(horsepower,weight)
            print("\n")  
        elif option == 2:
            caret=float(input("Enter the carat of the diamond: "))
            depth=float(input("Enter the depth of the diamond: "))
            print("\n")
            # caret = 0.3
            # depth = 60
            diamond_price_predictor(caret,depth)
            print("\n")
        elif option == 3:
            print("\nThank you for using the car mileage predictor and diamond price predictor\n")
            break
        else:
            print("\nInvalid option\n")


