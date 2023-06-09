import numpy as np
import util
import random
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics  import f1_score,accuracy_score

from embedding_encoder import EmbeddingEncoder

random.seed(42)

# Define categorical and numeric features
categorical = ["Industry_Sector", "Industry_Group", "Industry_Code",  "Global_Region",  "City", "Country", "Deal_Type", "Deal_Type_2", "Deal_Type_1", "Deal_Type_2_1"]
numeric = ["First_Unicorn_Round_Flag", "Close_Date", "_Deal_Size__millions__", "Mega_Deal_", "Pre_Value__millions_", 
            "Post_Value__millions_", "Traditional_VC_Investor_Count", "Non_Traditional_VC_Investor_Count", "US_Investor_Count", "Europe_Investor_Count", 
            "Female_Founder_Count", "Investor_Count", "CVC_Investor_Involvement", "PE_Investor_Involvement", "Hedge_Fund_Investor_Involvement", 
            "Asset_Manager_Investor_Involvement", "Government_SWF_Investor_Involvement", "Number_of_Lead_Investors_on_Deal", 
            "Number_of_Non_Lead_Investors_on_Deal", "Number_of_New_Investors", "Number_of_New_Lead_Investors", "Number_of_Follow_On_Investors", 
            "Number_of_Lead_Follow_On_Investors", "At_Least_One_Lead_Investor_is_New_and_Got_Board_Seat", "Crossover_Investor_was_a_Lead_Investor", 
            "Notable_Investor_Count", "Notable_Investor_Involvement", "VC_Raised_to_Date", "Founding_Year", "VC_Deal_Number", "Close_Date_1", 
            "_Deal_Size__millions___1", "Mega_Deal__1", "Pre_Value__millions__1", "Post_Value__millions__1", "Traditional_VC_Investor_Count_1", 
            "Non_Traditional_VC_Investor_Count_1", "US_Investor_Count_1", "Europe_Investor_Count_1", "Female_Founder_Count_1", "Investor_Count_1", 
            "CVC_Investor_Involvement_1", "PE_Investor_Involvement_1", "Hedge_Fund_Investor_Involvement_1", "Asset_Manager_Investor_Involvement_1", 
            "Government_SWF_Investor_Involvement_1", "Number_of_Lead_Investors_on_Deal_1", "Number_of_Non_Lead_Investors_on_Deal_1", "Number_of_New_Investors_1", 
            "Number_of_New_Lead_Investors_1", "Number_of_Follow_On_Investors_1", "Number_of_Lead_Follow_On_Investors_1", 
            "At_Least_One_Lead_Investor_is_New_and_Got_Board_Seat_1", "Crossover_Investor_was_a_Lead_Investor_1", "Notable_Investor_Count_1", 
            "Notable_Investor_Involvement_1", "VC_Raised_to_Date_1"]

def build_pipeline(mode, task):
    # categorical embedding pipeline
    if mode == "embedding":
        encoder = EmbeddingEncoder(task=task, mapping_path="mapping.txt")
    # one hot encoding pipeline 
    elif mode == "one hot":
        encoder = OneHotEncoder(handle_unknown="ignore")
    
    scaler = StandardScaler()
    processor = ColumnTransformer([("embeddings", encoder, categorical), ("scale", scaler, numeric)])

    if task == "regression":
        model = MLPRegressor(max_iter=300, activation="relu", hidden_layer_sizes= (100, 100))
    elif task == "classification":
        model = LogisticRegression()

    return make_pipeline(processor, model)
    

def main(file_path, save_path):
    """
    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """

    df = pd.read_csv(file_path)

    Y = df["IsUnicorn"]
    X = df.drop(["IsUnicorn"], axis=1)

    x_train, x_eval, y_train, y_eval = train_test_split(X, Y, test_size=0.2)

    # Define methods
    methods = {"regression", "classification"}
 

    for method in methods:
        pipeline = build_pipeline("embedding", method)

        pipeline.fit(x_train, y_train)
        pred = pipeline.predict(x_eval)

        # Have to convert predictions to binary values in order to get accuracy metrics
        pred_binary = (pred >= 0.5).astype(int)

        print("Accuracy Score: ", accuracy_score(y_eval,pred_binary))
        print("F1 Score: ", f1_score(y_eval,pred_binary))
        print("\n")

        np.savetxt(method + "_" + "embedding" + "_" + save_path, pred)

if __name__ == '__main__':
    # edit
    main(file_path='three_or_more_flattened.csv',
         save_path='pred.txt')