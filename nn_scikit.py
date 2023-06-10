import numpy as np
import util
import random
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
# from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics  import f1_score,accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.datasets import make_classification
from sklearn.utils import resample

from embedding_encoder import EmbeddingEncoder

random.seed(42)

# Define categorical and numeric features
three_or_more_categorical = ["Industry_Sector", "Industry_Group", "Industry_Code",  "Global_Region",  "City", "Country", "Deal_Type", "Deal_Type_2", "Deal_Type_1", "Deal_Type_2_1"]
three_or_more_numeric = ["Close_Date", "_Deal_Size__millions__", "Mega_Deal_", "Pre_Value__millions_", 
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
two_or_more_categorical = ["Industry_Sector", "Industry_Group", "Industry_Code",  "Global_Region",  "City", "Country", "Deal_Type", "Deal_Type_2"]
two_or_more_numeric = ["Close_Date", "_Deal_Size__millions__", "Mega_Deal_", "Pre_Value__millions_", 
            "Post_Value__millions_", "Traditional_VC_Investor_Count", "Non_Traditional_VC_Investor_Count", "US_Investor_Count", "Europe_Investor_Count", 
            "Female_Founder_Count", "Investor_Count", "CVC_Investor_Involvement", "PE_Investor_Involvement", "Hedge_Fund_Investor_Involvement", 
            "Asset_Manager_Investor_Involvement", "Government_SWF_Investor_Involvement", "Number_of_Lead_Investors_on_Deal", 
            "Number_of_Non_Lead_Investors_on_Deal", "Number_of_New_Investors", "Number_of_New_Lead_Investors", "Number_of_Follow_On_Investors", 
            "Number_of_Lead_Follow_On_Investors", "At_Least_One_Lead_Investor_is_New_and_Got_Board_Seat", "Crossover_Investor_was_a_Lead_Investor", 
            "Notable_Investor_Count", "Notable_Investor_Involvement", "VC_Raised_to_Date", "Founding_Year", "VC_Deal_Number"]

def build_pipeline(encoding_mode, task, categorical, numeric):
    # categorical embedding pipeline
    if encoding_mode == "embedding":
        encoder = EmbeddingEncoder(task="classification", mapping_path="mapping.txt")
    # one hot encoding pipeline 
    elif encoding_mode == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore")
    
    scaler = StandardScaler()
    processor = ColumnTransformer([("embeddings", encoder, categorical), ("scale", scaler, numeric)])

    if task == "SGD":
        model = SGDClassifier(max_iter=-1)
    elif task == "logistic_regression":
        model = LogisticRegression(max_iter=-1)
    elif task == "SVC":
        model = SVC(max_iter=-1)

    return encoder, make_pipeline(processor, model)
    

def main(file_path, save_path):
    """
    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """

    df = pd.read_csv(file_path)

    print('Fitting model to', file_path)

    if file_path.startswith('three'):
        categorical = three_or_more_categorical
        numeric = three_or_more_numeric
    elif file_path.startswith('two'):
        categorical = two_or_more_categorical
        numeric = two_or_more_numeric

    Y = df["IsUnicorn"]
    X = df.drop(["IsUnicorn"], axis=1)
    x_train, x_eval, y_train, y_eval = train_test_split(X, Y, test_size=0.2)

    # Define models and encoding_modes
    # models = ["regression", "classification"]
    models = ["logistic_regression", "SVC", "SGD"]
    encoding_modes = ["embedding", "onehot"]
    sampling_modes = ["original", "upsampling"]

    save_text = ''

    for model in models:
        for sampling_mode in sampling_modes:
            path_var = './scores/original/accuracy_'
            if sampling_mode == "upsampling":
                    path_var = './scores/upsampled/accuracy_'
                    print("X Shape: ", x_train.shape, "Y Shape: ", y_train.shape)
                    print("Upsampling minority class...")
                    # combine them back for resampling
                    train_data = pd.concat([x_train, y_train], axis=1)
                    # separate minority and majority classes
                    not_unicorn = train_data[train_data.IsUnicorn==0]
                    unicorn = train_data[train_data.IsUnicorn==1]
                    # upsample minority
                    upsampled_unicorn = resample(unicorn, replace=True, n_samples=len(not_unicorn), random_state=27)
                    # combine majority and upsampled minority
                    upsampled_train_data = pd.concat([not_unicorn, upsampled_unicorn])
                    print(upsampled_train_data.IsUnicorn.value_counts())
                    # split back into X and Y
                    y_train = upsampled_train_data["IsUnicorn"]
                    x_train = upsampled_train_data.drop(["IsUnicorn"], axis=1)
                    print("Upsampled X Shape: ", x_train.shape, "Upsampled Y Shape: ", y_train.shape)
            for encoding_mode in encoding_modes:
                print('Running ' + model + ' with ' + encoding_mode + ' and ' + sampling_mode + ' data...')
                
                encoder, pipeline = build_pipeline(encoding_mode, model, categorical, numeric)
                pipeline.fit(x_train, y_train)
                pred = pipeline.predict(x_eval)

                if encoding_mode == "embedding":
                    encoder.fit(x_train, y_train)
                    for feature in categorical:
                        encoder.plot_embeddings(variable=feature, model="pca")
                        plot_path = './embeddings/' + feature + '/' + model + '_' + file_path + '.png'
                        plt.savefig(plot_path)
                        plt.close()

                # Have to convert predictions to binary values in order to get accuracy metrics
                pred_binary = (pred >= 0.5).astype(int)

                # generate confusion matrix
                cm = confusion_matrix(y_eval, pred_binary)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                plot_path = './confusion_matrices/' + 'cm_' + model + '_' + sampling_mode + '_' + file_path + '.png'
                plt.savefig(plot_path)
                plt.close()

                save_text += model + ' with ' + encoding_mode + ' and ' + sampling_mode + ' data:\n'
                save_text += "Accuracy Score: " + str(accuracy_score(y_eval,pred_binary)) + '\n'
                save_text += "F1 Score: " + str(f1_score(y_eval,pred_binary)) + '\n'
                np.savetxt('./predictions/' + model + "_" + encoding_mode + "_" + sampling_mode + '_' + save_path, pred)
                print('Completed ' + model + ' with ' + encoding_mode + ' and' + sampling_mode + ' data!\n')
            accuracy_file = open(path_var + file_path.split('.')[0] + ".txt", "w")
            accuracy_file.write(save_text)
            accuracy_file.close()


if __name__ == '__main__':
    main(file_path='three_or_more_flattened.csv',
         save_path='pred_three_or_more.txt')
    main(file_path='two_or_more_stripped.csv',
         save_path='pred_two_or_more.txt')