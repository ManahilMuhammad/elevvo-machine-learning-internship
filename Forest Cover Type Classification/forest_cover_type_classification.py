# --> BEGINNING OF: importing libraries
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
# <-- END OF: importing libraries

# --> BEGINNING OF: defining a dictionary of soil types
soil_types = {
    1: "Cathedral - Rock outcrop complex",
    2: "Vanet - Ratake complex",
    3: "Haploborolis - Rock outcrop",
    4: "Ratake family - Rock outcrop complex",
    5: "Vanet family - Rock outcrop complex",
    6: "Vanet - Wetmore complex",
    7: "Gothic family",
    8: "Supervisor - Limber complex",
    9: "Troutville family",
    10: "Bullwark - Catamount complex",
    11: "Bullwark - Catamount complex, rubbly",
    12: "Legault family - Rock outcrop complex",
    13: "Catamount family - Rock outcrop complex",
    14: "Pachic Argiborolis - Aquolis complex",
    15: "Unspecified in the USFS Soil and ELU Survey",
    16: "Cryaquolis - Cryoborolis complex",
    17: "Gateview family - Cryaquolis complex",
    18: "Rogert family, very stony",
    19: "Typic Cryaquolis - Borohemists complex",
    20: "Typic Cryaquepts - Typic Cryaquolls complex",
    21: "Typic Cryaquolls - Leighcan family complex",
    22: "Leighcan family - Typic Cryoboralf complex",
    23: "Leighcan family - Typic Cryochrept complex",
    24: "Leighcan family - Typic Cryaquept complex",
    25: "Leighcan family - Lithic Cryobarolfs complex",
    26: "Leighcan family - Typic Cryoboralf complex",
    27: "Leighcan family - Rock outcrop complex",
    28: "Leighcan family - Cryaquolis complex",
    29: "Leighcan family - Cryoborolls complex",
    30: "Leighcan family - Rock outcrop complex",
    31: "Cryorthents - Rock land complex",
    32: "Cryumbrepts - Rock outcrop complex",
    33: "Cryaquepts - Rock outcrop complex",
    34: "Leighcan family - Cryaquolls complex",
    35: "Leighcan family - Rock land complex",
    36: "Cryumbrepts - Leighcan family complex",
    37: "Cryochrepts - Leighcan family complex",
    38: "Cryaquepts - Leighcan family complex",
    39: "Leighcan family - Cryaquolls complex",
    40: "Cryorthents - Rock outcrop complex"
}
# <-- END OF: defining a dictionary of soil types

# --> BEGINNING OF: function to load and preprocess data
def get_data():
    # load data
    data = fetch_covtype(as_frame=True)
    
    # store data as dataframe
    df = data.frame

    # define list of features upon which classification will be based
    selected_features = [
        'Elevation',
        'Slope',
        'Horizontal_Distance_To_Roadways',
        'Horizontal_Distance_To_Hydrology',
        'Hillshade_9am'
    ]

    # extract relevant rows from dataframe
    X_base = df[selected_features]
    X_wild = df[[col for col in df.columns if col.startswith("Wilderness_Area")]]
    X_soil = df[[col for col in df.columns if col.startswith("Soil_Type")]]
    X = pd.concat([X_base, X_wild, X_soil], axis=1)
    y = df['Cover_Type'] - 1

    return X, y, selected_features
# <-- END OF: function to load and preprocess data

# --> BEGINNING OF: function to train model
def train_model(X, y):
    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=7,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # fit model to training data
    model.fit(X_train, y_train)
    
    # predict cover type using testing data
    # using model trained on training data
    y_pred = model.predict(X_test)
    
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    return model, X.columns, X_test, y_test, y_pred
# <-- END OF: function to train model

# --> BEGINNING OF: function to plot confusion matrix
def plot_confusion(y_test, y_pred):
    # create confusion matrix of predictions based on test data
    cm = confusion_matrix(y_test, y_pred)
    
    # display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.PuRd)
    plt.title("Confusion Matrix")
    plt.show()
# <-- END OF: function to plot confusion matrix

# --> BEGINNING OF: function to plot feature importance
def plot_feat_imp(model):
    plt.figure(figsize=(10, 5))
    plot_importance(model, max_num_features=10, importance_type='gain')
    plt.title("Top 10 Most Important Features")
    plt.show()
# <-- END OF: function to plot feature importance

# --> BEGINNING OF: function to create UI and get input
def create_ui(model, selected_features):
    
    # --> BEGINNING OF: function to predict cover type
    def predict():
        try:
            inputs = [float(entries[feature].get()) for feature in selected_features]
            
            # identify wilderness area chosen
            wilderness_one_hot = [0] * 4
            wilderness_idx = int(wilderness_var.get()[0])
            wilderness_one_hot[wilderness_idx] = 1

            # identify soil type chosen
            soil_label = soil_var.get()
            soil_idx = [k for k, v in soil_types.items() if v == soil_label][0] - 1
            soil_one_hot = [0] * 40
            soil_one_hot[soil_idx] = 1

            # make prediction based on inputs
            pred = model.predict([inputs + wilderness_one_hot + soil_one_hot])[0]
            
            # define a dictionary of textual definitions of each prediction
            cover_types = {
                0: "Spruce/Fir",
                1: "Lodgepole Pine",
                2: "Ponderosa Pine",
                3: "Cottonwood/Willow",
                4: "Aspen",
                5: "Douglas-fir",
                6: "Krummholz"
            }
            
            # output the cover type corresponding to the number predicted
            messagebox.showinfo("Prediction Result", f"Predicted Cover Type: {cover_types[pred]}")
        
        # error handling
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
    # <-- END OF: function to predict cover type

    # --> BEGINNING OF: function to reset all fields when reset button clicked
    def reset_fields():
        for field in entries.values():
            field.delete(0, tk.END)
        wilderness_var.set("0 - Rawah")
        soil_var.set(list(soil_types.values())[0])
    # <-- END OF: function to reset all fields when reset button clicked

    # --> BEGINNING OF: creating UI window for classifier
    root = tk.Tk()
    root.title("⸙ Forest Cover Type Classifier")
    root.geometry("700x600")
    root.resizable(False, False)
    root.configure(bg="#f5f6f7")
    # <-- END OF: creating UI window for classifier

    # --> BEGINNING OF: adding title to UI
    ttk.Label(root, text="Forest Cover Type Classifier", font=("Arial UI", 18, "bold")).pack(pady=10)
    ttk.Label(root, text="Provide the following information about an area to estimate its forest cover type.",
              font=("Arial UI", 10)).pack(pady=5)
    # <-- END OF: adding title to UI

    # separate input area
    frame = ttk.LabelFrame(root, text="Input Features", padding=15)
    frame.pack(padx=20, pady=10, fill="x")

    # initialise empty dictionary to store textual input 
    entries = {}

    # store user selections in dictionary
    for i, feature in enumerate(selected_features):
        ttk.Label(frame, text=f"{feature.replace('_', ' ')}:").grid(row=i, column=0, sticky=tk.W, pady=4)
        entry = ttk.Entry(frame, width=25)
        entry.grid(row=i, column=1, pady=4)
        entries[feature] = entry

    # create dropdown for wilderness area
    ttk.Label(frame, text="Wilderness Area:").grid(row=len(selected_features), column=0, sticky=tk.W, pady=4)
    wilderness_var = tk.StringVar()
    wilderness_cb = ttk.Combobox(frame, textvariable=wilderness_var, values=[
        "0 - Rawah", "1 - Neota", "2 - Comanche Peak", "3 - Cache la Poudre"
    ], width=30, state="readonly")
    wilderness_cb.grid(row=len(selected_features), column=1, pady=4)
    wilderness_cb.current(0)

    # create dropdown for soil type
    ttk.Label(frame, text="Soil Type:").grid(row=len(selected_features)+1, column=0, sticky=tk.W, pady=4)
    soil_var = tk.StringVar()
    max_soil_length = max(len(name) for name in soil_types.values())
    soil_cb = ttk.Combobox(
        frame,
        textvariable=soil_var,
        values=list(soil_types.values()),
        width=int(max_soil_length * 0.85),  # dynamic width
        state="readonly"
    )
    soil_cb.grid(row=len(selected_features)+1, column=1, pady=4)
    soil_cb.current(0)

    # initialise button properties
    button_frame = ttk.Frame(root)
    button_frame.pack(pady=15)
    
    # classification button
    ttk.Button(button_frame, text="✔ Classify", command=predict).grid(row=0, column=0, padx=10)
    
    # reset button
    ttk.Button(button_frame, text="⟲ Reset", command=reset_fields).grid(row=0, column=1, padx=10)

    root.mainloop()
# <-- END OF: function to create UI and get input

# --> BEGINNING OF: function to run programmme
if __name__ == "__main__":
    # get data
    X, y, selected_features = get_data()
    
    # train model
    model, feature_names, X_test, y_test, y_pred = train_model(X, y)
    
    # plot confusion matrix
    plot_confusion(y_test, y_pred)
    
    # plot feature importance
    plot_feat_imp(model)
    
    # create UI
    create_ui(model, selected_features)
# <-- END OF: function to run programmme
