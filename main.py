from text_processing import get_data
from model import train_final_model, evaluate_model
#from tensorflow.keras.models import load_model, saved_model


def main():
    '''
    Driver function for the model. Imports model and text processing scripts.
    '''
    X_train, X_val, X_test, y_train, y_val, y_test, max_len, combined_dim = get_data()
    model = train_final_model(X_train, X_val, y_train, y_val, max_len, combined_dim)
    evaluate_model(model, X_test, y_test)
    
    #play with as needed
    #final_model.save("final_model.h5")
    #print("Model saved to final_model.h5")
    #model = load_model("final_model.h5")
    #evaluate_model(model,X_test,y_test)

if __name__ == '__main__':
    main()
