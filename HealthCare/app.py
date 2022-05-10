from msilib.schema import RadioButton
from operator import truediv
from pickle import TRUE
from tabnanny import check
from click import option
from flask import Flask, jsonify, render_template, request
import pandas as pd
from nltk.stem import WordNetLemmatizer
from keras.models import model_from_json
lemmatizer = WordNetLemmatizer()
import json
import nltk
import numpy as np
import pickle

app = Flask(__name__)

infile = open('words.pkl','rb')
words = pickle.load(infile)
infile2 = open('classes.pkl','rb')
classes = pickle.load(infile2)
data_file = open('static/Disease_Category_intenets_.json').read()
intents = json.loads(data_file)
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))
def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_disease(symptom):
    symptom=symptom.split(",")
    print(symptom)
    disease=[]
    for i in range(0,len(symptom)):
        if("Above 100 degree F" in symptom[i]):
            disease.append("high_fever")
        elif("Below 100 degree F" in symptom[i]):
            disease.append("mild_fever")
        elif("Yes I was" in symptom[i]):
            disease.append("silver_like_dusting") 
        elif("Regularly" in symptom[i]):
            disease.append("loss_of_balance")
        elif("I get chills" in symptom[i]):
            disease.append("chills")
        elif("I experience continuous sneezing" in symptom[i]):
            disease.append("continuous_sneezing")
        elif("I have a running nose" in symptom[i]):
            disease.append("runny_nose")
        elif("I shiver a lot" in symptom[i]):
            disease.append("shivering")
        elif("I have lost my sense of smell" in symptom[i]):
            disease.append("loss_of_smell")
        elif("I have sinus" in symptom[i]):
            disease.append("sinus_pressure")
        elif("My eyes continuously water" in symptom[i]):
            disease.append("watering_from_eyes")
        elif("I have a blocked nose" in symptom[i]):
            disease.append("congestion")
        elif("My hands and feet are cold" in symptom[i]):
            disease.append("cold_hands_and_feets")
        elif("I have sticky and stringy mucus" in symptom[i]):
            disease.append("phlegm")
        elif("I have irritation in my throat" in symptom[i]):
            disease.append("throat_irritation")
        elif("I think my thyroid has enlarged" in symptom[i]):
            disease.append("enlarged_thyroid")

        elif("I have blurred and disoted vision" in symptom[i]):
            disease.append("blurred_and_distorted_vision")
        elif("I have visual disturbances" in symptom[i]):
            disease.append("visual_disturbances")
        elif("My eyes have turned red in color" in symptom[i]):
            disease.append("redness_of_eyes")   

        elif("I am experiencing pain during bowel movements" in symptom[i]):
            disease.append("pain_during_bowel_movements")
        elif("I have pain in my joints" in symptom[i]):
            disease.append("joint_pain")
        elif("I have pain in my abdomen" in symptom[i]):
            disease.append("abdominal_pain")
        elif("I have pain in my stomach" in symptom[i]):
            disease.append("stomach_pain")
        elif("I have a muscle pain" in symptom[i]):
            disease.append("muscle_pain")
        elif("I have pain in my anal region" in symptom[i]):
            disease.append("pain_in_anal_region")
        elif("I have pain in my chest" in symptom[i]):
            disease.append("chest_pain")
        elif("I experience pain while walking" in symptom[i]):
            disease.append("painful_walking")
        elif("I am experiencing stifness in my movements" in symptom[i]):
            disease.append("movement_stiffness")
        elif("I have got a stiff neck" in symptom[i]):
            disease.append("stiff_neck")
        
        elif("I have a skin rash" in symptom[i]):
            disease.append("skin_rash")
        elif("I have skin erruptions" in symptom[i]):
            disease.append("nodal_skin_eruptions") 
        elif("My skin is peeling" in symptom[i]):
            disease.append("skin_peeling")
        elif("My skin has turned to yellow" in symptom[i]):
            disease.append("yellowish_skin")
        elif("I have blackheads" in symptom[i]):
            disease.append("blackheads")
        elif("I have got pus filled pimples" in symptom[i]):
            disease.append("pus_filled_pimples")
        elif("I have patches on my body" in symptom[i]):
            disease.append("dischromic_patches")
        elif("I have got a red sore around my nose" in symptom[i]):
            disease.append("red_sore_around_nose")
        elif("I have yellow pus filled blisters" in symptom[i]):
            disease.append("yellow_crust_ooze")
        elif("I have inflamation around my nails" in symptom[i]):
            disease.append("inflammatory_nails")
        elif("I have red spots over the body" in symptom[i]):
            disease.append("red_spots_over_body")
        elif("I have blisters" in symptom[i]):
            disease.append("blister")
        elif("I have small dents in my nails" in symptom[i]):
            disease.append("small_dents_in_nails")
        elif("I am experiencing scurring" in symptom[i]):
            disease.append("scurring")
        
        elif("I feel dizzi" in symptom[i]):
            disease.append("dizziness")
        elif("I have weakness on one side of the body" in symptom[i]):
            disease.append("weakness_of_one_body_side")
        elif("I have muscle weakness" in symptom[i]):
            disease.append("muscle_weakness")
        elif("I am expereincing fatigue" in symptom[i]):
            disease.append("fatigue")
        elif("I am feeling lethargic" in symptom[i]):
            disease.append("lethargy")
        elif("I am exeriencing malaise" in symptom[i]):
            disease.append("malaise")
      
        elif("I have an external itch" in symptom[i]):
            disease.append("itching")
        elif("I have an internal itch" in symptom[i]):
            disease.append("internal_itching")

        
    return disease






def getResponse(ints, intents_json):
    result_question=[]
    result_option=[] 
    result_type=[]
    result_tag=[]
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result_tag.append(tag)
            result = i['responses']
            for j in range(0,len(result)):
              result_question.append(result[j]['question'])
              result_option.append(result[j]['options'])
              result_type.append(result[j]['type'])
            break

    return result_question,result_option,result_tag,result_type
def chatbot_response(text):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("chatbot_model.h5")
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

global symptom_list
global tag_list    

@app.route("/")
def hello_world():
    question='Are You Sick?'
    option=["Yes","No"]
    radio=TRUE
    check=False
    return render_template("index.html",start=True,  radio=radio,check=check, question=question, len_option= len(option), option=option)

@app.route("/chat", methods=['GET','POST'])
def chat():
    button= request.form['input_response']  
    if button=="Yes":
        disease_list=pd.read_csv("static/Category.csv")
        #option=disease_list['Category'].tolist()
        option=disease_list['Category_option'].tolist()
        question="What are the symptoms?"
        radio=False
        check=True
        return jsonify({'option':option,'radio':True,'question':question})
        #return render_template("index.html",start=False,  radio=radio,check=check, question=question, len_option= len(option), option=option)
    if button=="No":
        user_msg="Lets start the chat"

        return jsonify({'user_msg':user_msg})
    return jsonify({'error':"This is an error mate"})
    


@app.route("/chat_disease", methods=['GET','POST'])
def Chat_disease():
    
    type_q= request.form['type']
    symptom_list=""
    if(type_q=="nothing"):
        
        symp=""
        disease= request.form['input_response2']
        print(disease)
        print(len(disease))
        symptom= disease.split(",")
        option=[]
        result_ques,result_opt,result_tag,result_type= chatbot_response(symptom[0])
        for i in range(0,len(result_opt)):
            for j in result_opt[i]:
                option.append(j)
        symptom.remove(symptom[0])
        for i in symptom:
            symp+=str(i)+","
        l=len(symp)
        symp_1 = symp[:l-1]
        
        return jsonify({'result_ques':result_ques[0][0],'symptom_list':"",'symptom':symp_1,'result_opt':option,'result_tag':result_tag[0],'type':result_type[0][0]})
    else:
       
        option_chosen= request.form['input_response2']
        tag_selected=request.form['type']
        symptom_list_resp=request.form['symptom_list']
        symptom_list+=symptom_list_resp+","+option_chosen
        symptom=request.form['symptom']
        bin=option_chosen.split(",")
        
        sym=symptom
        
        if(len(symptom)>0):
            symptom= sym.split(",")
            symp=""
            print(len(symptom))
            option=[]
            result_ques,result_opt,result_tag,result_type= chatbot_response(symptom[0])
            for i in range(0,len(result_opt)):
                for j in result_opt[i]:
                    option.append(j)
            symptom.remove(symptom[0])
            for i in symptom:
                symp+=str(i)+","
            l=len(symp)
            symp_1 = symp[:l-1]
            return jsonify({'result_ques':result_ques[0][0],'symptom_list':symptom_list,'symptom_list':symptom_list,'symptom':symp_1,'result_opt':option,'result_tag':result_tag[0],'type':result_type[0][0]})
        else:
            option_chosen= request.form['input_response2']
            tag_selected=request.form['type']
            symptom_list_resp=request.form['symptom_list']
            symptom_list_resp=symptom_list_resp+","+option_chosen
            symptom_list_resp=symptom_list_resp[1:]
            disease=get_disease(symptom_list_resp)
            print(disease)
            return jsonify({'disease':disease,'symp':symptom_list_resp})





@app.route("/aboutUs")
def AboutUs():
    return "<p>This Is About Us Page</p>"

@app.route("/login")
def Login():
    return "<p>This Is Login Page</p>"

@app.route("/register")
def Register():
    return "<p>This Is Register Page</p>"

if __name__ == '__main__':
   app.run(debug=True)