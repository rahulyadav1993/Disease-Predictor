
from flask import Flask, request, render_template
import azure.cognitiveservices.speech as speechsdk
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from textblob import TextBlob
import re



app = Flask(__name__)
API_Key='e439ab8e759d4479947bf1d53b9276c1'
End_point='https://eastus.api.cognitive.microsoft.com/sts/v1.0/issuetoken'

model_imp = pickle.load(open('disease_prediction.pkl', 'rb'))

text=''
@app.route('/')
def home():
    return render_template('index.html')





@app.route('/recognize_from_microphone',methods=['POST'])

def recognize_from_microphone():
    speech_config = speechsdk.SpeechConfig(subscription=API_Key, endpoint=End_point)
    speech_config.speech_recognition_language="en-US"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    

    print("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()
    
    

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
        text="Recognized: "+speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
        text="No speech could be recognized: Kindly specify your Symptoms "
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        text="Speech Recognition canceled"
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
            text="Did you set the speech resource key and region values?: please check"
    return render_template('index.html', predict_text=text)


## Tokenization, nomalization and removing punctuation, stopwords function

def tokenize(text):
    
    textBlb = TextBlob(text)
    textCorrected = textBlb.correct()   # Correcting the text
    text=str(textCorrected)
    
    waste_words=['ah','uh','hmm','oh','uhh','ahh','ohhh','ohh','shhh','mm','mmm','hmmm','hm','yeah','ya','yup','nope','im','naw','noo','yess']
    stopword = set(stopwords.words('english'))
    text=re.sub(r'[^\w\s]','', text) #Remove Punctuation
    text=text.lower() #Lower the text

    #tokenize text
    tokens = word_tokenize(text)
    
    tokens=[word for word in tokens if word not in stopword] #Remove stopwords from the sentences
    tokens=[word for word in tokens if word not in waste_words] #Remove some grabage words from the sentences like Ah,oh etc
     
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    

    # iterate through each token
    clean_tokens =''
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok,pos='v').strip()
        #print(clean_tok)
        clean_tokens=clean_tokens+clean_tok+' '

    return clean_tokens

@app.route('/predict',methods=['GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    try:
        region='eastus'

        speech_config = speechsdk.SpeechConfig(subscription=API_Key, region=region)
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        
        # The language of the voice that speaks.
        speech_config.speech_synthesis_voice_name='en-IN-PrabhatNeural'
        
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        text1=request.args.get("p")
        text2=request.args.get("q")
        print(text2)
        if text1=="" and text2=="" :
             text="Please tell or type your symptoms"
             speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
             
            
        else:    
            text=text1 or text2
            print(text)
            text=tokenize(text)
            print(text)
           
            
            prediction = model_imp.predict([text])
            print(prediction)
            
            #######################################
            

            # Get text from the console and synthesize to the default speaker.
            if prediction[0]=="Some Disease":
                text="We cannot predict a Disease"
            else:
        
                text="You may have: "+ prediction[0]
            
            speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
            
            if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print("Speech synthesized for text [{}]".format(text))
            elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = speech_synthesis_result.cancellation_details
                print("Speech synthesis canceled: {}".format(cancellation_details.reason))
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    if cancellation_details.error_details:
                        print("Error details: {}".format(cancellation_details.error_details))
                    print("Did you set the speech resource key and region values?")
    

        return render_template('index.html',prediction_text=text)
    
    except:
        error_string = "Kindly tell us your disease again"
        return render_template('index_new.html', prediction_text=error_string)


if __name__ == "__main__":
    app.run(debug=False)