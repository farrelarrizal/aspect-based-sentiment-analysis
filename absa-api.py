import uvicorn
from fastapi import FastAPI, Request
from pyabsa import AspectPolarityClassification as APC
from pyabsa import AspectTermExtraction as ATE
import json
import yaml


class ABSAAPI:
    
    

    def __init__(self):
        self.config = yaml.safe_load(open('config.yaml')) 

        self.app = FastAPI()
        self.lang_support = self.config['language-support']
        
        self.aspect_extraction_en = self.load_ate_model(self.config['checkpoint']['ate']['en'])
        self.aspect_extraction_id = self.load_ate_model(self.config['checkpoint']['ate']['id'])
        
        self.sentiment_classifier_en = self.load_apc_model(self.config['checkpoint']['apc']['en'])
        self.sentiment_classifier_id = self.load_apc_model(self.config['checkpoint']['apc']['id'])
        


        
        @self.app.get("/")
        async def root():
            response = {
                "status": 200,
                "message": "Your PyABSA API is UP!!"
            }
            return response

        # Route for Aspect Based Sentiment Analysis with name predict
        @self.app.post("/predict")
        async def predict(request: Request):
            data = await request.json()
            lang = data['lang']
            aspect_tag = data['aspect_tag']
            
            if aspect_tag == False:
                # extract aspect
                extracted_aspects = self.extract_aspect(text=data['text'], lang=lang)

                # get inference format
                extracted_aspects = self.get_inference_format(extracted_aspects)
            elif aspect_tag == True:
                print('have aspect tag')
                extracted_aspects = data['text']
            # predict sentiment
            
            predicted_sentiment = self.predict_sentiment(text=extracted_aspects, lang=lang)

            return predicted_sentiment
            
        
        @self.app.get("/check")
        async def check():
            message = {
                "status":200
            }
            return message

            
    def load_apc_model(self, checkpoint):
        return APC.SentimentClassifier(
            checkpoint=checkpoint
        )
    
    def load_ate_model(self, checkpoint):
        return ATE.AspectExtractor(
            checkpoint=checkpoint
        )
    
    def extract_aspect(self, text, lang):   
        if lang == 'en':
            aspect = self.aspect_extraction_en.predict(text=text, pred_sentiment=False, save_result=False)
        elif lang == 'id':
            aspect = self.aspect_extraction_id.predict(text=text, pred_sentiment=False, save_result=False)
        else:
            raise Exception('language not supported')
        
        return aspect
    
    def get_inference_format(self, extracted_aspect):
        text = extracted_aspect
        # find the index
        idxs = []
        IOB_LIST = text['IOB']
        for i, tag in enumerate(IOB_LIST):
            if tag == 'B-ASP':
                idxs.append(i)

        # add the tag
        for idx in idxs:
            token = text['tokens'][idx]
            new_token = f'[B-ASP]{token}[E-ASP]'
            text['tokens'][idx] = new_token
        # convert to string
        sentence = " ".join(text['tokens'])
        return sentence

    
    def predict_sentiment(self, text, lang):
        try:
            if lang == 'en':
                prediction = self.sentiment_classifier_en.predict(
                            text=text,
                            ignore_error=True
            )
            elif lang == 'id':
                prediction = self.sentiment_classifier_id.predict(
                                text=text,
                                ignore_error=True
                )
            print(prediction)
            if prediction:
                response = {
                    "status_code": 200,
                    "message": "success",
                    "text": prediction['text'],
                    "aspect": prediction['aspect'],
                    "sentiment": prediction['sentiment'],
                    "confidence": prediction['confidence']
                }
                return response
            else:
                raise Exception('sorry, we are cannot predicting your text :( ')
        except Exception as e:
            response = {
                "status": 204,
                "error_message": str(e) 
            }
            return response


    def run(self):
        
        if __name__ == "__main__":
            uvicorn.run(self.app, host="0.0.0.0", port=8080)
            

api = ABSAAPI()
api.run()
