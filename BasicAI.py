#Huggingface imports
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util

#Other imports
import torch
import re
import os
import pandas as pd
import PyPDF2
from dotenv import load_dotenv

#Retrieve access token from environment variables
load_dotenv()
accessToken = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

class DataService:
    def __init__(self, datapath):
        self.datapath = datapath
    
    def readData(self):
        reader = PyPDF2.PdfReader(self.datapath)
        #Get the number of pages
        numPages = len(reader.pages)
        #Get text from each page for the amount of pages we've found
        dataset = ""
        for i in range(numPages):
            extractedText = reader.pages[i].extract_text()
            dataset += extractedText
        return dataset

class ConversationalAI:
    def __init__(self, data):
        try:
            #Load in the data from .csv file
            dataString = data
            print("Data loaded successfully")

            
            #Split the data into sentences using Regular Expressions (REGEX)
            self.sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', dataString)

            #Initialise the semantic search model
            print("Initialising Semantic Search")
            self.semanticSearchModel = SentenceTransformer('all-MiniLM-L6-v2')

            #Tensors are arrays which are used for calculations to find similarity.
            #Tokens are the word in the sentence, embeddings are the vectors, which are numerical representations of tokens, that are used to find similarity between sentences
            self.embeddings = self.semanticSearchModel.encode(self.sentences, convert_to_tensor=True)

            #Set the device to use, CPU or GPU
            self.device = self.setProcessingDevice()
            
            #Initialise the quantisation config. This reduces the model size to increase speed.
            quantisationConfig = BitsAndBytesConfig(load_in_4bit=True)
            print("Initialising quantisation config")

            #Initialise the model
            self.model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-1B",
                quantization_config=quantisationConfig
            )
            print("Model initialised and quantised")

            #Initialise tokeniser, which turns text into tokens
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
            print("Tokenizer initialised")
            
            #Initialise the textgen pipeline, setting params for model, access, tokeniser and device
            self.QAPipeline = pipeline(
                "text-generation",
                model=self.model,
                token=accessToken,
                tokenizer=self.tokenizer
                )
            print("Q&A Pipeline initialised")

        except Exception as e:
            print(f"Failed to initialise Conversational AI: {e}")
    
    #Set Processing Device method to set if we're going to use CPU or GPU. GPU is faster if available.
    def setProcessingDevice(self):
        if torch.cuda.is_available():
            logging.info("CUDA available. Using GPU for processing.")
            return torch.device('cuda')
        else:
            logging.info("CUDA is NOT available. Defaulting to CPU for processing.")
            return torch.device('cpu')
        
    #This method generates responses to user inputs
    def generateResponse(self, userInput, topKSetting=5, topP=0.9, temperature=0.7, maxLlamaTokens=40):
        #Encode the input into embeddings (tensors)
        queryEmbeddings = self.semanticSearchModel.encode(
            userInput,
            convert_to_tensor=True,
            batch_size=16
            ).to(self.device)
        
        #Find similarity (cosine) score between input and sentences
        scores = util.pytorch_cos_sim(queryEmbeddings, self.embeddings)[0]

        #Grab the top scoring similar sentences 
        topK = min(topKSetting, len(self.sentences))

        #Top k results - this has the top k scores and corresponding locations in the content (indices)
        topScoringResults = torch.topk(scores, k=topK) 

        topScoringSentences = [self.sentences[index] for index in topScoringResults.indices]

        #Join the sentences with a space delimiter
        context = " ".join(topScoringSentences)

        #Combine the user input with the context (our .csv data)
        combinedDataContext = f"{userInput} {context}"

        #Define the response with our input, dataset and parameters
        response = self.QAPipeline(
                            combinedDataContext,
                            max_new_tokens=maxLlamaTokens,
                            temperature=temperature,
                            top_p=topP,
                            truncation=True,
                            pad_token_id=self.QAPipeline.tokenizer.eos_token_id
                            )
        
        #Remove whitespace before the first letter of the response, for a cleaner output
        output = response[0]['generated_text'][len(userInput):].strip()
        
        #Invoke auto evaluation - getting the similarity between user input and the bot output
        try:
            self.AutoEvaluateResponse(userInput, output)
        except Exception as e:
            logging.error(f"Failed to autoevaluate response: {e}")
    
        return output
    
    def AutoEvaluateResponse(self, userInput, output):
        try:
            #Create encoded embeddings for the user input and output
            inputEmbeddings = self.semanticSearchModel.encode(userInput, convert_to_tensor=True)
            outputEmbeddings = self.semanticSearchModel.encode(output, convert_to_tensor=True)
            #Work out the cosine similarity to score it
            cosineSimilarity = util.pytorch_cos_sim(inputEmbeddings, outputEmbeddings).item() #item will get the tensor value

        except Exception as e:
            logging.error(f"Failed to autoevaluate response: {e}")
            return None
        
        return cosineSimilarity
    
def main():
    #Initialise data service
    dataService = DataService('c:\\Users\\Matt\\Downloads\\amazonannualreport.pdf')
    print("Data service initialised")
    data = dataService.readData()
    print("Data loaded successfully")

    #Initialise Conversational AI
    conversationalAI = ConversationalAI(data)
    print("Conversational AI object initialised")

    #Get the user input
    while True:
        userInput = input("Enter your message: ")
        #Generate a response
        response = conversationalAI.generateResponse(userInput)
        #Auto eval the response
        autoEvaluationScore = conversationalAI.AutoEvaluateResponse(userInput, response)
        #Print the bot response and the score
        print(f"""**************************************************
              ******************************************************
              ******************************************************
            YOUR QUESTION: {userInput}

            ****************************************************

              BOT RESPONSE: {response}
              ******************************************************
              ******************************************************
                ****************************************************""")
        print(f"The above response scored: {autoEvaluationScore}")

if __name__ == "__main__":
    main()