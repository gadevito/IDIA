import os
import time
import json
from c4se.config import *
from langchain.chains.llm import LLMChain
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from c4se.agents.codeSnippet import CodeSnippet
from c4se.agents.github import AutomaticTaskExecutor
from c4se.agents.testCases import TestCasesProducer
from c4se.prompts import *
from typing import Dict
import logging
from pathlib import Path
from c4se.memory.chat_memory import ChatMemory
#from c4se.vector_memory import VectorMemory, RedisVectorChatMessageHistory
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.vectorstores import Pinecone
import pinecone
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#
# Thie is the BL Manager component
#
class DC4SEChatBot():

    def __init__(self, sessionID: str = "my-session"):
        #url = f"redis://localhost:6379/0"
        #self.message_history = RedisChatMessageHistory(url=url, ttl=600, session_id=sessionID)
        #self.message_history = RedisVectorChatMessageHistory(withScore=True)
        #self.message_history.ttl = 6000
        #self.memory = VectorMemory(memory_key="history", chat_memory=self.message_history)
        model_name="gpt-3.5-turbo"
        #model_name = "gpt-3.5-turbo-0301"
        #self.llm= ChatOpenAI(openai_api_key = OPENAI_API_KEY,model_name=model_name, streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, verbose=True)
        self.llm= ChatOpenAI(openai_api_key = OPENAI_API_KEY,model_name=model_name, temperature=0, verbose=True)
        self.memory = ChatMemory(memory_key="chat_history")
        self.c_chain= None
        self.retriever = None
        self.doc_chain = None
        self.texts = []
        self.db = None
        self.docs =[]
        self.loadDocuments()

        #self.memory.clear()
        

    def get_chat_history(self, history):
        return history

    def loadDocuments(self):
        #loader = DirectoryLoader(path="./data",silent_errors=True)
        #docs = loader.load_and_split()
        #text_splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
        #self.texts = text_splitter.split_documents(docs)
        #print("DIM:", len(self.texts))
        
        isNew = False
        try:
            self.db = FAISS.load_local("./index", embeddings=OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY), index_name="cvs")
        except Exception as es:
            print(es)
            isNew = True
            self.loadDocuments1()
            self.db = FAISS.from_documents(self.texts, OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY))
            self.db.save_local("./index", "cvs")
        

        self.retriever = self.db.as_retriever()
        #self.retriever.search_kwargs['distance_metric'] = 'cos'
        self.retriever.search_kwargs['fetch_k'] = 20
        #self.retriever.search_kwargs['maximal_marginal_relevance'] = True
        self.retriever.search_kwargs['k'] = 15 #20
        self.retriever.search_type = "mmr" #"similarity_score_threshold" #"mmr" #"similarity"
        self.retriever.search_kwargs['lambda_mult'] = 0.2
        #similarity_score_threshold mmr
        # nei search args score_threshold
        print("PINECONE CREATO")

        question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
        Return any relevant text translated into italian. If there is no relevant text to the question, return "None".
        {context}
        Question: {question}
        Relevant text, if any, in Italian:"""

        question_prompt_template = """Use the following portion of a long curriculum vitae to see if any of the text is relevant or partially relevant to answer the question or are relevant to at least one candidate mentioned in the question, and summarize it. 
If it is not so, return "None". Report the following metadata in you answer: surname, name, date of Birth, years of experience, and the quality of the curriculum. 

        {context}
        Question: {question}
        Relevant text, if any, in Italian:"""




        QUESTION_PROMPT = PromptTemplate(
            template=question_prompt_template, input_variables=["context", "question"]
        )

        combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer in Italian. 
        Please ignore the parts that don't have helpful information to answer and use only the valuable extracts to create the final answer.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.

        QUESTION: {question}
        =========
        {summaries}
        =========
        Answer in Italian:"""

        combine_prompt_template = """Please read through all of the extracted parts of the curricula vitae carefully before answering the question in Italian. Make sure to only use information from the extracts to answer the question. Let's work through this step by step.

        QUESTION: {question}
        
        =========
        {summaries}
        =========
        Answer in Italian:"""

        COMBINE_PROMPT = PromptTemplate(
            template=combine_prompt_template, input_variables=["summaries", "question"]
        )


        self.doc_chain = load_qa_chain(self.llm, chain_type="map_reduce", return_map_steps=True, question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)
        #self.c_chain = ConversationalRetrievalChain(retriever=self.retriever,question_generator=question_generator, combine_docs_chain=self.doc_chain)
        #self.c_chain.get_chat_history=self.get_chat_history 
        
        
        print("RETRIEVER CREATO")

    def loadDocuments1(self):
        for dirpath, dirnames, filenames in os.walk("./data"):
            for file in filenames:
                try: 
                    self.metadataExtractor(os.path.join(dirpath, file))
                except Exception as e: 
                    pass

    def metadataExtractor(self, file):
        from langchain.document_loaders import UnstructuredFileLoader
        from langchain.docstore.document import Document
        loader = UnstructuredFileLoader(file)
        doc_chain = load_qa_chain(self.llm, chain_type="map_reduce", return_map_steps=True)
        # surname, name, date of birth,
        doc = loader.load()
        print(doc)
        
        text_splitter = CharacterTextSplitter(separator=" ", chunk_size=3000, chunk_overlap=0) #2000      
        texts = text_splitter.split_documents(doc)

        pages = 0
        words = 0
        for i in texts:
            l = len(i.page_content.split())
            words = words + l
        pages = round(words/383)

        print("chunks: ",len(texts)," words: ", words, " pages:", pages)

        #print(texts)
        res = doc_chain({"input_documents": texts, 
        "question": """Extract from the curriculum surname, name, date of birth, quality of resume writing style, list of working experiences. Report the information in a well-formatted JSON, as in the example: 
        {{"name":"name", "surname": "surname", "dateOfBirth":"date of birth", "quality": "the rank ranges from 0=poor to 10=excellent", "experiences": [{{"years": "number of years of working experience with the company", "start": "first year with the company", "end": "last year with the company", "company": "company"}}]}}
        
        Remember that the current year is 2023 and a University is not a company. Remember to always assign a quality rate to the curriculum. Provide only the JSON, nothin else.
        """})     

        #print("******** OUTPUT ====>")
        #print(res["output_text"],"\n")
        #print("******* INTERMEDIATE STEPS")
        #print(res["intermediate_steps"])
        metadata = {"source": file, "name": "", "surname": "", "dateOfBirth": "", "yearsOfExperience": 0, "quality": 0, "pages":pages}
        start = True
        years = 0
        min = 10000
        max = 0
        for step in res["intermediate_steps"]:
            print(step)
            try:
                j = json.loads(step)
                #print(j)
                if start:
                    metadata["name"] = j["name"]
                    metadata["surname"] = j["surname"]
                    metadata["dateOfBirth"] = j["dateOfBirth"]
                    metadata["quality"] = j["quality"]
                    start = False
                for exp in j["experiences"]:
                    print(exp["start"], " ", exp["end"], exp["company"])
                    try:
                        dst = exp["start"]
                        print(dst)
                        if( len(dst) >4):
                            s = int(dst[-4:])
                        else:    
                            s = int(dst)
                        dend = exp["end"]
                        print(dend)
                        if(dend =="present"):
                            dend = "2023"
                        if(len(dend)>4):
                            e = int(dend[-4:])
                        else:
                            e = int(dend)
                        if( s < min):
                            min = s
                        if e > max:
                            max = e        
                        print("attuale: ",min, " ", max, "start: ",s, " end: ",e)
                    except Exception as x:
                        print(x)
            except Exception as ej:
                print(ej)
        years = max - min
        if( years <=0):
            years = 1
        metadata["yearsOfExperience"] = years
        print(metadata)
        nominativo = metadata["surname"]+" "+metadata["name"]
        dtn = metadata["dateOfBirth"]
        quality = metadata["quality"]
        for t in texts:
            header =f"Candidato: {nominativo} data di nascita: {dtn} qualità del curriculum: {quality} out of 10 numero di pagine del curriculum: {pages} anni di esperienza: {years}"
            cv = Document(page_content=header+"\n"+t.page_content, metadata=metadata)
            self.texts.append(cv)
    #
    # Answer a specific question (user_input) of the user
    #
    def answer(self, user:str, user_input: str)->Dict:
        res =""
        markdown = True
        start = time.time()
        try:
            dcx = self.retriever.get_relevant_documents(user_input) #, include_metadata=True)
            print(len(dcx),"\n",dcx,"\n\n")

 

            rx = self.doc_chain({"input_documents": dcx, "question": user_input}, return_only_outputs=True)
            print(rx)
            res = rx["output_text"]
            
            #chat_history = self.memory.load_memory_variables({"input":user_input,"user":user})["chat_history"]
            #print("History", chat_history)
            #resx = self.c_chain({"question":user_input, "chat_history":chat_history})
            #res = resx["answer"]
            #print("RES ",resx)
            logging.debug(f"CLASSIFICATION {res}")
        except Exception as e:
            print(e)
            raise e
        #self.memory._save_chat(user, user_input, res)
        #self.message_history.add_user_message(user_input)
        #self.message_history.add_ai_message(res)
        end = time.time()
        print("TEMPO: ", end-start)
        return {"markdown": markdown, "response": res}

    #
    # Utility function to check if a file is visible in the file system
    #
    def _is_visible(self, p: Path) -> bool:
        parts = p.parts
        for _p in parts:
            if _p.startswith("."):
                return False
        return True

    #
    # Upsert a document into the dynamic memory
    #
    def upsertDocument(self, path, author):
        self.memory.upsertDocument(path, author)

    #
    # Upsert a complete directory into the dynamic memory
    #
    def upsertDirectory(self, path, author):
        p = Path(path)
        glob: str = "**/[!.]*"
        docs = []
        items = list(p.rglob(glob))
        for i in items:
            if i.is_file():
                if self._is_visible(i.relative_to(p)):
                    try:
                        sub_docs = self.upsertDocument(str(i),author)
                        docs.extend(sub_docs)
                    except Exception as e:
                        logging.warning(e)

    def addCurriculum(self, path):
        self.texts=[]
        self.metadataExtractor(path)
    
        t = [d.page_content for d in self.texts]
        m = [d.metadata for d in self.texts]
        self.db.add_texts(t,m)
        self.db.save_local("./index", "cvs")

    def addCurricula(self, path):
        p = Path(path)
        glob: str = "**/[!.]*"
        items = list(p.rglob(glob))
        for i in items:
            if i.is_file():
                if self._is_visible(i.relative_to(p)):
                    try:
                        sub_docs = self.addCurriculum(str(i))
                    except Exception as e:
                        logging.warning(e)

def main():
    bot = DC4SEChatBot()

    print(bot.answer("hi"))

    print(bot.answer("""
    Write a python snippet to parse the list of expenses and return the list of triples (date, value, currency).
    Ignore lines starting with #.
    Parse the date using datetime.
    Example expenses_string:
        2016-01-02 -34.01 USD
        2016-01-03 2.59 DKK
        2016-01-03 -2.72 EUR
    """))


if __name__ == "__main__":
    main()