
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from idia.config import *
from idia.prompts import *
import logging
from concurrent.futures import *
from pathlib import Path
import os
import json
import time
from typing import Dict

MAX_RELEVANT_RESULTS = 15
MAX_CONTEXT_SIZE = 3000
MAX_THREADS = 30

#
# Vector Index Wrapper
#
class VectorIndex:

    def __init__(self):
        self.db = None
        self.retriever = None
    
    #
    # Load the vector index, if it exists
    #
    def load(self):
        isNew = True

        try:
            self.db = FAISS.load_local("./index", embeddings=OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY), index_name="cvs")
        except Exception as es:
            isNew = False

        return isNew

    #
    # Upsert data into the Vector Index
    #
    def upsert(self, texts):
        if self.db is None:
            self.db = FAISS.from_documents(texts, OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY))
        else:
            t = [d.page_content for d in texts]
            m = [d.metadata for d in texts]
            self.db.add_texts(t,m)
        self.db.save_local("./index", "cvs")

    #
    # Crate the Retriever to use to get relevant data
    #
    def createRetriever(self):
        if self.db is not None:
            self.retriever = self.db.as_retriever()

    #
    # search relevant data
    #
    def searchRelevantData(self,q,num):
        if self.retriever is None:
            self.createRetriever()
        self.retriever.search_kwargs['fetch_k'] = num + 5
        self.retriever.search_kwargs['k'] = num 
        self.retriever.search_type = "mmr" 
        self.retriever.search_kwargs['lambda_mult'] = 0.2
        return self.retriever.get_relevant_documents(q) 

#
# KnowledgeExtractor implementation
#
class KnowledgeExtractor:

    #
    # Initialization
    #
    def __init__(self):
        #TODO

        self.__directPrompt_template = DIRECT_KNOWLEDGE_EXTRACTION_PROMPT    
        self.ind = VectorIndex()
        loaded = self.ind.load()
        # if the vector index doesn't exist, try to load documents from the "./data" directory
        if not loaded:
            self.__loadDocumentsFromPath()

        self.ind.createRetriever()
    
        # Create the LLM list
        self.LLMs = [OpenAI(openai_api_key = OPENAI_API_KEY,model_name="gpt-3.5-turbo", temperature=0, verbose=True)]
    
    #
    # Load documents from the default path
    #
    def __loadDocumentsFromPath(self):
        for dirpath, dirnames, filenames in os.walk("./data"):
            for file in filenames:
                try: 
                    co = ChatOpenAI(openai_api_key = OPENAI_API_KEY,model_name="gpt-3.5-turbo", temperature=0, verbose=True)
                    doc_chain = load_qa_chain(co, chain_type="map_reduce", return_map_steps=True)
                    self.indexData(os.path.join(dirpath, file), self.ind, doc_chain)
                except Exception as e: 
                    pass    

    #
    # Add all the curricula from a specific path
    #
    def addCurricula(self, path):
        p = Path(path)
        glob: str = "**/[!.]*"
        items = list(p.rglob(glob))
        for i in items:
            if i.is_file():
                if self._is_visible(i.relative_to(p)):
                    try:
                        co = ChatOpenAI(openai_api_key = OPENAI_API_KEY,model_name="gpt-3.5-turbo", temperature=0, verbose=True)
                        doc_chain = load_qa_chain(co, chain_type="map_reduce", return_map_steps=True)
                        sub_docs = self.indexData(str(i), self.ind, doc_chain)
                    except Exception as e:
                        logging.warning(e)


    def addCurriculum(self,path):
        self.texts = []
        co = ChatOpenAI(openai_api_key = OPENAI_API_KEY,model_name="gpt-3.5-turbo", temperature=0, verbose=True)
        doc_chain = load_qa_chain(co, chain_type="map_reduce", return_map_steps=True)
        sub_docs = self.indexData(path, self.ind, doc_chain)
    #
    # Load the data from the stream
    #
    def __loadData(self,d):
        loader = UnstructuredFileLoader(d)
        return loader.load()

    #
    # Split the document list into chunks
    #
    def __split(self, d,chunkSize):
        text_splitter = CharacterTextSplitter(separator=" ", chunk_size=chunkSize, chunk_overlap=0)     
        texts = text_splitter.split_documents(d)
        return texts

    #
    # Get the metadata of a specific curriculum
    #
    def getMetadata(self,doc, LLM):
        pages = 0
        words = 0
        for i in doc:
            l = len(i.page_content.split())
            words = words + l
        pages = round(words/383)

        res = LLM({"input_documents": doc, 
        "question": METADATA_EXTRACTOR_PROMPT})     

        metadata = { "name": "", "surname": "", "dateOfBirth": "", "yearsOfExperience": 0, "quality": 0, "pages":pages}
        start = True
        years = 0
        min = 10000
        max = 0
        for step in res["intermediate_steps"]:
            print(step)
            try:
                j = json.loads(step)

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
        return metadata

    #
    # IndexData algorithm implementation
    #
    def indexData(self, d, ind, LLMs):

        doc = self.__loadData(d)
        texts = self.__split(doc, 2000)
        
        
        metadata = self.getMetadata(texts,LLMs)
        metadata["source"] = d
        nominativo = metadata["surname"]+" "+metadata["name"]
        dtn = metadata["dateOfBirth"]
        quality = metadata["quality"]
        pages = metadata["pages"]
        years = metadata["yearsOfExperience"]
        
        for t in texts:
            header =f"Candidato: {nominativo} data di nascita: {dtn} qualità del curriculum: {quality} out of 10 numero di pagine del curriculum: {pages} anni di esperienza: {years}"
            cv = Document(page_content=header+"\n"+t.page_content, metadata=metadata)
            self.texts.append(cv)

        ind.upsert(self.texts)    
    
    #
    # MRExtractKnowledge algorithm implementation
    #
    def MRExtractKnowledge(self, qPrompt, sPrompt, cPrompt, q, ind, LLMs):
        num = MAX_RELEVANT_RESULTS
        d = ind.searchRelevantData(q, num)
        if( self.__size(d) > MAX_CONTEXT_SIZE):
            docs = self.__ExctractValuableKnowledge(qPrompt, sPrompt, d, q, LLMs)
            answer = self.__Combine(docs, cPrompt, q, LLMs)
            return answer
        else:
            return self.__DirectGetKnowledge(q,d,self.select(LLMs))

    #
    # Calculate the number of character contained in the document list
    #
    def __size(self,d):
        dim = 0
        for i in d:
            dim = dim + len(i.page_content)
        
        return dim

    #
    # Implementation of ExctractValuableKnowledge algorithm
    #
    def __ExctractValuableKnowledge(self,qPrompt, sPrompt, d, q, LLMs):
        docs = []
        dim = 0
        executor = ThreadPoolExecutor(MAX_THREADS)
        numDocs = len(d)
        for i in range(1,numDocs):
            context= d[i]
            prompt = qPrompt.format(context=context, question=q)
            future = executor.submit(self.select(LLMs).predict, prompt)
            result = future.result()
            if self.__isAppropriate(result):
                dim = dim + len(result)
                docs.append(result)
        
        print(dim, " LEN ", numDocs)
        if dim > MAX_CONTEXT_SIZE:
            docs = self.__SummarizeKnowledge(sPrompt, docs, q, LLMs)
        
        return docs

    #
    # Verify if the LLM answer is acceptable or must be avoided
    #
    def __isAppropriate(self,r):
        return True

    #
    # SummarizeKnowledge algorithm implementation
    #
    def __SummarizeKnowledge(self, sPrompt, d, q, LLMs):
        docs = []
        executor = ThreadPoolExecutor(MAX_THREADS)
        numDocs = len(d)
        dim = 0
        i = 1
        first = 1
        totalDim = 0
        print("I=",i," NUMDOCS ",numDocs)
        while( i < numDocs):
            if (dim + len(d[i]) > MAX_CONTEXT_SIZE) or i+1>=numDocs:
                prompt = "content: "
                
                for j in range(first,i-1):
                    prompt = "Content: " + d[j]+"\n"
                prompt = sPrompt.format(summaries=prompt, question=q)
                print(first, " i = ", i, " DIM ", dim, "NUM ", numDocs)
                future = executor.submit(self.select(LLMs).predict, prompt)
                result = future.result()
                docs.append(result)
                first = i
                if( i+1>= numDocs):
                    break
                i = i -1
                totalDim = totalDim + dim
                dim = 0
            else:
                dim = dim+len(d[i])
                i = i+1


        print("TOTAL DIM ", totalDim, " ", MAX_CONTEXT_SIZE, "DOC ", len(docs))
        if totalDim > MAX_CONTEXT_SIZE:
            docs = self.__SummarizeKnowledge(sPrompt, docs, q, LLMs)
        
        return docs

    #
    # Combine algorithm implementation
    #
    def __Combine(self, docs, cPrompt, q, LLMs):
        prompt =""
        for i in range(1,len(docs)):
            prompt = prompt + docs[i]+"\n"
        prompt = cPrompt.format(summaries=prompt, question= q)
        
        answer = self.select(LLMs).predict(prompt)
        return answer

    #
    # DirectGetKnowledge algorithm implementation
    #
    def __DirectGetKnowledge(self,q,d,LLM):
        context =""
        for i in d:
            context = context+d.page+"\n"
        prompt = self.____directPrompt_template.format(context,question=q)
        
        answer = LLM.predict(prompt)
        return answer

    #
    # Select the most appropriate LLM
    #
    def select(self, LLMs):
        return self.LLMs[0]

    #
    # Answer a specific question to extract the knowledge
    #
    def answer(self, user:str, user_input: str)->Dict:
        res =""
        markdown = True
        start = time.time()
        try:
            qPrompt = KNOWLEDGE_EXTRACTION_PROMPT

            sPrompt = SUMMARIZE_PROMPT

            cPrompt = COMBINE_PROMPT
            res = self.MRExtractKnowledge(qPrompt,sPrompt,cPrompt,user_input,self.ind, self.LLMs)
            logging.debug(f"RESULT: {res}")
        except Exception as e:
            print(e)
            raise e

        end = time.time()
        logging.debug("TEMPO: ", end-start)
        return {"markdown": markdown, "response": res}

def main():
    bot = KnowledgeExtractor()

    start = time.time()
    print(bot.answer("Gabriele", "Fammi un sommario del curriculum di Schiavottiello"))
    end = time.time()
    print("TIME: ", end-start)

    start = time.time()
    print(bot.answer("Gabriele", "Che domanda faresti in un ipotetico colloquio a Schiavottiello?"))

    end = time.time()
    print("TIME: ", end-start)

if __name__ == "__main__":
    main()
