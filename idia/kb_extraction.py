
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from idia.config import *
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

        self.__directPrompt_template = """Given the following context, answer the question:
        context: {context}
        question: {question}
        """    
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
                    doc_chain = load_qa_chain(self.llm, chain_type="map_reduce", return_map_steps=True)
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
                        doc_chain = load_qa_chain(self.llm, chain_type="map_reduce", return_map_steps=True)
                        sub_docs = self.indexData(str(i), self.ind, doc_chain)
                    except Exception as e:
                        logging.warning(e)


    #
    # Load the data from the stream
    #
    def __loadData(d):
        loader = UnstructuredFileLoader(file)
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
        "question": """Extract from the curriculum surname, name, date of birth, quality of resume writing style, list of working experiences. Report the information in a well-formatted JSON, as in the example: 
        {{"name":"name", "surname": "surname", "dateOfBirth":"date of birth", "quality": "the rank ranges from 0=poor to 10=excellent", "experiences": [{{"years": "number of years of working experience with the company", "start": "first year with the company", "end": "last year with the company", "company": "company"}}]}}
        
        Remember that the current year is 2023 and a University is not a company. Remember to always assign a quality rate to the curriculum. Provide only the JSON, nothin else.
        """})     

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

        for t in texts:
            header =f"Candidato: {nominativo} data di nascita: {dtn} qualità del curriculum: {quality} out of 10 numero di pagine del curriculum: {pages} anni di esperienza: {years}"
            cv = Document(page_content=header+"\n"+t.page_content, metadata=metadata)
            self.texts.append(cv)

        ind.upsert(texts)    
    
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
            qPrompt = """Use the following portion of a long curriculum vitae to see if any of the text is relevant or partially relevant to answer the question or are relevant to at least one candidate mentioned in the question, and summarize it. 
If it is not so, return "None". Report the following metadata in you answer: surname, name, date of Birth, years of experience, and the quality of the curriculum. 

{context}
Question: {question}
Relevant text, if any, in Italian:"""

            sPrompt = """Given the following extracted parts of a long document and a question, create a final answer. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.

Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.

Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
=========
FINAL ANSWER: This Agreement is governed by English law.

QUESTION: What did the president say about Michael Jackson?
=========
Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.

Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.

Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.

Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
=========
FINAL ANSWER: The president did not mention Michael Jackson.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

            cPrompt = """Please read through all of the extracted parts of the curricula vitae carefully before answering the question in Italian. Make sure to only use information from the extracts to answer the question. Let's work through this step by step.

        QUESTION: {question}
        
        =========
        {summaries}
        =========
        Answer in Italian:"""
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