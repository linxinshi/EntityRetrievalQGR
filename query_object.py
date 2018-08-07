# coding=utf-8
import sys
import string
from lib_process import cleanSentence, remove_stopwords, convStr2Vec,stemSentence
from list_term_object import List_Term_Object
from config import *
from document_object import Document_Object
from nltk.util import ngrams

class Query_Object(Document_Object):
      contents_obj=None
      subqueries=None
      bigrams=None
      query_entities=None
      query_predicates=None
      numeric_data=None
      aggregation_node=None
      
      def __init__(self,query,structure,lucene_handler,debug_mode=False):
          mongoObj,w2vmodel=structure.mongoObj,structure.w2vmodel
          self.dict_attr={}
          # query: query_id, clusterd query, raw query
          if debug_mode==False:
             self.setAttr('id',query[0].strip())
             self.setAttr('clustered_query',query[1].strip().lower())
             if IS_STOPWORD_REMOVED:
                qstr=remove_stopwords(cleanSentence(query[2].strip(),True,' '),' ')
             self.setAttr('raw_query',qstr)
             self.setAttr('original_query',query[2].strip())
          else:
             self.setAttr('id','ID_DEBUG_QUERY')
             self.setAttr('raw_query',query)
             
          self.setAttr('stemmed_raw_query',stemSentence(self.raw_query,None,True))
          self.setAttr('querystr',self.dict_attr[USED_QUERY_VERSION])
          self.setAttr('queryID',self.id)
          
          self.contents_obj=List_Term_Object(self.querystr,True,' ',mongoObj,w2vmodel)
          self.update_bigrams()
          if IS_QUERY_GRAPH_USED==True or IS_ELR_USED==True:
             self.update_query_entities(mongoObj)
             self.update_query_predicates(mongoObj)
             self.update_query_numeric_data()
             self.update_query_aggregation()  
             
             '''
             print (self.original_query)
             if self.numeric_data is not None:
                print (self.numeric_data)  
             if self.aggregation_node is not None:
                print (self.aggregation_node)
             '''                
             #print (self.query_entities)
             
             #print (self.query_predicates)
          if IS_SUBQUERY_USED==True:
             self.update_subqueries(mongoObj,w2vmodel)
      
      def update_query_numeric_data(self):
          self.numeric_data=[]
          for term in self.contents_obj.term:
              try:
                 data=int(term)
                 self.numeric_data.append(data) 
              except:
                 continue
      def update_query_aggregation(self):
          for term in self.original_query.split():
              if term in ['biggest','latest','heaviest','largest','highest','longest']:
                 self.aggregation_node='max'
              elif term in ['smallest','lowest','earliest']:
                 self.aggregation_node='min'
              elif term in ['since']:
                 self.aggregation_node='>'
              elif term in ['before']:
                 self.aggregation_node='<'
              elif term in ['between']:
                 self.aggregation_node='range'
          if self.original_query.find('more than')>-1:
             self.aggregation_node='>'
          if self.original_query.find('less than')>-1:
             self.aggregation_node='<'
          
              
      def update_query_entities(self,mongoObj):
          self.query_entities={}
          list_item=mongoObj.conn_qe.find({'qid':self.id})
          if list_item is None:
             return
          for item in list_item:
              self.query_entities[item['title']]=item['score']
              #print (self.id,item['title'])  
              
      def update_query_predicates(self,mongoObj):
          self.query_predicates={}
          list_item=mongoObj.conn_qp.find({'qid':self.id})
          if list_item is None:
             return
          for item in list_item:
              self.query_predicates[item['pred']]=item['score']
          #print (self.query_predicates)   
      
      def update_subqueries(self,mongoObj,w2vmodel):
          '''
          sub query str seperated by '|'
          now queryTerms and qt_final is a list of each sub query word list
          in other word,  [[],[],[]]
          '''
          self.subqueries=[]
          list_subquery=self.clustered_query.split(SEPERATE_CHAR_SUBQUERY)
          
          len_subqueries=len(list_subquery)
          for i in range(len_subqueries):
              #item=stemSentence(list_subquery[i],None,False)
              item=list_subquery[i]
              self.subqueries.append(List_Term_Object(item,True,' ',mongoObj,w2vmodel))

      def update_bigrams(self):
          self.bigrams = list(set(ngrams(self.querystr.split(),2)))
          #print (str(self.bigrams))
                        
        