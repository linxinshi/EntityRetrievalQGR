import sys,datetime 
from nltk.stem.snowball import SnowballStemmer
from lib_process import *
from list_term_object import List_Term_Object
from config import *
from doc_tree_object import Doc_Tree_Object

from document_object import Document_Object
from nltk.util import ngrams

class Entity_Object(Document_Object):
      #__slots__=('id','title','name','raw_name','value','category','abstract','all_text_obj','attribute_obj','category_obj','name_obj','abstract_obj','en_ens','en_vector','mentioned_entity','rel_vec')    
      categories=None
      dict_obj=None
      factoid_obj_list=None
      bigrams=None
      
      mentioned_entity=None
      
      wiki_doc_tree=None
      term_freq=None
      term_freqs=None
      lengths=None
      
      literal_properties=None
      
      def __init__(self):
          self.dict_obj={}
          self.dict_attr={}
          
      def updateFromIndex(self,d_pair,mongoObj,w2vmodel,lucene_obj):
          # d_pair:(document,docid) entity: dict   
          entity,docid=d_pair[0],d_pair[1]
          for idf in entity.iterator():
              self.setAttr(idf.name(),idf.stringValue())
              #print ('%s\t%s'%(idf.name(),idf.stringValue()))
          self.setAttr('name',self.label)    
          # for caching  
          #self.setAttr('title',self.category)
          # for caching wikipedia index
          #self.setAttr('title',self.wiki_id)
          
          if IS_SAS_USED==True or IS_QUERY_GRAPH_USED==True or IS_TSWE_USED==True:
             self.update_categories(mongoObj)
          #self.dict_obj['contents']=List_Term_Object(self.contents,True,' ',mongoObj,w2vmodel)
          #self.dict_obj['stemmed_contents']=List_Term_Object(self.stemmed_contents,True,' ',mongoObj,w2vmodel)
          #self.dict_obj['label']=List_Term_Object(self.label,False,' ',mongoObj,w2vmodel)
          self.update_term_freq(docid,USED_CONTENT_FIELD,lucene_obj)
          self.length=sum(self.term_freq.values())
          self.update_term_freqs(docid,lucene_obj)
          
          #for idf in entity.iterator():
              #self.update_bigrams(idf.name(),idf.stringValue())
          #print (str(self.dict_obj[USED_CONTENT_FIELD].term))
          #print (str(self.term_freq))
          
          if IS_FACTOID_USED==True: 
             self.update_factoid(mongoObj,w2vmodel)
          if IS_MENTIONED_ENTITY_USED==True:
             self.update_mentioned_entity(mongoObj)
          
          if IS_QUERY_GRAPH_USED==True:
             self.update_literal_properties(mongoObj)
          
          if IS_WIKI_DOC_TREE_USED==True:
             wiki_id=findOneDBEntry(mongoObj.conn_page_id,'uri',self.uri,'wiki_id')
             article=findOneDBEntry(mongoObj.conn_wiki_aws,'wiki_id',wiki_id,'content')
             if article is not None:
                self.wiki_doc_tree=Doc_Tree_Object(article)
                self.wiki_doc_tree.title=self.wiki_id
          
      def update_term_freq(self,docid,field,lucene_obj):
          self.term_freq=lucene_obj.get_term_freq(docid,field,False)
          
      def update_term_freqs(self,docid,lucene_obj):
          self.term_freqs={}
          self.lengths={}
          for f in LIST_F:
              try:
                self.term_freqs[f]=lucene_obj.get_term_freq(docid,f,False)
                self.lengths[f]=sum(self.term_freqs[f].values())
              except:
                self.term_freqs[f]={}
                self.lengths[f]=0
          if LIST_F[0].find('stemmed')>-1:
             self.term_freqs['stemmed_catchall']=lucene_obj.get_term_freq(docid,'stemmed_catchall',False)
             self.lengths['stemmed_catchall']=sum(self.term_freqs['stemmed_catchall'].values()) 
          else:
             self.term_freqs['catchall']=lucene_obj.get_term_freq(docid,'catchall',False)
             self.lengths['catchall']=sum(self.term_freqs['catchall'].values())              
          
      def update_categories(self,mongoObj):
          conn=mongoObj.conn_acs if TAXONOMY=='Wikipedia' else mongoObj.conn_it
          field='categories' if TAXONOMY=='Wikipedia' else 'type'
          if conn==None:
             return
          item=conn.find_one({'uri':self.uri})
          if item is None:
             self.categories=[]
             return
          if TAXONOMY=='Wikipedia':
             self.categories=item[field].split('|')
          else:
             temp=item[field]
             pos=temp.rfind('/')
             self.categories=[temp[pos+1:-1]]
             #print (self.categories)
          
      def update_bigrams(self,field,value):
          if self.bigrams==None:
             self.bigrams={}
          for bigram in ngrams(value.split(),2):
              if (field,bigram) not in self.bigrams:
                 self.bigrams[(field,bigram)]=0
              self.bigrams[(field,bigram)]+=1          
              
      def update_factoid(self,mongoObj,w2vmodel):
          self.factoid_obj_list=[]
          for item in mongoObj.conn_wiki_cluster.find({'uri':self.uri}):
              value=item['contents'] if USED_QUERY_VERSION=='raw_query' else item['stemmed_contents']
              self.factoid_obj_list.append(List_Term_Object(value,True,' ',mongoObj,w2vmodel))
          #for obj in self.factoid_obj_list:
              #print str(obj.term)

      def update_literal_properties(self,mongoObj):
          self.literal_properties={}
          list_item=mongoObj.conn_mbpl.find({'uri':self.uri})
          if list_item is None:
             return
             
          for item in list_item:
              value_type=item['value_type']
              if value_type=='NONE':
                 continue
              value_str=item['value']
              property=item['property']
              value=None
              if value_type.find('Integer')>-1:
                 value=int(value_str)
              elif value_type.find('gYear')>-1:
                   try:
                      value=int(value_str) 
                   except:
                      value=int(value_str.split('-')[0])
              else:
                 continue
              #print ('add entity:%s  prop:%s value:%d'%(self.title,property,value))
              self.literal_properties[property]=(value,value_type)