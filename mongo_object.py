import pymongo

class Mongo_Object(object):
      client = None
      db = None
      conn_wiki = None
      conn_acs = None # article category sentence
      conn_it = None # DBpedia instance types
      conn_wiki_aws = None # article_with_section
      conn_page_id = None
      conn_qe = None # query_entities
      conn_mbpo = None # mapping_based_properties_object
      conn_mbpl = None # mapping_based_properties_literal
      conn_qp = None # query predicates
      conn_bfvc = None # best_field_value_category
      conn_tswe = None # type-specific word embedding
      
      def __init__(self,hostname,port):
          self.client = pymongo.MongoClient(hostname,port)
          self.db = (self.client).wiki2015
          self.conn_wiki_aws=self.db['wiki_article_contents_with_section_clean']
          self.conn_page_id=self.db['page_id']
          self.conn_acs=self.db['article_categories']
          self.conn_it=self.db['instance_types']
          self.conn_qe=self.db['query_entities']
          self.conn_mbpo=self.db['mapping_based_properties_object']
          self.conn_mbpl=self.db['mapping_based_properties_literal']
          self.conn_qp=self.db['query_predicates_v2']
          self.conn_bfvc=self.db['best_field_value_category']
          self.conn_tswe=self.db['type_specific_word_embedding']
          self.conn_cache_tf=self.db['conn_cache_tf']
          
      def __del__(self):
          (self.client).close()