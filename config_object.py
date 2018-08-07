# -*- coding: utf-8 -*-
from config import *
import os

class Config_Object(object):

      system_flag=None    
      mongo_port=58903
      
      def __init__(self):
          self.system_flag=SYSTEM_FLAG
          self.LUCENE_INDEX_DIR=os.path.join('mmapDirectory','dbpedia_v3_FSDM3')
          self.LUCENE_INDEX_WIKI_DIR=os.path.join('mmapDirectory','index_wikipedia_2015')
          if TAXONOMY=='Wikipedia':
             self.LUCENE_INDEX_CATEGORY_CORPUS=os.path.join('mmapDirectory','category_corpus_dbpedia201510_top10_fsdm3')
          else:
             self.LUCENE_INDEX_CATEGORY_CORPUS=os.path.join('mmapDirectory','category_corpus_ontology_dbpedia201510_notop_fsdm3')

          self.LUCENE_INDEX_URI=os.path.join('mmapDirectory','dbpedia_uri_v1')
          
          #queries_all_v2.txt
          self.QUERY_FILEPATH=os.path.join('query','simple_cluster','queries_all_v2.txt')
          self.PATH_GROUNDTRUTH=os.path.join('qrels-v2.txt')
          
          if TAXONOMY=='Wikipedia':
             self.PATH_CATEGORY_DAG='category_dag_dbpedia_top10.pkl.gz'
          else:
             self.PATH_CATEGORY_DAG='category_dag_dbpedia_ontology_top10.pkl.gz'
          
          if WORD_EMBEDDING_TYPE=='WORD2VEC':
            self.PATH_WORD2VEC=os.path.join('modified_w2v','wiki-201510-dbpedia-source-stemmed-sentence','wiki-201510-dbpedia-source-stemmed-sentence.bin.vector')
          else:
            self.PATH_WORD2VEC=os.path.join('fasttext','wiki-201510-dbpedia-source-stemmed-sentence-fasttext','wiki-201510-dbpedia-source-stemmed-sentence-fasttext.model')
          #self.PATH_WORD2VEC=os.path.join('modified_w2v','wiki-201510-dbpedia-source-stemmed-sentence','wiki-201510-dbpedia-source-stemmed-sentence.bin.vector')
          '''
          if IS_TSWE_USED==True:
             if self.QUERY_FILEPATH.find('INEX')>-1:
                filename='tswe_INEX_LD_v2.pkl.gz'
             elif self.QUERY_FILEPATH.find('QALD2')>-1:
                filename='tswe_QALD2_v2.pkl.gz'
             elif self.QUERY_FILEPATH.find('ListSearch')>-1:
                filename='tswe_ListSearch_v2.pkl.gz'
             elif self.QUERY_FILEPATH.find('SemSearch')>-1:
                filename='tswe_SemSearch_ES_v2.pkl.gz'
             self.PATH_TSWE=os.path.join('type_specific_word_embedding',filename)
          '''
