# -*- coding: utf-8 -*-
# mcl_clustering.mcl()
# average mentioned entity score  otherwise will be dominated by keyword matching case

import os, sys
#from franges import drange

from query_object import Query_Object
from entity_object import Entity_Object
from mongo_object import Mongo_Object
from structure_object import Structure_Object
from lucene_object import Lucene_Object
from list_term_object import List_Term_Object
from lib_process import *
from lib_metric import *
from config import *
from config_object import *

import lucene
from java.io import File
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader, Term
from org.apache.lucene.store import MMapDirectory
from org.apache.lucene.queryparser.classic import QueryParserBase, ParseException, QueryParser, MultiFieldQueryParser
from org.apache.lucene.search import IndexSearcher, Query, ScoreDoc, TopScoreDocCollector, TermQuery, TermRangeQuery
from org.apache.lucene.search.similarities import BM25Similarity

from queue import Queue
import heapq
       
def read_query(queries,conf_paras):
    src = open(conf_paras.QUERY_FILEPATH,'r',encoding='utf-8')
    for line in src.readlines():
        list = line.strip().split('\t')
        #queries.append((list[0],list[1],list[2],list[3])) # raw_ID,querystr(for w2v mark ngram),raw merge query, original query
        queries.append((list[0],list[1],list[2])) # query_id, clusterd query, raw query  

def createEntityObject(d_pair,flag,structure,lucene_obj):
    #d_pair:(document,docid)
    d=d_pair[0]
    title=d.get('title')

    entityObjects=structure.entityObjects
    if title not in entityObjects:
       entityObj=Entity_Object()
       entityObj.updateFromIndex(d_pair,structure.mongoObj,structure.w2vmodel,lucene_obj)
       entityObj.update_categories(structure.mongoObj)
       entityObjects[title]=entityObj
    structure.currentEntity.add(title)
    return entityObjects[title]
             
def handle_process(id_process,queries,RES_STORE_PATH,conf_paras):   
    structure=Structure_Object(conf_paras,id_process)
    lucene_handler={}
    lucene_handler['first_pass']=Lucene_Object(conf_paras.LUCENE_INDEX_DIR,'BM25',False,False,structure.mongoObj)

    hitsperpage=500
    set_types=set()
    for i in range(len(queries)):
        lucene_obj=lucene_handler['first_pass']
        # build query object for computeScore
        queryObj=Query_Object(queries[i],structure,lucene_handler,False)
        querystr=queryObj.querystr   # no stemming may encourter zero candidates if field contents has stemming
        docs=lucene_obj.retrieve(querystr,USED_CONTENT_FIELD,hitsperpage)
        
        # initialize duplicate remover and score record
        structure.clear()     
        # find candidate results after 1st round filter
        # d_pair:(document,docid)
        for d_pair in docs:
            d=d_pair[0]
            if d is None:
               continue
            uri,title=d['uri'],d['title']
            if title in structure.currentEntity:
               continue    
            obj=createEntityObject(d_pair,FROM_INDEX,structure,lucene_obj)  
        
        
        for entity in structure.currentEntity:
            # title
            for cat in structure.entityObjects[entity].categories[:TOP_CATEGORY_NUM]:
                set_types.add(cat)
    
    #for type in set_types:
        #print (type)
    print (len(set_types))
    with open('entity_type_list_%d_QALD2_v2.txt'%(hitsperpage),'w',encoding='utf-8') as dest:
         for type in set_types:
             dest.write(type+'\n')
    
def main(conf_paras):
    system_flag=conf_paras.system_flag

    # read queries
    queries=[]
    read_query(queries,conf_paras)
    cnt_query=len(queries)

    handle_process(0,queries,'',conf_paras)

if __name__ == '__main__':
   conf_paras=Config_Object()
   main(conf_paras)
