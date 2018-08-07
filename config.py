# coding=utf-8
# global parameter
import platform

SYSTEM_FLAG=platform.system()
DATA_VERSION = 2015
hitsPerPage = 500
NUM_PROCESS= 4
FROM_INDEX=0
FROM_DB=1
ENTITY_EXIST=False
ENTITY_NOT_EXIST=True
SEPERATE_CHAR_SUBQUERY='|'
CHOICE_WEIGHT='weight' # 'weight' or 'similarity'

NEGATIVE_INFINITY=-99999999

MODEL_NAME='mlm-all'
MAX_CHAIN_LENGTH=2

WORD_EMBEDDING_TYPE='NONE'

IS_TSWE_USED=False  # type-specific word embedding
IS_WIKI_DOC_TREE_USED=False # for wiki doc tree
IS_SAS_USED=True
IS_ELR_USED=False
IS_QUERY_GRAPH_USED=False

BOUND_SIM=0.97

# for bigram related operation
IS_BIGRAM_CACHE_USED=False
if MODEL_NAME.find('sdm')>-1:
   IS_BIGRAM_CACHE_USED=True

# for MLM model
MLMtc_FIELD_WEIGHTS={'stemmed_names':0.2,'stemmed_catchall':0.8}

# for FSDM model
LAMBDA_T=0.8
LAMBDA_O=0.1
LAMBDA_U=0.1

# for structure-aware smoothing
SAS_MAX_ARTICLE_PER_CAT=100
SAS_MODE='BOTTOMUP'
TAXONOMY='Wikipedia'  # Wikipedia or DBpedia
LIMIT_SAS_PATH_LENGTH=1
# 10,20
TOP_CATEGORY_NUM=10
# 30
TOP_PATH_NUM_PER_CAT=500
ALPHA_SAS=0.75


# for doc smoothing
LIMIT_D_PATH_LENGTH=10
ALPHA_D=0.8

# for factoid model
IS_FACTOID_MODEL_USED=False

# for Query_Object
USED_QUERY_VERSION='stemmed_raw_query'
IS_STOPWORD_REMOVED=True
IS_SUBQUERY_USED=False
IS_QUERY_CATEGORY_USED=False
IS_QUERY_CATEGORY_BOOSTLIST_USED=False
NUM_QUERY_CATEGORY_RETRIEVED=3

if CHOICE_WEIGHT=='similarity':
   IS_MENTIONED_ENTITY_USED=True
   IS_ENTITY_VECTOR_USED=True
   IS_RELATION_VECTOR_USED=True
   
if USED_QUERY_VERSION=='raw_query':
   USED_CONTENT_FIELD='catchall'
   LIST_F=['names','attributes','categories','similar_entities','related_entities']
   if MODEL_NAME=='mlm-tc':
      LIST_F=['names','catchall']
   elif MODEL_NAME=='sdm':
      LIST_F=['catchall']
elif USED_QUERY_VERSION=='stemmed_raw_query':
     USED_CONTENT_FIELD='stemmed_catchall' 
     LIST_F=['stemmed_names','stemmed_attributes','stemmed_categories','stemmed_similar_entities','stemmed_related_entities','stemmed_wikipedia']
     if MODEL_NAME=='mlm-tc':
        LIST_F=['stemmed_names','stemmed_catchall']
     elif MODEL_NAME=='sdm':
        LIST_F=['stemmed_catchall']
else:
     print ('Wrong query version !')
     USED_QUERY_VERSION='raw_query'
     USED_CONTENT_FIELD='catchall'
NO_SMOOTHING_LIST=[]
#NO_SMOOTHING_LIST=['stemmed_categories','stemmed_related_entities']
     
# for Entity_Object
IS_FACTOID_USED=False
if IS_FACTOID_MODEL_USED==True:
   IS_FACTOID_USED=True
IS_MENTIONED_ENTITY_USED=False
