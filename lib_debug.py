def interact(conf_paras):
    id_process=0
    lucene_obj=Lucene_Object(conf_paras.LUCENE_INDEX_DIR,'BM25',False)
    lucene_balog=Lucene_Object('E:\\mmapDirectory\\index7_stopped','BM25',True)
    
    # initialize mongodb client
    mongoObj=Mongo_Object('localhost',conf_paras.mongo_port)
      
    # initialize word2vec
    print 'id=%d  load word2vec model'%(id_process)
    w2vmodel = gensim.models.KeyedVectors.load_word2vec_format(conf_paras.PATH_WORD2VEC, binary=True)
    print 'id=%d  finish loading word2vec model'%(id_process)
    
    # search
    # data structure
    structure=Structure_Object(conf_paras)
    lucene_category=Lucene_Object(conf_paras.LUCENE_CATEGORY_INDEX_DIR,'BM25',True)
    lucene_category.updateBoostList(structure.queryBoost)
    
    query=''
    queryObj=None
    print 'finish initializing'
    while True:
          print '===================================================='
          print '1: set a query'
          print '2: query finding mode'
          print '3: entity checking mode, given query'
          print 'exit:exit interactive mode'
          
          sign=raw_input().strip()
          
          if sign=='1':
             print 'input a query'
             query=raw_input().strip()
             print 'current query=%s'%(query)
             queryObj=Query_Object(query,mongoObj,w2vmodel,lucene_category,True)
             
          elif sign=='2':
             if len(query.strip())==0:
                print 'you need to type a query first!'
             else:
                print 'begin query finding mode'
                docs=(lucene_balog.retrieve(cleanSentence(query),'contents',hitsPerPage,None))[0]
                # build query object for computeScore
                
                for d_temp in docs:
                    title=d_temp['id'][9:-1]
                    d=findEntityDocFromIndex(title,lucene_obj.getSecondarySearcher())
                    if d is not None:      
                       obj=createEntityObject(d,mongoObj,w2vmodel,FROM_INDEX,structure)               
                createGraph(queryObj,mongoObj,w2vmodel,lucene_obj.getSecondarySearcher(),structure,conf_paras)
                scoreList=systemSolver(id_process,structure,conf_paras)
                candidates=[]
                for j in range(len(scoreList)):
                    candidates.append((scoreList[j],j,structure.ID2entity[j]))
                candidates.sort(key=lambda pair:pair[0],reverse=True)      
                for rank in range(min(20,len(candidates))):
                    item=candidates[rank]
                    title='<dbpedia:%s>' %(item[2])
                    res_line="%s\t%s\t%f\n" %(query[0],title,item[0])
                    print res_line
                    
          elif sign=='3':
             if len(query.strip())==0:
                print 'you need to type a query first!'
             else:
                while True:
                      print 'type a entity you want to investigate, to quit input \'exit\' '     
                      title=raw_input().strip()
                      if title=='exit':
                         break
                      d=findEntityDocFromIndex(title,lucene_obj.getSecondarySearcher())                       
                      entityObj=createEntityObject(d,mongoObj,w2vmodel,FROM_INDEX,structure)
                      score=computeScore(queryObj,entityObj,mongoObj,w2vmodel,structure,conf_paras,lucene_obj)
          elif sign=='clear':
             os.system('clear')          
          elif sign=='exit':
             print 'bye'
             break
             
          else:
             print 'wrong command'